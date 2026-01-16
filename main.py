from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File. HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import base64
import uuid
import asyncio
from collections import defaultdict
from pathlib import Path
import shutil

from sys import argv, exit
import os
import cv2
import numpy as np

from face_detection import FaceDetector
from embeddings import Embeddings


from utils import load_db

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import uuid
import asyncio
from collections import defaultdict
from pathlib import Path
import shutil

# ==================== APP ====================
app = FastAPI(title="Smart EdgeAI Security API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK para red local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images" / "family"
FAMILY_JSON = DATA_DIR / "family.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Calculate embedding database
def save_db(npz_path: str, names, embs):
    np.savez(npz_path, names=np.array(names), embs=embs.astype(np.float32))


def calculate_embeddings_database(label, image_paths):
    # print("  python3 enroll.py <face_detector.tflite> <embed_model.tflite> <storage.npz> <label> <img1> [img2 ... imgN]")

    detector_model_path = ""
    embed_model_path = ""
    db_path = ""
    label = label
    image_paths = image_paths

    detector = FaceDetector(detector_model_path)
    embedder = Embeddings(embed_model_path)

    # Load existing DB (if any)
    names, embs = load_db(db_path)
    new_embs = []

    try:
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[SKIP] Could not read image: {p}")
                continue

            try:
                crop = detector.detect_and_crop_largest_face_tasks(img, expand=0.2)
                emb = embedder.get_embedding(crop, normalization="arcface")  # already L2-normalized
                # the normalization is done inside get_embedding
                new_embs.append(emb)
                print(f"[OK] Added embedding for '{label}' from: {p}")
            except Exception as e:
                print(f"[SKIP] {p} -> {e}")

        if not new_embs:
            print("No embeddings were added. Nothing to save.")
            return

        new_embs = np.vstack(new_embs).astype(np.float32)  # (K,D)

        if embs is None:
            # first time creating DB
            embs = new_embs
            names = [label] * new_embs.shape[0]
        else:
            # append to existing
            embs = np.vstack([embs, new_embs]).astype(np.float32)
            names.extend([label] * new_embs.shape[0])

        save_db(db_path, names, embs)
        print(f"Saved {len(new_embs)} new embeddings to: {db_path}")
        print(f"DB now contains {len(names)} embeddings total.")

    finally:
        detector.close()


# ==================== MODELOS ====================
class FamilyMember(BaseModel):
    id: str
    name: str
    image_path: str
    added_date: datetime

class AccessEvent(BaseModel):
    id: str
    timestamp: datetime
    person_name: Optional[str]
    is_family: bool
    confidence: float
    image_snapshot: Optional[str] = None

class SystemConfig(BaseModel):
    threshold: float = 0.40
    enable_alerts: bool = True

class DetectionResult(BaseModel):
    timestamp: datetime
    detected: bool
    person_name: Optional[str]
    is_family: bool
    distance: float
    frame: Optional[str] = None

# ==================== STORAGE ====================
family_members: List[FamilyMember] = []
access_events: List[AccessEvent] = []
system_config = SystemConfig()
connected_cameras = {}
connected_clients = set()

# ==================== JSON HELPERS ====================
def load_family():
    global family_members
    if FAMILY_JSON.exists():
        with open(FAMILY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            family_members = [FamilyMember(**m) for m in data]
    else:
        family_members = []

def save_family():
    with open(FAMILY_JSON, "w", encoding="utf-8") as f:
        json.dump([m.dict() for m in family_members], f, indent=2, default=str)

# ==================== WEBSOCKET HELPERS ====================
async def broadcast_to_clients(message: dict):
    dead = set()
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except:
            dead.add(ws)
    connected_clients.difference_update(dead)

async def notify_cameras(message: dict):
    for ws in connected_cameras.values():
        try:
            await ws.send_json(message)
        except:
            pass

# ==================== FAMILY API ====================
@app.get("/api/family", response_model=List[FamilyMember])
async def get_family():
    return family_members

@app.post("/api/family", response_model=FamilyMember)
async def add_family(name: str, image: UploadFile = File(...)):
    member_id = str(uuid.uuid4())
    suffix = Path(image.filename).suffix or ".jpg"
    image_path = IMAGES_DIR / f"{member_id}{suffix}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    member = FamilyMember(
        id=member_id,
        name=name,
        image_path=str(image_path),
        added_date=datetime.now()
    )

    family_members.append(member)
    save_family()

    await broadcast_to_clients({
        "type": "update_family",
        "family_members": [m.dict() for m in family_members]
    })

    await notify_cameras({
        "type": "update_family",
        "action": "add",
        "member": member.dict()
    })

    return member

@app.delete("/api/family/{member_id}")
async def delete_family(member_id: str):
    global family_members
    member = next((m for m in family_members if m.id == member_id), None)
    if not member:
        raise HTTPException(404, "Member not found")

    if Path(member.image_path).exists():
        Path(member.image_path).unlink()

    family_members = [m for m in family_members if m.id != member_id]
    save_family()

    await broadcast_to_clients({
        "type": "update_family",
        "family_members": [m.dict() for m in family_members]
    })

    await notify_cameras({
        "type": "update_family",
        "action": "delete",
        "member_id": member_id
    })

    return {"ok": True}

# ==================== EVENTS ====================
@app.get("/api/events", response_model=List[AccessEvent])
async def get_events(limit: int = 100):
    return access_events[-limit:]

@app.get("/api/events/stats")
async def get_stats():
    total = len(access_events)
    family = sum(e.is_family for e in access_events)
    daily = defaultdict(int)
    for e in access_events:
        daily[e.timestamp.date().isoformat()] += 1

    return {
        "total": total,
        "family": family,
        "intruders": total - family,
        "daily": dict(daily)
    }

# ==================== CONFIG ====================
@app.get("/api/config", response_model=SystemConfig)
async def get_config():
    return system_config

@app.put("/api/config", response_model=SystemConfig)
async def update_config(cfg: SystemConfig):
    global system_config
    system_config = cfg
    await notify_cameras({"type": "config_update", "config": cfg.dict()})
    return system_config

# ==================== WEBSOCKETS ====================
@app.websocket("/ws/camera/{camera_id}")
async def camera_ws(ws: WebSocket, camera_id: str):
    await ws.accept()
    connected_cameras[camera_id] = ws
    try:
        while True:
            data = await ws.receive_json()
            if data["type"] == "detection":
                det = DetectionResult(**data["payload"])
                event = AccessEvent(
                    id=str(uuid.uuid4()),
                    timestamp=det.timestamp,
                    person_name=det.person_name,
                    is_family=det.is_family,
                    confidence=1.0 - det.distance,
                    image_snapshot=det.frame if not det.is_family else None
                )
                access_events.append(event)
                await broadcast_to_clients({"type": "new_event", "event": event.dict()})
    except WebSocketDisconnect:
        connected_cameras.pop(camera_id, None)

@app.websocket("/ws/client")
async def client_ws(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    try:
        await ws.send_json({
            "type": "init",
            "family_members": [m.dict() for m in family_members],
            "config": system_config.dict()
        })
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(ws)

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup():
    load_family()

# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
