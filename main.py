from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
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

app = FastAPI(title="Smart EdgeAI Security API")

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, poner dominios reales
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    webhook_url: Optional[str] = None
    enable_alerts: bool = True
    save_intruder_images: bool = True

class DetectionResult(BaseModel):
    timestamp: datetime
    detected: bool
    person_name: Optional[str] = None
    is_family: bool
    distance: float
    frame: str

# ==================== RUTAS JSON ====================
LOCAL_JSON_PATH = Path("family_local.json")
RASPBERRY_JSON_PATH = Path("/path/to/raspberry/family_raspberry.json")  # Cambiar según tu setup

def load_family_json(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_family_json(path: Path, data: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str, indent=2)

def load_combined_family():
    local = load_family_json(LOCAL_JSON_PATH)
    raspberry = load_family_json(RASPBERRY_JSON_PATH)
    combined = {m["id"]: m for m in local}
    for m in raspberry:
        combined[m["id"]] = m
    return list(combined.values())

# ==================== ALMACENAMIENTO EN MEMORIA ====================
family_members: List[FamilyMember] = [FamilyMember(**m) for m in load_combined_family()]
access_events: List[AccessEvent] = []
system_config = SystemConfig()
connected_cameras: dict = {}
connected_clients: set = set()

# ==================== AUXILIARES ====================
async def notify_cameras(event_type: str, payload: dict):
    for ws in connected_cameras.values():
        try:
            await ws.send_json({"type": event_type, "payload": payload})
        except:
            pass

async def broadcast_to_clients(message: dict):
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    connected_clients.difference_update(disconnected)

async def sync_family_with_json():
    """Revisar cada X segundos la Raspberry y sincronizar"""
    while True:
        combined = load_combined_family()
        global family_members
        # Detectar cambios
        ids_in_memory = {m.id for m in family_members}
        ids_combined = {m["id"] for m in combined}
        
        if ids_in_memory != ids_combined:
            # Actualizar memoria
            family_members = [FamilyMember(**m) for m in combined]
            # Guardar JSON local
            save_family_json(LOCAL_JSON_PATH, [m.dict() for m in family_members])
            # Notificar a todos los clientes
            await broadcast_to_clients({
                "type": "update_family",
                "family_members": [m.dict() for m in family_members]
            })
        
        await asyncio.sleep(5)  # cada 5 segundos

# ==================== ENDPOINTS ====================
@app.get("/api/family", response_model=List[FamilyMember])
async def get_family_members():
    return family_members

@app.post("/api/family", response_model=FamilyMember)
async def add_family_member(name: str, image: UploadFile = File(...)):
    member_id = str(uuid.uuid4())
    image_data = await image.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    new_member = FamilyMember(
        id=member_id,
        name=name,
        image_path=f"data:image/jpeg;base64,{image_base64}",
        added_date=datetime.now()
    )

    family_members.append(new_member)
    # Guardar cambios en JSON local y opcional Raspberry
    save_family_json(LOCAL_JSON_PATH, [m.dict() for m in family_members])
    # save_family_json(RASPBERRY_JSON_PATH, [m.dict() for m in family_members])  # si tienes acceso

    # Notificar a clientes y cámaras
    await broadcast_to_clients({"type": "update_family", "family_members": [m.dict() for m in family_members]})
    await notify_cameras("update_family", {"action": "add", "member": new_member.dict()})

    return new_member

@app.delete("/api/family/{member_id}")
async def delete_family_member(member_id: str):
    global family_members
    family_members = [m for m in family_members if m.id != member_id]
    save_family_json(LOCAL_JSON_PATH, [m.dict() for m in family_members])
    await broadcast_to_clients({"type": "update_family", "family_members": [m.dict() for m in family_members]})
    await notify_cameras("update_family", {"action": "delete", "member_id": member_id})
    return {"message": "Member deleted successfully"}

@app.get("/api/events", response_model=List[AccessEvent])
async def get_access_events(limit: int = 100):
    return access_events[-limit:]

@app.get("/api/events/stats")
async def get_event_statistics():
    total = len(access_events)
    family_access = sum(1 for e in access_events if e.is_family)
    intruders = total - family_access
    daily_stats = defaultdict(int)
    for e in access_events[-200:]:
        day = e.timestamp.date().isoformat()
        daily_stats[day] += 1
    return {
        "total_events": total,
        "family_access": family_access,
        "intruder_alerts": intruders,
        "daily_stats": dict(daily_stats)
    }

@app.get("/api/config", response_model=SystemConfig)
async def get_config():
    return system_config

@app.put("/api/config", response_model=SystemConfig)
async def update_config(config: SystemConfig):
    global system_config
    system_config = config
    await notify_cameras("config_update", config.dict())
    return system_config

# ==================== WEBSOCKETS ====================
@app.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    connected_cameras[camera_id] = websocket
    try:
        while True:
            data = await websocket.receive_json()
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
            elif data["type"] == "frame":
                await broadcast_to_clients({"type": "video_frame", "camera_id": camera_id, "frame": data["frame"]})
    except WebSocketDisconnect:
        del connected_cameras[camera_id]

@app.websocket("/ws/client")
async def client_websocket(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        await websocket.send_json({
            "type": "init",
            "config": system_config.dict(),
            "family_members": [m.dict() for m in family_members],
            "connected_cameras": list(connected_cameras.keys())
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# ==================== HEALTH ====================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cameras_connected": len(connected_cameras),
        "clients_connected": len(connected_clients),
        "family_members": len(family_members),
        "total_events": len(access_events)
    }

# ==================== INICIAR BACKEND ====================
@app.on_event("startup")
async def startup_event():
    # Iniciar tarea de sincronización de familiares
    asyncio.create_task(sync_family_with_json())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
