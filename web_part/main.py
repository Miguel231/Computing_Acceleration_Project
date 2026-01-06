from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import base64
import asyncio
from collections import defaultdict
import uuid

app = FastAPI(title="Smart EdgeAI Security API")

# CORS para permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar el dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELOS DE DATOS ====================

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
    image_snapshot: Optional[str] = None  # Base64 encoded image

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
    frame: str  # Base64 encoded frame

# ==================== ALMACENAMIENTO EN MEMORIA ====================

family_members: List[FamilyMember] = []
access_events: List[AccessEvent] = []
system_config = SystemConfig()
connected_cameras: dict = {}  # {camera_id: websocket}
connected_clients: set = set()  # WebSockets de clientes frontend

# ==================== ENDPOINTS DE GESTIÓN DE FAMILIA ====================

@app.get("/api/family", response_model=List[FamilyMember])
async def get_family_members():
    """Obtener lista de familiares registrados"""
    return family_members

@app.post("/api/family", response_model=FamilyMember)
async def add_family_member(name: str, image: UploadFile = File(...)):
    """Añadir un nuevo miembro de la familia"""
    member_id = str(uuid.uuid4())
    
    # Guardar la imagen (en producción, guardar en disco o S3)
    image_data = await image.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    new_member = FamilyMember(
        id=member_id,
        name=name,
        image_path=f"data:image/jpeg;base64,{image_base64}",
        added_date=datetime.now()
    )
    
    family_members.append(new_member)
    
    # Notificar a las cámaras conectadas para actualizar embeddings
    await notify_cameras("update_family", {"action": "add", "member": new_member.dict()})
    
    return new_member

@app.delete("/api/family/{member_id}")
async def delete_family_member(member_id: str):
    """Eliminar un miembro de la familia"""
    global family_members
    family_members = [m for m in family_members if m.id != member_id]
    
    # Notificar a las cámaras
    await notify_cameras("update_family", {"action": "delete", "member_id": member_id})
    
    return {"message": "Member deleted successfully"}

# ==================== ENDPOINTS DE EVENTOS ====================

@app.get("/api/events", response_model=List[AccessEvent])
async def get_access_events(limit: int = 100):
    """Obtener historial de eventos de acceso"""
    return access_events[-limit:]

@app.get("/api/events/stats")
async def get_event_statistics():
    """Obtener estadísticas de accesos"""
    total = len(access_events)
    family_access = sum(1 for e in access_events if e.is_family)
    intruders = total - family_access
    
    # Accesos por día (últimos 7 días)
    daily_stats = defaultdict(int)
    for event in access_events[-200:]:
        day = event.timestamp.date().isoformat()
        daily_stats[day] += 1
    
    return {
        "total_events": total,
        "family_access": family_access,
        "intruder_alerts": intruders,
        "daily_stats": dict(daily_stats)
    }

# ==================== CONFIGURACIÓN DEL SISTEMA ====================

@app.get("/api/config", response_model=SystemConfig)
async def get_config():
    """Obtener configuración actual del sistema"""
    return system_config

@app.put("/api/config", response_model=SystemConfig)
async def update_config(config: SystemConfig):
    """Actualizar configuración del sistema"""
    global system_config
    system_config = config
    
    # Notificar a las cámaras de los cambios
    await notify_cameras("config_update", config.dict())
    
    return system_config

# ==================== WEBSOCKET PARA CÁMARAS (RASPBERRY PI) ====================

@app.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    """WebSocket para que la Raspberry Pi envíe detecciones en tiempo real"""
    await websocket.accept()
    connected_cameras[camera_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "detection":
                # Procesar resultado de detección
                detection = DetectionResult(**data["payload"])
                
                # Crear evento de acceso
                event = AccessEvent(
                    id=str(uuid.uuid4()),
                    timestamp=detection.timestamp,
                    person_name=detection.person_name,
                    is_family=detection.is_family,
                    confidence=1.0 - detection.distance,  # Convertir distancia a confianza
                    image_snapshot=detection.frame if not detection.is_family else None
                )
                
                access_events.append(event)
                
                # Broadcast a todos los clientes conectados
                await broadcast_to_clients({
                    "type": "new_event",
                    "event": event.dict()
                })
                
            elif data["type"] == "frame":
                # Stream de video en tiempo real
                await broadcast_to_clients({
                    "type": "video_frame",
                    "camera_id": camera_id,
                    "frame": data["frame"]
                })
                
    except WebSocketDisconnect:
        del connected_cameras[camera_id]
        print(f"Camera {camera_id} disconnected")

# ==================== WEBSOCKET PARA CLIENTES FRONTEND ====================

@app.websocket("/ws/client")
async def client_websocket(websocket: WebSocket):
    """WebSocket para que el frontend reciba actualizaciones en tiempo real"""
    await websocket.accept()
    connected_clients.add(websocket)
    
    try:
        # Enviar configuración inicial
        await websocket.send_json({
            "type": "init",
            "config": system_config.dict(),
            "family_members": [m.dict() for m in family_members],
            "connected_cameras": list(connected_cameras.keys())
        })
        
        while True:
            # Mantener conexión abierta
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# ==================== FUNCIONES AUXILIARES ====================

async def notify_cameras(event_type: str, payload: dict):
    """Notificar a todas las cámaras conectadas"""
    for camera_ws in connected_cameras.values():
        try:
            await camera_ws.send_json({
                "type": event_type,
                "payload": payload
            })
        except:
            pass

async def broadcast_to_clients(message: dict):
    """Enviar mensaje a todos los clientes conectados"""
    disconnected = set()
    for client_ws in connected_clients:
        try:
            await client_ws.send_json(message)
        except:
            disconnected.add(client_ws)
    
    # Limpiar conexiones muertas
    connected_clients.difference_update(disconnected)

# ==================== ENDPOINT DE SALUD ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cameras_connected": len(connected_cameras),
        "clients_connected": len(connected_clients),
        "family_members": len(family_members),
        "total_events": len(access_events)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)