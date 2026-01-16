"""
Cliente WebSocket adaptado a tu pipeline existente
"""

import asyncio
import websockets
import json
import base64
from datetime import datetime
import cv2
import numpy as np
from typing import Optional, Dict, Any


class WSClient:
    """Cliente WebSocket para comunicarse con el backend web"""
    
    def __init__(self, backend_url: str, camera_id: str):
        """
        Args:
            backend_url: URL base del backend (ej: "ws://localhost:8000")
            camera_id: ID único de esta cámara
        """
        self.backend_url = f"{backend_url}/ws/camera/{camera_id}"
        self.camera_id = camera_id
        self.websocket = None
        self.threshold = 0.7
        self.is_connected = False
    
    async def connect(self):
        """Conectar al backend WebSocket"""
        try:
            self.websocket = await websockets.connect(self.backend_url)
            self.is_connected = True
            print(f"✓ Conectado al backend: {self.backend_url}")
            
            # Escuchar mensajes del backend en paralelo
            asyncio.create_task(self._listen_backend())
            
            return True
            
        except Exception as e:
            print(f"✗ Error conectando al backend: {e}")
            print(f"  Asegúrate de que el backend esté corriendo en {self.backend_url}")
            self.is_connected = False
            return False
    
    async def _listen_backend(self):
        """Escuchar actualizaciones del backend (threshold, nuevos familiares, etc.)"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data["type"] == "config_update":
                    self.threshold = data["payload"]["threshold"]
                    print(f"⚙️  Threshold actualizado desde backend: {self.threshold}")
                    
                elif data["type"] == "update_family":
                    print(f"📥 Actualización de familia recibida")
                    # Aquí podrías actualizar tu storage.npz si quieres
                    
        except websockets.exceptions.ConnectionClosed:
            print("✗ Conexión con backend cerrada")
            self.is_connected = False
        except Exception as e:
            print(f"✗ Error escuchando backend: {e}")
            self.is_connected = False
    
    async def send_event(self, event_type: str, data: Dict[str, Any]):
        """
        Enviar evento al backend (DEPRECADO - usa send_detection en su lugar)
        
        Args:
            event_type: Tipo de evento ("detection", etc.)
            data: Datos del evento
        """
        if not self.is_connected:
            return
        
        try:
            # Convertir a formato que espera el backend
            label = data.get("label")
            name = data.get("name")
            similarity = data.get("similarity", 0.0)
            
            # En tu código: similarity es de 0 a 1 (más alto = más similar)
            # El backend espera: distance (más bajo = más similar)
            # Convertir: distance = 1 - similarity
            distance = 1.0 - similarity
            
            is_family = label == "known"
            
            await self.send_detection(
                frame=None,  # Si no tienes el frame aquí, pasamos None
                person_name=name,
                is_family=is_family,
                distance=distance
            )
            
        except Exception as e:
            print(f"✗ Error enviando evento: {e}")
    
    async def send_detection(self, frame: Optional[np.ndarray], 
                            person_name: Optional[str], 
                            is_family: bool, 
                            distance: float):
        """
        Enviar detección al backend en el formato correcto
        
        Args:
            frame: Frame de la cámara (opcional, puede ser None)
            person_name: Nombre de la persona (None si es desconocido)
            is_family: True si es familia, False si es intruso
            distance: Distancia/diferencia (0 = idéntico, más alto = más diferente)
        """
        if not self.is_connected:
            return
        
        try:
            # Convertir frame a base64 si existe
            frame_base64 = None
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            detection_data = {
                "type": "detection",
                "payload": {
                    "timestamp": datetime.now().isoformat(),
                    "detected": True,
                    "person_name": person_name,
                    "is_family": is_family,
                    "distance": float(distance),
                    "frame": frame_base64
                }
            }
            
            await self.websocket.send(json.dumps(detection_data))
            
            # Log
            status = f"✓ {person_name}" if is_family else "⚠️ INTRUSO"
            print(f"📤 Detección enviada: {status} (dist: {distance:.3f})")
            
        except Exception as e:
            print(f"✗ Error enviando detección: {e}")
    
    async def send_frame(self, frame: np.ndarray):
        """
        Enviar frame para streaming de video en tiempo real
        
        Args:
            frame: Frame de la cámara (numpy array BGR)
        """
        if not self.is_connected or frame is None:
            return
        
        try:
            # Comprimir frame (calidad reducida para streaming)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_data = {
                "type": "frame",
                "frame": frame_base64
            }
            
            await self.websocket.send(json.dumps(frame_data))
            
        except Exception as e:
            # No imprimir errores de streaming para no saturar logs
            pass
    
    async def close(self):
        """Cerrar conexión"""
        if self.websocket:
            await self.websocket.close()
            print("✓ Conexión WebSocket cerrada")