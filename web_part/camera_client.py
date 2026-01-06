"""
Cliente para Raspberry Pi que conecta vuestro sistema de detección 
con el backend FastAPI
"""

import cv2
import asyncio
import websockets
import json
import base64
from datetime import datetime
from typing import Optional
import numpy as np

# Importar vuestros módulos existentes
# from face_detection import FaceDetector
# from embeddings import EmbeddingGenerator
# from utils import calculate_distance

class CameraClient:
    def __init__(self, backend_url: str = "ws://localhost:8000", camera_id: str = "rpi_cam_001"):
        self.backend_url = f"{backend_url}/ws/camera/{camera_id}"
        self.camera_id = camera_id
        self.websocket = None
        self.threshold = 0.40
        self.family_embeddings = {}
        
        # Inicializar vuestros modelos
        # self.detector = FaceDetector()
        # self.embedding_generator = EmbeddingGenerator()
        
    async def connect(self):
        """Conectar al backend"""
        try:
            self.websocket = await websockets.connect(self.backend_url)
            print(f"✓ Connected to backend: {self.backend_url}")
            
            # Escuchar mensajes del backend en paralelo
            asyncio.create_task(self.listen_to_backend())
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise
    
    async def listen_to_backend(self):
        """Escuchar comandos del backend"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data["type"] == "update_family":
                    await self.update_family_embeddings(data["payload"])
                    
                elif data["type"] == "config_update":
                    self.threshold = data["payload"]["threshold"]
                    print(f"✓ Threshold updated to: {self.threshold}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("✗ Backend connection closed")
    
    async def update_family_embeddings(self, payload: dict):
        """Actualizar embeddings cuando se añade/elimina un familiar"""
        action = payload["action"]
        
        if action == "add":
            member = payload["member"]
            # Aquí deberíais calcular el embedding de la nueva imagen
            # image_data = base64.b64decode(member["image_path"].split(",")[1])
            # embedding = self.embedding_generator.generate(image_data)
            # self.family_embeddings[member["name"]] = embedding
            print(f"✓ Added family member: {member['name']}")
            
        elif action == "delete":
            member_id = payload["member_id"]
            # Eliminar del diccionario
            # del self.family_embeddings[member_id]
            print(f"✓ Removed family member: {member_id}")
    
    async def send_detection(self, frame: np.ndarray, detected_face: bool, 
                            person_name: Optional[str], is_family: bool, distance: float):
        """Enviar resultado de detección al backend"""
        
        # Convertir frame a base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        detection_data = {
            "type": "detection",
            "payload": {
                "timestamp": datetime.now().isoformat(),
                "detected": detected_face,
                "person_name": person_name,
                "is_family": is_family,
                "distance": distance,
                "frame": frame_base64
            }
        }
        
        await self.websocket.send(json.dumps(detection_data))
    
    async def send_frame(self, frame: np.ndarray):
        """Enviar frame de video para streaming en tiempo real"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        frame_data = {
            "type": "frame",
            "frame": frame_base64
        }
        
        await self.websocket.send(json.dumps(frame_data))
    
    async def run_detection_loop(self):
        """
        Loop principal de detección - Integrar con vuestro código existente
        """
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Starting detection loop...")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to grab frame")
                    break
                
                # ========== INTEGRAR VUESTRO CÓDIGO AQUÍ ==========
                
                # 1. Detectar caras en el frame
                # faces = self.detector.detect(frame)
                
                # 2. Para cada cara detectada, generar embedding
                # for face in faces:
                #     face_embedding = self.embedding_generator.generate(face)
                #     
                #     # 3. Comparar con embeddings de familiares
                #     min_distance = float('inf')
                #     recognized_person = None
                #     
                #     for name, family_embedding in self.family_embeddings.items():
                #         distance = calculate_distance(face_embedding, family_embedding)
                #         if distance < min_distance:
                #             min_distance = distance
                #             recognized_person = name
                #     
                #     # 4. Determinar si es familiar o intruso
                #     is_family = min_distance < self.threshold
                #     
                #     # 5. Enviar detección al backend
                #     await self.send_detection(
                #         frame=frame,
                #         detected_face=True,
                #         person_name=recognized_person if is_family else None,
                #         is_family=is_family,
                #         distance=min_distance
                #     )
                
                # ===================================================
                
                # Enviar frame cada 5 frames para streaming (reducir carga)
                if frame_count % 5 == 0:
                    await self.send_frame(frame)
                
                frame_count += 1
                
                # Pequeño delay para no saturar
                await asyncio.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n✓ Detection stopped by user")
        finally:
            cap.release()
            await self.websocket.close()


async def main():
    """Función principal"""
    # Configurar la URL de vuestro backend
    BACKEND_URL = "ws://localhost:8000"  # Cambiar a la IP del servidor
    CAMERA_ID = "rpi_cam_001"
    
    client = CameraClient(backend_url=BACKEND_URL, camera_id=CAMERA_ID)
    
    print("=" * 50)
    print("Smart EdgeAI Security - Camera Client")
    print("=" * 50)
    
    # Conectar al backend
    await client.connect()
    
    # Ejecutar loop de detección
    await client.run_detection_loop()


if __name__ == "__main__":
    # Ejecutar en Raspberry Pi:
    # python camera_client.py
    asyncio.run(main())