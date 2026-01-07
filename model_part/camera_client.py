"""
Cliente integrado para Raspberry Pi - Adaptado a vuestra estructura
Conecta vuestro sistema de detección facial con el backend web.
"""

import cv2
import asyncio
import websockets
import json
import base64
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np
import sys
import os

# ========== IMPORTAR VUESTROS MÓDULOS ==========
from model_part.face_detection import FaceDetector
from model_part.embeddings import Embeddings
# Si tenéis función de distancia en utils.py:
# from utils import calculate_distance


class IntegratedSecurityClient:
    """Cliente que integra detección facial con el backend web"""
    
    def __init__(self, 
                 backend_url: str = "ws://localhost:8000",
                 camera_id: str = "rpi_cam_001",
                 camera_index: int = 0):
        """
        Args:
            backend_url: URL del backend WebSocket
            camera_id: ID único de esta cámara
            camera_index: Índice de la cámara (0 = cámara por defecto)
        """
        self.backend_url = f"{backend_url}/ws/camera/{camera_id}"
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.websocket = None
        
        # Configuración del sistema
        self.threshold = 0.40
        self.family_embeddings: Dict[str, np.ndarray] = {}
        self.is_running = False
        
        print("🔧 Inicializando modelos...")
        
        # Inicializar vuestros modelos
        self.detector = FaceDetector()
        self.embedding_generator = EmbeddingGenerator()
        
        print("✓ Modelos inicializados (FaceDetector + EmbeddingGenerator)")
    
    async def connect(self):
        """Conectar al backend WebSocket"""
        try:
            self.websocket = await websockets.connect(self.backend_url)
            print(f"✓ Conectado al backend: {self.backend_url}")
            
            # Escuchar mensajes del backend en paralelo
            asyncio.create_task(self.listen_to_backend())
            
            return True
        except Exception as e:
            print(f"✗ Error de conexión: {e}")
            print(f"   Asegúrate de que el backend esté corriendo en: {self.backend_url}")
            return False
    
    async def listen_to_backend(self):
        """Escuchar comandos del backend"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data["type"] == "update_family":
                    await self.handle_family_update(data["payload"])
                    
                elif data["type"] == "config_update":
                    self.threshold = data["payload"]["threshold"]
                    print(f"✓ Threshold actualizado: {self.threshold}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("✗ Conexión con backend cerrada")
            self.is_running = False
        except Exception as e:
            print(f"✗ Error escuchando backend: {e}")
    
    async def handle_family_update(self, payload: dict):
        """Manejar actualizaciones de la base de datos familiar"""
        action = payload["action"]
        
        if action == "add":
            member = payload["member"]
            name = member["name"]
            
            try:
                # Extraer imagen base64
                image_b64 = member["image_path"].split(",")[1]
                
                # Decodificar imagen
                image_data = base64.b64decode(image_b64)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Calcular embedding usando vuestro EmbeddingGenerator
                embedding = self.embedding_generator.generate(image)
                
                # Guardar en diccionario
                self.family_embeddings[name] = embedding
                
                print(f"✓ Miembro añadido: {name} (embedding shape: {embedding.shape})")
                
            except Exception as e:
                print(f"✗ Error procesando miembro {name}: {e}")
            
        elif action == "delete":
            member_id = payload["member_id"]
            # Buscar y eliminar por ID (necesitarías mapear ID a nombre)
            # Por ahora, simplemente reportamos
            print(f"ℹ Solicitud de eliminar miembro: {member_id}")
    
    def calculate_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcular distancia euclidiana entre dos embeddings
        Si tenéis vuestra propia función en utils.py, usadla aquí
        """
        # Opción 1: Si tenéis función propia
        # return calculate_distance(emb1, emb2)
        
        # Opción 2: Distancia euclidiana estándar
        return np.linalg.norm(emb1 - emb2)
    
    def recognize_face(self, face_image: np.ndarray) -> tuple[Optional[str], float]:
        """
        Reconocer cara comparando con embeddings familiares
        
        Args:
            face_image: Imagen de la cara detectada
            
        Returns:
            (nombre, distancia) - nombre es None si es intruso
        """
        if len(self.family_embeddings) == 0:
            return None, float('inf')
        
        # Generar embedding de la cara detectada
        try:
            face_embedding = self.embedding_generator.generate(face_image)
        except Exception as e:
            print(f"✗ Error generando embedding: {e}")
            return None, float('inf')
        
        min_distance = float('inf')
        recognized_name = None
        
        # Comparar con cada familiar
        for name, family_embedding in self.family_embeddings.items():
            distance = self.calculate_distance(face_embedding, family_embedding)
            
            if distance < min_distance:
                min_distance = distance
                recognized_name = name
        
        # Determinar si es familiar basado en threshold
        if min_distance < self.threshold:
            return recognized_name, min_distance
        else:
            return None, min_distance
    
    async def send_detection(self, frame: np.ndarray, person_name: Optional[str], 
                            is_family: bool, distance: float):
        """Enviar resultado de detección al backend"""
        
        try:
            # Convertir frame a base64
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
            
        except Exception as e:
            print(f"✗ Error enviando detección: {e}")
    
    async def send_frame(self, frame: np.ndarray):
        """Enviar frame para streaming en tiempo real"""
        try:
            # Reducir calidad para streaming
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_data = {
                "type": "frame",
                "frame": frame_base64
            }
            
            await self.websocket.send(json.dumps(frame_data))
            
        except Exception as e:
            # No imprimir error aquí para no saturar logs
            pass
    
    async def run(self):
        """Loop principal de detección"""
        
        # Inicializar cámara
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print("✗ Error: No se puede abrir la cámara")
            print(f"   Intentando con índice {self.camera_index}")
            print("   Prueba cambiar el índice: 0, 1, 2...")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("✓ Cámara inicializada")
        print(f"📹 Resolución: {actual_width}x{actual_height}")
        print(f"🎯 Threshold: {self.threshold}")
        print(f"👨‍👩‍👧‍👦 Familiares registrados: {len(self.family_embeddings)}")
        print("\n🚀 Iniciando detección...")
        print("   Presiona Ctrl+C para detener\n")
        
        self.is_running = True
        frame_count = 0
        detection_interval = 10  # Detectar cada N frames (ajusta según performance)
        stream_interval = 5      # Stream cada N frames
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    print("✗ Error capturando frame")
                    await asyncio.sleep(0.1)
                    continue
                
                # ========== DETECCIÓN Y RECONOCIMIENTO ==========
                
                # Solo detectar cada N frames (optimización)
                if frame_count % detection_interval == 0:
                    
                    # 1. Detectar caras usando vuestro FaceDetector
                    faces = self.detector.detect(frame)
                    
                    if faces is not None and len(faces) > 0:
                        print(f"👤 {len(faces)} cara(s) detectada(s)")
                        
                        # Procesar cada cara detectada
                        for i, face in enumerate(faces):
                            try:
                                # Extraer región de la cara
                                # Ajusta según cómo devuelva tu FaceDetector
                                # Puede ser: face['bbox'], face['coordinates'], etc.
                                
                                # Si face es un diccionario con 'bbox':
                                if isinstance(face, dict) and 'bbox' in face:
                                    x, y, w, h = face['bbox']
                                    face_roi = frame[y:y+h, x:x+w]
                                # Si face es directamente la imagen ROI:
                                elif isinstance(face, np.ndarray):
                                    face_roi = face
                                else:
                                    # Usar frame completo si no sabemos el formato
                                    face_roi = frame
                                
                                # 2. Reconocer la cara
                                name, distance = self.recognize_face(face_roi)
                                is_family = name is not None
                                
                                # Log
                                if is_family:
                                    print(f"✓ Familiar reconocido: {name} (dist: {distance:.3f})")
                                else:
                                    print(f"⚠️  INTRUSO detectado (dist: {distance:.3f})")
                                
                                # 3. Enviar detección al backend
                                await self.send_detection(frame, name, is_family, distance)
                                
                            except Exception as e:
                                print(f"✗ Error procesando cara {i}: {e}")
                
                # Stream de video cada N frames
                if frame_count % stream_interval == 0:
                    await self.send_frame(frame)
                
                frame_count += 1
                
                # Delay para mantener ~30 FPS
                await asyncio.sleep(0.033)
                
        except KeyboardInterrupt:
            print("\n\n✓ Detención solicitada por usuario")
        except Exception as e:
            print(f"\n✗ Error en loop principal: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            cap.release()
            if self.websocket:
                await self.websocket.close()
            print("\n✓ Recursos liberados")


async def main():
    """Función principal"""
    
    print("=" * 60)
    print("   🔒 Smart EdgeAI Security - Integrated Client")
    print("=" * 60)
    print()
    
    # ========== CONFIGURACIÓN ==========
    # Cambia estos valores según tu setup:
    
    BACKEND_URL = "ws://localhost:8000"  
    # Si el backend está en otro PC: "ws://192.168.1.100:8000"
    
    CAMERA_ID = "rpi_cam_001"
    CAMERA_INDEX = 0  # 0 = cámara por defecto, prueba 1, 2... si no funciona
    
    # ====================================
    
    # Crear cliente
    client = IntegratedSecurityClient(
        backend_url=BACKEND_URL,
        camera_id=CAMERA_ID,
        camera_index=CAMERA_INDEX
    )
    
    # Conectar al backend
    print("🔌 Conectando al backend...")
    connected = await client.connect()
    
    if not connected:
        print("\n" + "="*60)
        print("  ❌ NO SE PUDO CONECTAR AL BACKEND")
        print("="*60)
        print("\nVerifica que:")
        print("  1. El backend esté corriendo:")
        print("     cd web_part && python main.py")
        print("  2. La URL sea correcta:")
        print(f"     {BACKEND_URL}")
        print("  3. No haya firewall bloqueando el puerto 8000")
        print()
        return
    
    print()
    # Ejecutar sistema
    await client.run()


if __name__ == "__main__":
    """
    Ejecutar en Raspberry Pi:
    
    1. Asegúrate de tener instalado:
       pip install opencv-python websockets numpy
    
    2. Ejecuta:
       python integrated_client.py
    
    3. Para detener: Ctrl+C
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Programa terminado por usuario")
    except Exception as e:
        print(f"\n✗ Error fatal: {e}")
        import traceback
        traceback.print_exc()