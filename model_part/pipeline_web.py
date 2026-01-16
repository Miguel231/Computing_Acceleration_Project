"""
Pipeline mejorado con integración WebSocket para interfaz web
"""

import cv2
import numpy as np
import os, time
import asyncio

from face_detection import FaceDetector
from embeddings import Embeddings
from comunication import NtfyNotifier
from utils import load_db
from ws_client import WSClient


async def main():
    # ========== CONFIGURACIÓN ==========
    OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
    SLOTS = 10
    THRESH = 0.7
    EXPAND = 0.0

    detector_model_path = "/home/miserasp/Desktop/projecte/Computing_Acceleration_Project/data_projecte/detector.tflite"
    embed_model_path = "/home/miserasp/Desktop/projecte/Computing_Acceleration_Project/data_projecte/mobilefacenet_int8.tflite"
    db_embs_path = "/home/miserasp/Desktop/projecte/Computing_Acceleration_Project/data_projecte/storage.npz"

    # WebSocket backend URL - AJUSTA ESTO
    BACKEND_URL = "ws://localhost:8000"  # Si está en otro PC: "ws://192.168.1.100:8000"
    CAMERA_ID = "rpi_cam_001"

    # ========== INICIALIZACIÓN ==========
    print("=" * 60)
    print("   🔒 Smart EdgeAI Security - Pipeline")
    print("=" * 60)
    print()

    print("🔧 Inicializando modelos...")
    detector = FaceDetector(detector_model_path)
    embedder = Embeddings(embed_model_path)
    notifier = NtfyNotifier(topic="home-door-83f9a2")

    print("📂 Cargando base de datos de familiares...")
    names, db_embs = load_db(db_embs_path)
    print(f"✓ {len(names)} personas en base de datos: {names}")

    # Conectar al backend web
    print(f"🌐 Conectando al backend: {BACKEND_URL}")
    ws = WSClient(BACKEND_URL, CAMERA_ID)
    connected = await ws.connect()
    
    if not connected:
        print("\n⚠️  Backend no disponible - continuando sin interfaz web")
        print("   El sistema funcionará normalmente con notificaciones Ntfy")
        ws = None  # Desactivar websocket
    
    print()
    print("🚀 Sistema iniciado - Monitoreando carpeta...")
    print(f"   Carpeta: {OUT_DIR}")
    print(f"   Threshold: {THRESH}")
    print(f"   Slots: {SLOTS}")
    print()

    # ========== VARIABLES DE CONTROL ==========
    last_alert_time = 0.0
    ALERT_COOLDOWN_SEC = 10
    frame_count = 0
    STREAM_INTERVAL = 3  # Enviar frame cada N procesados

    # ========== LOOP PRINCIPAL ==========
    try:
        while True:
            did_work = False

            for slot in range(1, SLOTS + 1):
                path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")
                
                # Verificar si existe el archivo
                if not os.path.exists(path):
                    continue

                # Leer imagen
                img = cv2.imread(path)
                if img is None:
                    await asyncio.sleep(0.01)
                    continue

                try:
                    # ========== DETECCIÓN Y RECONOCIMIENTO ==========
                    cropped_face = detector.detect_and_crop_largest_face_tasks(
                        img, expand=EXPAND
                    )
                    
                    emb_face = embedder.get_embedding(
                        cropped_face, normalization="arcface"
                    )

                    # Comparar con base de datos
                    scores = db_embs @ emb_face
                    best_i = int(np.argmax(scores))
                    best_sim = float(scores[best_i])
                    best_name = names[best_i]

                    # ========== CLASIFICACIÓN ==========
                    if best_sim < THRESH:
                        # INTRUSO DETECTADO
                        label = "unknown"
                        is_family = False
                        
                        # Alerta Ntfy (con cooldown)
                        now = time.time()
                        if now - last_alert_time >= ALERT_COOLDOWN_SEC:
                            notifier.send(
                                title="Unknown person detected!",
                                message=f"Unknown at door. Best similarity: {best_sim:.4f}",
                                priority=4
                            )
                            last_alert_time = now
                        
                        print(f"⚠️  UNKNOWN (best_sim={best_sim:.4f})")
                        
                    else:
                        # FAMILIAR RECONOCIDO
                        label = "known"
                        is_family = True
                        print(f"✓ KNOWN: {best_name} (sim={best_sim:.4f})")

                    # ========== ENVIAR AL BACKEND WEB ==========
                    if ws and ws.is_connected:
                        # Convertir similarity (0-1, mayor=mejor) a distance (0-1, menor=mejor)
                        distance = 1.0 - best_sim
                        
                        # Enviar detección con el frame
                        await ws.send_detection(
                            frame=img,
                            person_name=best_name if is_family else None,
                            is_family=is_family,
                            distance=distance
                        )
                        
                        # Enviar frame para streaming (cada N frames)
                        if frame_count % STREAM_INTERVAL == 0:
                            await ws.send_frame(img)
                        
                        frame_count += 1

                except Exception as e:
                    print(f"✗ Error procesando {path}: {e}")

                # Eliminar archivo procesado
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

                did_work = True

            # Si no hubo trabajo, esperar un poco
            if not did_work:
                await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n✓ Deteniendo sistema...")
    
    finally:
        # ========== LIMPIEZA ==========
        detector.close()
        if ws:
            await ws.close()
        print("✓ Sistema detenido correctamente")


if __name__ == "__main__":
    asyncio.run(main())