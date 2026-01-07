"""
Script de prueba para verificar que tus módulos funcionan correctamente
antes de integrar con el backend.
"""

import cv2
import numpy as np
from face_detection import FaceDetector
from embeddings import Embeddings

print("=" * 60)
print("   🧪 Test de Módulos - Smart EdgeAI Security")
print("=" * 60)
print()

# ========== TEST 1: Inicializar Detector ==========
print("1️⃣  Inicializando FaceDetector...")
try:
    detector = FaceDetector()
    print("   ✓ FaceDetector inicializado correctamente")
except Exception as e:
    print(f"   ✗ Error inicializando FaceDetector: {e}")
    exit(1)

# ========== TEST 2: Inicializar EmbeddingGenerator ==========
print("\n2️⃣  Inicializando EmbeddingGenerator...")
try:
    embedder = Embeddings()
    print("   ✓ EmbeddingGenerator inicializado correctamente")
except Exception as e:
    print(f"   ✗ Error inicializando EmbeddingGenerator: {e}")
    exit(1)

# ========== TEST 3: Probar cámara ==========
print("\n3️⃣  Probando cámara...")
try:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   ⚠️  No se puede abrir cámara con índice 0")
        print("   Prueba con: python check_camera.py")
    else:
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"   ✓ Cámara funcionando ({w}x{h})")
        else:
            print("   ✗ Error capturando frame")
    
    cap.release()
    
except Exception as e:
    print(f"   ✗ Error con cámara: {e}")

# ========== TEST 4: Detectar caras en imagen de prueba ==========
print("\n4️⃣  Probando detección de caras...")
try:
    # Crear imagen de prueba o capturar una
    cap = cv2.VideoCapture(0)
    ret, test_frame = cap.read()
    cap.release()
    
    if ret:
        faces = detector.detect(test_frame)
        
        if faces is None:
            print("   ℹ️  No se detectaron caras (puede ser normal)")
        elif len(faces) == 0:
            print("   ℹ️  No se detectaron caras (puede ser normal)")
        else:
            print(f"   ✓ Detectadas {len(faces)} cara(s)")
            print(f"   Tipo de retorno: {type(faces)}")
            if len(faces) > 0:
                print(f"   Estructura de face[0]: {type(faces[0])}")
                if isinstance(faces[0], dict):
                    print(f"   Keys: {faces[0].keys()}")
    else:
        print("   ⚠️  No se pudo capturar frame para test")
        
except Exception as e:
    print(f"   ✗ Error en detección: {e}")
    import traceback
    traceback.print_exc()

# ========== TEST 5: Generar embedding ==========
print("\n5️⃣  Probando generación de embeddings...")
try:
    # Crear imagen de prueba (cara sintética)
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    embedding = embedder.generate(test_image)
    
    print(f"   ✓ Embedding generado")
    print(f"   Shape: {embedding.shape}")
    print(f"   Dtype: {embedding.dtype}")
    print(f"   Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    # Verificar dimensiones esperadas
    if embedding.shape[0] == 512:
        print("   ✓ Dimensión correcta (512)")
    else:
        print(f"   ⚠️  Dimensión inesperada: {embedding.shape[0]} (esperado: 512)")
        
except Exception as e:
    print(f"   ✗ Error generando embedding: {e}")
    import traceback
    traceback.print_exc()

# ========== TEST 6: Calcular distancia ==========
print("\n6️⃣  Probando cálculo de distancias...")
try:
    # Crear dos embeddings de prueba
    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    
    # Distancia euclidiana
    distance = np.linalg.norm(emb1 - emb2)
    
    print(f"   ✓ Distancia calculada: {distance:.3f}")
    
    # Probar con embeddings idénticos
    distance_same = np.linalg.norm(emb1 - emb1)
    print(f"   ✓ Distancia mismo embedding: {distance_same:.6f} (debería ser ~0)")
    
except Exception as e:
    print(f"   ✗ Error calculando distancia: {e}")

# ========== RESUMEN ==========
print("\n" + "=" * 60)
print("   ✅ TESTS COMPLETADOS")
print("=" * 60)
print()
print("Si todos los tests pasaron, puedes ejecutar:")
print("  python integrated_client.py")
print()
print("Si algún test falló, revisa:")
print("  - Que los modelos estén en la carpeta correcta")
print("  - Que las rutas en face_detection.py y embeddings.py sean correctas")
print("  - Que la cámara esté conectada y funcionando")
print()