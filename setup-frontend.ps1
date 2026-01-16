# Script de configuración automática del Frontend
# Smart EdgeAI Security - Frontend Setup

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Smart EdgeAI Security - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Node.js
Write-Host "Verificando Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "✓ Node.js instalado: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Node.js no está instalado" -ForegroundColor Red
    Write-Host "Por favor, instala Node.js desde: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Crear estructura de carpetas
Write-Host ""
Write-Host "Creando estructura de carpetas..." -ForegroundColor Yellow

if (Test-Path "frontend") {
    Write-Host "⚠ La carpeta 'frontend' ya existe. ¿Deseas sobrescribir? (S/N)" -ForegroundColor Yellow
    $respuesta = Read-Host
    if ($respuesta -ne "S" -and $respuesta -ne "s") {
        Write-Host "Operación cancelada." -ForegroundColor Red
        exit 0
    }
    Remove-Item -Recurse -Force frontend
}

New-Item -ItemType Directory -Path "frontend" | Out-Null
New-Item -ItemType Directory -Path "frontend/src" | Out-Null
Write-Host "✓ Carpetas creadas" -ForegroundColor Green

# Crear package.json
Write-Host "Creando package.json..." -ForegroundColor Yellow
$packageJson = @"
{
  "name": "smart-edgeai-security-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "lucide-react": "^0.263.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "vite": "^5.0.8"
  }
}
"@
$packageJson | Out-File -FilePath "frontend/package.json" -Encoding utf8
Write-Host "✓ package.json creado" -ForegroundColor Green

# Crear vite.config.js
Write-Host "Creando vite.config.js..." -ForegroundColor Yellow
$viteConfig = @"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      }
    }
  }
})
"@
$viteConfig | Out-File -FilePath "frontend/vite.config.js" -Encoding utf8
Write-Host "✓ vite.config.js creado" -ForegroundColor Green

# Crear tailwind.config.js
Write-Host "Creando tailwind.config.js..." -ForegroundColor Yellow
$tailwindConfig = @"
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"@
$tailwindConfig | Out-File -FilePath "frontend/tailwind.config.js" -Encoding utf8
Write-Host "✓ tailwind.config.js creado" -ForegroundColor Green

# Crear postcss.config.js
Write-Host "Creando postcss.config.js..." -ForegroundColor Yellow
$postcssConfig = @"
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"@
$postcssConfig | Out-File -FilePath "frontend/postcss.config.js" -Encoding utf8
Write-Host "✓ postcss.config.js creado" -ForegroundColor Green

# Crear index.html
Write-Host "Creando index.html..." -ForegroundColor Yellow
$indexHtml = @"
<!doctype html>
<html lang="es">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Smart EdgeAI Security</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
"@
$indexHtml | Out-File -FilePath "frontend/index.html" -Encoding utf8
Write-Host "✓ index.html creado" -ForegroundColor Green

# Crear src/main.jsx
Write-Host "Creando src/main.jsx..." -ForegroundColor Yellow
$mainJsx = @"
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"@
$mainJsx | Out-File -FilePath "frontend/src/main.jsx" -Encoding utf8
Write-Host "✓ src/main.jsx creado" -ForegroundColor Green

# Crear src/index.css
Write-Host "Creando src/index.css..." -ForegroundColor Yellow
$indexCss = @"
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
"@
$indexCss | Out-File -FilePath "frontend/src/index.css" -Encoding utf8
Write-Host "✓ src/index.css creado" -ForegroundColor Green

# Crear src/App.jsx (IMPORTANTE: Debes pegar el código del componente aquí)
Write-Host "Creando src/App.jsx..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "⚠ ACCIÓN REQUERIDA" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "El archivo App.jsx es muy grande para incluirlo en este script." -ForegroundColor Yellow
Write-Host "Por favor, copia manualmente el contenido del artefacto 'Frontend React - App.jsx'" -ForegroundColor Yellow
Write-Host "en el archivo: frontend/src/App.jsx" -ForegroundColor Cyan
Write-Host ""
Write-Host "Presiona ENTER cuando hayas copiado el archivo..." -ForegroundColor Yellow
Read-Host

# Verificar si App.jsx existe
if (-not (Test-Path "frontend/src/App.jsx")) {
    Write-Host "✗ App.jsx no encontrado. Creando plantilla básica..." -ForegroundColor Yellow
    $appJsx = @"
import React from 'react';

export default function App() {
  return (
    <div style={{padding: '20px', fontFamily: 'Arial'}}>
      <h1>Smart EdgeAI Security</h1>
      <p>Por favor, reemplaza este archivo con el código completo del artefacto App.jsx</p>
    </div>
  );
}
"@
    $appJsx | Out-File -FilePath "frontend/src/App.jsx" -Encoding utf8
    Write-Host "⚠ Plantilla básica creada. DEBES reemplazarla con el código completo." -ForegroundColor Yellow
} else {
    Write-Host "✓ App.jsx encontrado" -ForegroundColor Green
}

# Instalar dependencias
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
Write-Host "Esto puede tardar unos minutos..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location frontend
try {
    npm install
    Write-Host ""
    Write-Host "✓ Dependencias instaladas correctamente" -ForegroundColor Green
} catch {
    Write-Host "✗ Error al instalar dependencias" -ForegroundColor Red
    Write-Host "Por favor, ejecuta manualmente: cd frontend && npm install" -ForegroundColor Yellow
    Set-Location ..
    exit 1
}

Set-Location ..

# Resumen final
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✓ INSTALACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Estructura creada:" -ForegroundColor Cyan
Write-Host "  frontend/" -ForegroundColor White
Write-Host "  ├── src/" -ForegroundColor White
Write-Host "  │   ├── App.jsx" -ForegroundColor White
Write-Host "  │   ├── main.jsx" -ForegroundColor White
Write-Host "  │   └── index.css" -ForegroundColor White
Write-Host "  ├── index.html" -ForegroundColor White
Write-Host "  ├── package.json" -ForegroundColor White
Write-Host "  ├── vite.config.js" -ForegroundColor White
Write-Host "  ├── tailwind.config.js" -ForegroundColor White
Write-Host "  └── postcss.config.js" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PRÓXIMOS PASOS:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Asegúrate de tener el backend corriendo:" -ForegroundColor Yellow
Write-Host "   cd web_part" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor White
Write-Host ""
Write-Host "2. En otra terminal, ejecuta el frontend:" -ForegroundColor Yellow
Write-Host "   cd frontend" -ForegroundColor White
Write-Host "   npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "3. Abre tu navegador en:" -ForegroundColor Yellow
Write-Host "   http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "¡Listo para usar! 🚀" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""