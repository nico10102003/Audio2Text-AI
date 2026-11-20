@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] No se encontro .venv\Scripts\streamlit.exe
    echo Crea el entorno virtual e instala requirements.txt antes de continuar.
    pause
    exit /b 1
)

echo Iniciando Audio2Text AI...
echo URL local: http://localhost:8501
echo Cierra esta ventana para detener el servidor.
echo.

".venv\Scripts\streamlit.exe" run streamlit_app.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Streamlit termino con codigo %EXIT_CODE%.
    pause
)

endlocal & exit /b %EXIT_CODE%
