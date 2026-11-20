# Audio2Text AI - atajos locales para Windows.
# GNU Make se puede invocar desde PowerShell, cmd o Git Bash. Las recetas usan
# cmd.exe explícitamente para que las rutas de .venv y "set VAR=..." sean
# consistentes en Windows.

SHELL := cmd.exe
.SHELLFLAGS := /C
.DEFAULT_GOAL := help

PYTHON := .venv\Scripts\python.exe
STREAMLIT := .venv\Scripts\streamlit.exe
CHECK_PATHS := streamlit_app.py src scripts tests

.PHONY: help run test test-real check clean

help:
	@echo.
	@echo Audio2Text AI - comandos disponibles
	@echo.
	@echo   make run        Inicia Streamlit en http://localhost:8501
	@echo   make test       Ejecuta la suite rapida con unittest
	@echo   make test-real  Ejecuta la inferencia real desde Streamlit
	@echo   make check      Compila el codigo Python con compileall
	@echo   make clean      Elimina solo caches y archivos temporales seguros
	@echo   make help       Muestra esta ayuda
	@echo.
	@echo Entorno virtual: $(PYTHON)

run:
	@"$(STREAMLIT)" run streamlit_app.py

test:
	@"$(PYTHON)" -m unittest discover -s tests -v

test-real:
	@set "AUDIO2TEXT_RUN_UI_INFERENCE=1" && "$(PYTHON)" -m unittest tests.test_streamlit_inference -v

check:
	@"$(PYTHON)" -m compileall -q $(CHECK_PATHS)

clean:
	@"$(PYTHON)" scripts\clean_caches.py
