@echo off
REM ===================================================================
REM  Lance l interface web de demonstration (double-cliquer ce fichier)
REM  Pre-requis : Ollama lance + "ollama pull qwen2.5:3b" + "uv sync"
REM ===================================================================
cd /d "%~dp0"
echo.
echo  Demarrage de l interface... (1 a 2 minutes au premier lancement)
echo  Elle s ouvrira dans le navigateur : http://localhost:8501
echo  Pour arreter : fermer cette fenetre.
echo.
uv run python main.py interface
pause
