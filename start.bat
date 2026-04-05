@echo off
title EmoTune — Speech Emotion ^& Music
cd /d "%~dp0"
echo.
echo  Starting EmoTune Web App...
echo  Open http://127.0.0.1:5000/ in your browser
echo.
set PYTHONUTF8=1
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
python app.py
pause
