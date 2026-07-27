@echo off
title AirOS++ Desktop Control Center
echo Launching AirOS++ Control Center GUI...
cd /d "%~dp0"
call C:\Users\vipul\miniconda3\Scripts\activate.bat airos-env
python app.py
pause
