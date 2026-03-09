@echo off
cd /d "%~dp0"
echo Downloading LingBot-World base-act model...
echo.
python download.py --model base-act
echo.
echo Done.
pause
