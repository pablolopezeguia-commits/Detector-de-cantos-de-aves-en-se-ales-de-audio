@echo off
cd ..
call .venv\Scripts\activate
pyinstaller build\ecocanto.spec --clean --noconfirm
echo Ejecutable en: dist\EcoCanto\EcoCanto.exe
pause

