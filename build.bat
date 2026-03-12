@echo off
echo ==============================================
echo Building SANFA V5 Standalone Executable...
echo ==============================================
echo.

IF NOT EXIST "env\Scripts\activate.bat" (
    echo [ERROR] No virtual environment found. Make sure you are running 'pip install -r requirements.txt'
    pause
    exit /b
)

call env\Scripts\activate.bat
echo [OK] Virtual Environment activated.
echo.

echo Running PyInstaller...
pyinstaller sanfa.spec --clean --y

echo.
echo ==============================================
echo SUCCESS!
echo Your executable is located at: dist/SANFA/SANFA.exe
echo Send the entire 'SANFA' folder inside 'dist' to your friends.
echo ==============================================
pause
