@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if errorlevel 1 rmdir /s /q ".venv"
)

if not exist ".venv\Scripts\python.exe" (
    echo Python ortami hazirlaniyor...
    for %%V in (3.12 3.11 3.13 3.14 3.10) do call :try_py %%V
    if not exist ".venv\Scripts\python.exe" (
        for %%P in (python.exe python3.exe) do call :try_python %%P
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Python bulunamadi. Python 3.11 veya ustunu kurun.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_error

if /I "%~1"=="--rebuild" (
    if exist "results_market_1x2" rmdir /s /q "results_market_1x2"
)

if not exist "results_market_1x2\market_1x2_summary.json" (
    if not exist "xgo.csv" (
        echo Gecmis veri dosyasi xgo.csv bulunamadi.
        pause
        exit /b 1
    )
    echo Gecmis piyasa analizi ilk kez hazirlaniyor...
    ".venv\Scripts\python.exe" market_1x2_analysis.py xgo.csv --out results_market_1x2
    if errorlevel 1 (
        echo Gecmis analiz basarisiz.
        pause
        exit /b 1
    )
)

echo.
echo T-OPTA Tahminkar web arayuzu baslatiliyor...
start "T-OPTA Tahminkar" ".venv\Scripts\python.exe" mackolik_web.py
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8765"
echo.
echo Site acildi. Maclar ve oranlar dogrudan Mackolik'ten yukleniyor.
echo 10 uygun mac yoksa bulunan uygun mac sayisi ve tum secimler ekranda listelenir.
echo Siteyi kapatmak icin acilan Python penceresini kapatin.
exit /b 0

:install_error
echo Paket kurulumu basarisiz. Internet baglantisini kontrol edin.
pause
exit /b 1

:try_py
if exist ".venv\Scripts\python.exe" exit /b 0
py -%~1 --version >nul 2>&1
if errorlevel 1 exit /b 0
py -%~1 -m venv .venv >nul 2>&1
if errorlevel 1 if exist ".venv" rmdir /s /q ".venv"
exit /b 0

:try_python
if exist ".venv\Scripts\python.exe" exit /b 0
%~1 --version >nul 2>&1
if errorlevel 1 exit /b 0
%~1 -m venv .venv >nul 2>&1
if errorlevel 1 if exist ".venv" rmdir /s /q ".venv"
exit /b 0
