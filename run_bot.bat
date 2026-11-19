@echo off
setlocal
cd /d "%~dp0"

REM Reuse a working virtual environment; rebuild it if its Python is broken.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if errorlevel 1 (
        echo Bozuk sanal ortam temizleniyor...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Uygun Python kurulumu araniyor...
    for %%V in (3.12 3.11 3.13 3.14 3.10) do call :try_py %%V
    if not exist ".venv\Scripts\python.exe" (
        for %%P in (python.exe python3.exe) do call :try_python %%P
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python bulunamadi veya sanal ortam olusturulamadi.
    echo Python 3.11 veya 3.12 kurun: https://www.python.org/downloads/
    echo Kurulumda "Add python.exe to PATH" secenegini isaretleyin.
    pause
    exit /b 1
)

echo.
echo Python ortami hazirlaniyor...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo pip guncellenemedi. Internet baglantisini kontrol edin.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Paket kurulumu basarisiz. Internet baglantisini ve requirements.txt dosyasini kontrol edin.
    pause
    exit /b 1
)

if /I "%~1"=="--rebuild" (
    if exist "results_market_1x2" rmdir /s /q "results_market_1x2"
)

if not exist "results_market_1x2\market_1x2_summary.json" (
    if not exist "xgo.csv" (
        echo Gecmis veri dosyasi xgo.csv bulunamadi.
        echo xgo.csv dosyasini run_bot.bat ile ayni klasore koyun.
        pause
        exit /b 1
    )
    echo Gecmis oran verisi analiz ediliyor...
    ".venv\Scripts\python.exe" market_1x2_analysis.py xgo.csv --out results_market_1x2
    if errorlevel 1 (
        echo Gecmis veri analizi basarisiz.
        pause
        exit /b 1
    )
)

if not exist "upcoming.csv" (
    copy "upcoming_template.csv" "upcoming.csv" >nul
    echo.
    echo upcoming.csv olusturuldu.
    echo Bu dosyayi Excel'de acip mac tarihi, takimlar ve AvgH AvgD AvgA oranlarini girin.
    echo Ardindan run_bot.bat dosyasini tekrar calistirin.
    pause
    exit /b 0
)

echo.
echo Maclar analiz ediliyor...
".venv\Scripts\python.exe" market_1x2_predict.py upcoming.csv --model-summary results_market_1x2\market_1x2_summary.json --out results_upcoming
if errorlevel 1 (
    echo Tahmin islemi basarisiz.
    pause
    exit /b 1
)

echo.
echo Sonuclar results_upcoming klasorune yazildi.
start "" "results_upcoming"
pause
exit /b 0

:try_py
if exist ".venv\Scripts\python.exe" exit /b 0
py -%~1 --version >nul 2>&1
if errorlevel 1 exit /b 0
py -%~1 -m venv .venv >nul 2>&1
if errorlevel 1 (
    if exist ".venv" rmdir /s /q ".venv"
)
exit /b 0

:try_python
if exist ".venv\Scripts\python.exe" exit /b 0
%~1 --version >nul 2>&1
if errorlevel 1 exit /b 0
%~1 -m venv .venv >nul 2>&1
if errorlevel 1 (
    if exist ".venv" rmdir /s /q ".venv"
)
exit /b 0
