@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\\Scripts\\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Python 3 bulunamadi. Python 3.11 veya 3.12 kurup tekrar deneyin.
        pause
        exit /b 1
    )
)

".venv\\Scripts\\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Paket kurulumu basarisiz.
    pause
    exit /b 1
)

if /I "%~1"=="--rebuild" (
    if exist "results_market_1x2" rmdir /s /q "results_market_1x2"
)

if not exist "results_market_1x2\\market_1x2_summary.json" (
    echo Gecmis oran verisi analiz ediliyor...
    ".venv\\Scripts\\python.exe" market_1x2_analysis.py xgo.csv --out results_market_1x2
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
".venv\\Scripts\\python.exe" market_1x2_predict.py upcoming.csv --model-summary results_market_1x2\\market_1x2_summary.json --out results_upcoming
if errorlevel 1 (
    echo Tahmin islemi basarisiz.
    pause
    exit /b 1
)

echo.
echo Sonuclar results_upcoming klasorune yazildi.
start "" "results_upcoming"
pause
