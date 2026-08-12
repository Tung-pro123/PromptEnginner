@echo off
echo ========================================================
echo [JETSON AI RACER] CONG CU TAI THU VIEN OFFLINE
echo ========================================================
echo.
echo Dang doc file requirements_jetson.txt...
echo Va tai cac goi thu vien vao thu muc offline_libs...
echo.

if not exist "offline_libs" (
    mkdir offline_libs
)

:: Su dung pip de tai cac goi python duoi dang file nén (khong cai dat vao Windows)
:: Vi day la cac thu vien Pure Python (nhu Adafruit), nen khong can chi dinh platform linux
pip download -r requirements_jetson.txt -d offline_libs

echo.
echo ========================================================
echo XONG! Tat ca cac file thu vien da nam trong thu muc offline_libs/
echo Ban bay gio co the copy toan bo thu muc project nay vao Jetson Nano
echo Va chay ./deploy_car.sh de cai dat offline (khong can mang).
echo ========================================================
pause
