@echo off

SET CONDA_PATH=.\Miniconda3

REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

SET KMP_DUPLICATE_LIB_OK=TRUE

start "" "http://127.0.0.1:8010/webrtcapi.html"

python app.py --model ernerf --transport webrtc

cmd /k