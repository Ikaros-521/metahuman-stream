@echo off

SET CONDA_PATH=.\Miniconda3

REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

SET KMP_DUPLICATE_LIB_OK=TRUE

start "" "http://127.0.0.1:8010/webrtcapi.html"

python app.py --tts gpt-sovits-v2 --transport webrtc --model wav2lip --avatar_id wav2lip_avatar2 --TTS_SERVER http://127.0.0.1:9880 --REF_FILE C:\\Users\\Administrator\\Music\\test.wav --REF_TEXT 大家有什么问题都可以直接问啊，主播会尽力回答的

cmd /k