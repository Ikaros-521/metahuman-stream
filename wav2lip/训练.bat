@echo off

SET CONDA_PATH=..\Miniconda3

REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

SET KMP_DUPLICATE_LIB_OK=TRUE


python genavatar.py --video_path 1.mp4 --img_size 96 --face_det_batch_size 8

cmd /k