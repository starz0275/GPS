@echo off
call D:\miniconda3\Scripts\activate.bat traj312
cd /d C:\Users\nxj\Desktop\GPS
set MPLBACKEND=Agg
set TF_CPP_MIN_LOG_LEVEL=2

echo === Step 1: data preprocessing ===
python -u data_preprocessing.py
if errorlevel 1 exit /b 1

echo === Step 2: training ===
python -u train_tcn_model.py
if errorlevel 1 exit /b 1

echo === Step 3: check training ===
python -u check_training.py
