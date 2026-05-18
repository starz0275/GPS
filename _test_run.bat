@echo off
call D:\miniconda3\Scripts\activate.bat traj312
cd /d C:\Users\nxj\Desktop\GPS
python -u _test_import.py > _test_out.txt 2>&1
echo EXIT=%ERRORLEVEL%>> _test_out.txt
