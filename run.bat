@echo off
call conda activate subtitles
python "%~dp0subtitle_app.py" %*
