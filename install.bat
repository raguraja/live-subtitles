@echo off
echo ========================================
echo  LiveSubtitles - Installing dependencies
echo ========================================
echo.

:: Create conda environment
call conda create -n subtitles python=3.11 -y
call conda activate subtitles

:: PyTorch with CUDA 12.1
echo [1/4] Installing PyTorch (CUDA 12.1)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: faster-whisper
echo [2/4] Installing faster-whisper...
pip install faster-whisper

:: pyaudiowpatch (WASAPI loopback)
echo [3/4] Installing pyaudiowpatch...
pip install pyaudiowpatch

:: scipy for resampling
echo [4/4] Installing scipy...
pip install scipy

echo.
echo ========================================
echo  Done! Run with: run.bat
echo  First launch will download Whisper model (~500MB for 'small')
echo ========================================
pause
