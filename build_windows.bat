@echo off
chcp 65001 >nul
setlocal

echo === Сборка «Фото на документы — Дни Озёрного» (Windows x64) ===
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python 3.11 не найден в PATH.
    echo Установите Python 3.11 x64 с python.org (с галочкой Add Python to PATH) или используйте GitHub Actions.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [1/4] Создаю виртуальное окружение .venv...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [2/4] Устанавливаю зависимости и PyInstaller...
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

echo [3/4] Скачиваю офлайн-модели AI (если ещё не скачаны)...
if not exist "models" mkdir models
if not exist "models\face_landmarker.task" (
    curl.exe -fL --retry 4 "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o "models\face_landmarker.task"
)
if not exist "models\rmbg_1_4_q.onnx" (
    curl.exe -fL --retry 4 "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx" -o "models\rmbg_1_4_q.onnx"
)

echo [4/4] Собираю FotoDniOzernogo.exe...
pyinstaller --clean --noconfirm FotoDniOzernogo.spec

echo.
echo === Готово! Программа собрана в папке: dist\FotoDniOzernogo\FotoDniOzernogo.exe ===
pause
