@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ショートカット／ダブルクリック時は PATH が薄く、streamlit 単体が無いことが多い
python -c "import streamlit" 2>nul
if errorlevel 1 (
  py -c "import streamlit" 2>nul
  if errorlevel 1 (
    echo [エラー] streamlit が入った Python が見つかりません。
    echo 次をコマンドプロンプトで試してください:
    echo   cd /d "%~dp0"
    echo   python -m pip install -r requirements.txt
    echo   python -m streamlit run app.py
    pause
    exit /b 1
  )
  set "PYRUN=py -m streamlit"
) else (
  set "PYRUN=python -m streamlit"
)

echo NOTAM 解析アプリを起動しています...
echo ブラウザが開かない場合は表示された URL を開いてください。
echo 終了するときはこのウィンドウで Ctrl+C を押してから閉じてください。
echo.
%PYRUN% run app.py

if errorlevel 1 (
  echo.
  echo エラーで終了しました。上のメッセージを確認してください。
  pause
)
