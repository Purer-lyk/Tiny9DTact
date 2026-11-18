@echo off
REM =====================================================================
REM  build_release.bat -- build the Tiny9DTact Windows exe and assemble
REM  the release folder.
REM
REM  Run this ON THE WINDOWS DEV MACHINE (double-click or from cmd),
REM  from anywhere -- it cd's to the repo root on its own.
REM
REM  Produces:
REM    release\Tiny9DTact\            exe + _internal + data folders
REM    release\Tiny9DTact_win64.zip   same content, zipped for sharing
REM
REM  Requires: Python 3.10+ on PATH (the same interpreter that runs the
REM  app is used for the build -- no version drift). PyInstaller is
REM  installed automatically if missing.
REM =====================================================================
setlocal EnableExtensions
cd /d "%~dp0"

REM ---- 1) locate python ------------------------------------------------
set "PY=python"
where python >nul 2>nul
if errorlevel 1 (
    set "PY=py -3"
    py -3 --version >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found on PATH. Install Python 3.10+ first.
        exit /b 1
    )
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo [1/5] Using %%v

REM ---- 2) ensure PyInstaller -------------------------------------------
%PY% -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo [2/5] Installing PyInstaller ...
    %PY% -m pip install --upgrade pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        exit /b 1
    )
) else (
    echo [2/5] PyInstaller ready.
)

REM ---- 3) build ---------------------------------------------------------
echo [3/5] Building Tiny9DTact.exe (onedir) ...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
%PY% -m PyInstaller --noconfirm --clean Tiny9DTact.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed -- see the log above.
    exit /b 1
)
if not exist "dist\Tiny9DTact\Tiny9DTact.exe" (
    echo [ERROR] Build finished but dist\Tiny9DTact\Tiny9DTact.exe is missing.
    exit /b 1
)

REM ---- 4) assemble the release folder ------------------------------------
echo [4/5] Assembling release\Tiny9DTact ...
set "REL=release\Tiny9DTact"
if exist "%REL%" rmdir /s /q "%REL%"

xcopy /e /i /y /q "dist\Tiny9DTact" "%REL%" >nul
if errorlevel 1 goto :copyfail

mkdir "%REL%\shape_reconstruction" 2>nul
xcopy /y /q "shape_reconstruction\shape_config.yaml" "%REL%\shape_reconstruction\" >nul
if errorlevel 1 goto :copyfail
xcopy /e /i /y /q "shape_reconstruction\calibration" "%REL%\shape_reconstruction\calibration" >nul
if errorlevel 1 goto :copyfail

mkdir "%REL%\web_viewer" 2>nul
xcopy /y /q "web_viewer\index.html" "%REL%\web_viewer\" >nul
if errorlevel 1 goto :copyfail

REM Copy the end-user notes if present (wildcard keeps this file pure
REM ASCII -- cmd parses batch files in the OEM codepage, so a literal
REM non-ASCII filename here would be mojibake on a Chinese Windows).
if exist "release\*.txt" (
    xcopy /y /q "release\*.txt" "%REL%\" >nul
)

REM ---- 5) zip for sharing -------------------------------------------------
echo [5/5] Zipping release\Tiny9DTact_win64.zip ...
powershell -NoProfile -Command "Compress-Archive -Path 'release\Tiny9DTact' -DestinationPath 'release\Tiny9DTact_win64.zip' -Force"
if errorlevel 1 (
    echo [WARN] Zip step failed -- the release folder itself is still fine.
)

echo.
echo =====================================================
echo  Done. Release folder:
echo    %CD%\release\Tiny9DTact\Tiny9DTact.exe
echo  Shareable archive:
echo    %CD%\release\Tiny9DTact_win64.zip
echo  First run needs internet (three.js CDN download).
echo =====================================================
REM Keep the console open when double-clicked so the result is visible.
pause
exit /b 0

:copyfail
echo [ERROR] Failed to copy data files into the release folder.
exit /b 1
