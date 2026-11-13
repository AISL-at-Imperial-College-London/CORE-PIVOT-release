@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0" || exit /b 1
call "%~dp0scripts\find_uv.cmd"
if errorlevel 2 goto failed
if defined UV_EXE goto sync

echo uv was not found. Installing from the winget community source.
set "WINGET_EXE="
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\winget.exe" set "WINGET_EXE=%LOCALAPPDATA%\Microsoft\WindowsApps\winget.exe"
if not defined WINGET_EXE for /f "delims=" %%W in ('%SystemRoot%\System32\where.exe winget.exe 2^>nul') do if not defined WINGET_EXE set "WINGET_EXE=%%W"
if not defined WINGET_EXE goto no_winget
"%WINGET_EXE%" install --id astral-sh.uv -e --source winget
if errorlevel 1 goto install_failed
rem Find the installed executable directly; this shell's PATH may still be old.
call "%~dp0scripts\find_uv.cmd"
if not defined UV_EXE goto not_found

:sync
echo Using "%UV_EXE%"
"%UV_EXE%" --version
if errorlevel 1 goto failed
"%UV_EXE%" sync --locked --project "%CD%"
if errorlevel 1 goto failed
"%UV_EXE%" run --locked --project "%CD%" python -c "import sys, torch, numpy, scipy, gymnasium, pandas, matplotlib, imageio_ffmpeg; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('Environment imports OK')"
if errorlevel 1 goto failed
echo.
echo Setup complete. No training was launched.
echo Next: .\review.cmd --help
echo Quick evaluation: .\review.cmd evaluate best-pd --winds 1
popd
exit /b 0

:no_winget
echo ERROR: Neither uv nor WinGet was found. 1>&2
goto manual
:install_failed
echo ERROR: WinGet installation did not complete. Read its message above. 1>&2
goto manual
:not_found
echo ERROR: WinGet finished, but uv.exe was not found in its usual locations. 1>&2
:manual
echo Download the standalone Windows uv ZIP from https://github.com/astral-sh/uv/releases 1>&2
echo Extract it and set CORE_PIVOT_UV to the full uv.exe path. See docs/SETUP.md. 1>&2
echo No administrator or execution-policy changes are made by this script. 1>&2
:failed
echo Setup stopped. No training was launched. 1>&2
popd
exit /b 1
