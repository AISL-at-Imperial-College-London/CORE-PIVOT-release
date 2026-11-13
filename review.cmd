@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0" || exit /b 1
call "%~dp0scripts\find_uv.cmd"
if not defined UV_EXE (
  echo ERROR: uv was not found. Run .\setup.cmd first; see docs/SETUP.md. 1>&2
  popd
  exit /b 1
)
rem Explicit python mode supports the lower-level scripts in the README.
if /i "%~1"=="python" goto python
if "%~1"=="" goto help
"%UV_EXE%" run --locked --project "%CD%" python run.py %*
goto done
:help
"%UV_EXE%" run --locked --project "%CD%" python run.py --help
goto done
:python
"%UV_EXE%" run --locked --project "%CD%" %*
:done
set "REVIEW_EXIT=%ERRORLEVEL%"
popd
exit /b %REVIEW_EXIT%
