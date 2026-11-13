@echo off
rem Called by setup.cmd/review.cmd inside their SETLOCAL scope.
rem An explicit override is useful for a manually downloaded standalone uv.exe.
set "UV_EXE="
if not defined CORE_PIVOT_UV goto known_locations
if not exist "%CORE_PIVOT_UV%" (
  echo ERROR: CORE_PIVOT_UV does not point to an existing uv.exe. 1>&2
  exit /b 2
)
set "UV_EXE=%CORE_PIVOT_UV%"
exit /b 0

:known_locations
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe" set "UV_EXE=%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
if defined UV_EXE exit /b 0
rem WinGet portable packages may exist even when their Links alias is missing.
rem Search only uv's package directories, not the user's other applications.
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*") do if exist "%%~fD\" call :scan_package "%%~fD"
if defined UV_EXE exit /b 0
for /d %%D in ("%ProgramFiles%\WinGet\Packages\astral-sh.uv_*") do if exist "%%~fD\" call :scan_package "%%~fD"
if defined UV_EXE exit /b 0
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if defined UV_EXE exit /b 0
if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.cargo\bin\uv.exe"
if defined UV_EXE exit /b 0
if exist "%ProgramFiles%\WinGet\Links\uv.exe" set "UV_EXE=%ProgramFiles%\WinGet\Links\uv.exe"
if defined UV_EXE exit /b 0
for /f "delims=" %%U in ('%SystemRoot%\System32\where.exe uv.exe 2^>nul') do if not defined UV_EXE set "UV_EXE=%%U"
if defined UV_EXE exit /b 0
exit /b 1

:scan_package
for /r "%~1" %%U in (uv.exe) do if exist "%%~fU" if not defined UV_EXE set "UV_EXE=%%~fU"
exit /b 0
