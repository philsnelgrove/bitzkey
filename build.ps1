# Builds dist\Bitzkey.exe: a single-file Windows executable with no console window.
# Run from anywhere:  .\build.ps1
# It uses the project's .venv Python when there is one, otherwise the python on PATH.
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

& $python -m pip install --disable-pip-version-check -r requirements.txt -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m PyInstaller --noconfirm --clean --onefile --windowed --name Bitzkey --add-data "i18n;i18n" Main.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built $PSScriptRoot\dist\Bitzkey.exe"
