# Builds dist\SimulaticsStudio.exe (one file, about 150 MB) from this checkout.
#   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m pip install -q -e ".[studio]" pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
& $py -m PyInstaller packaging\simulatics_studio.spec --noconfirm --distpath dist --workpath build\pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Get-Item dist\SimulaticsStudio.exe | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
