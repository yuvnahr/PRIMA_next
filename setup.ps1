$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv venv
    } else {
        throw 'Python 3 is required to create venv.'
    }
    if ($LASTEXITCODE) { throw 'Failed to create venv.' }
}

& git submodule update --init --recursive
if ($LASTEXITCODE) { throw 'Failed to initialize git submodules.' }

$goEmotions = Join-Path $PSScriptRoot 'benchmarks\goemotions\external'
if (-not (Test-Path $goEmotions)) {
    New-Item -ItemType Directory -Force (Split-Path $goEmotions) | Out-Null
    & git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git $goEmotions
    if ($LASTEXITCODE) { throw 'Failed to clone GoEmotions.' }
}
if (-not (Test-Path (Join-Path $goEmotions '.git'))) {
    throw "$goEmotions exists but is not a Git checkout."
}
& git -C $goEmotions sparse-checkout set goemotions
if ($LASTEXITCODE) { throw 'Failed to configure the GoEmotions sparse checkout.' }

& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE) { throw 'Failed to install requirements.' }
