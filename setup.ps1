$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
$venvReady = Test-Path $venvPython
if ($venvReady) {
    & $venvPython --version 2>$null
    $venvReady = $LASTEXITCODE -eq 0
}

if (-not $venvReady) {
    $venvCreator = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 --version 2>$null
        if ($LASTEXITCODE -eq 0) { $venvCreator = 'py' }
    }
    if (-not $venvCreator -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python --version 2>$null
        if ($LASTEXITCODE -eq 0) { $venvCreator = 'python' }
    }
    if (-not $venvCreator) {
        throw 'Python 3 is required to create venv.'
    }
    if (Test-Path 'venv') { Remove-Item -Recurse -Force 'venv' }
    if ($venvCreator -eq 'py') { & py -3 -m venv venv } else { & python -m venv venv }
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
& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-goemotions.txt')
if ($LASTEXITCODE) { throw 'Failed to install GoEmotions requirements.' }
