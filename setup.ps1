$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-Python {
    param(
        [string]$Executable,
        [string[]]$Prefix = @()
    )
    try {
        & $Executable @Prefix -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

$pythonExecutable = $null
$pythonPrefix = @()
if ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-Python "py" @("-3"))) {
    $pythonExecutable = "py"
    $pythonPrefix = @("-3")
}
elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-Python "python")) {
    $pythonExecutable = "python"
}
else {
    throw "Python 3.10 or newer is required. Install it, then run setup.ps1 again."
}

$venvPath = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -LiteralPath $venvPath -Recurse -Force
    }
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $pythonExecutable @pythonPrefix -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install project requirements." }

if (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".git")) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required to initialize benchmark submodules."
    }
    & git -c core.protectNTFS=false submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { throw "Failed to initialize benchmark submodules." }
}

$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot ".env.example") -Destination $envPath
}

Write-Host ""
Write-Host "PRIMA-NEXT is ready."
Write-Host "Activate it with: . .\.venv\Scripts\Activate.ps1"
Write-Host "Run checks with:  python -m pytest -q"
