# Keep Python's CLI canonical; use the presenter's active environment.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$demoExitCode = 1
$demoArguments = $args
# Legacy PowerShell drops empty strings and embedded quotes when invoking native apps.
$demoNativePassing = Get-Variable PSNativeCommandArgumentPassing -ValueOnly -ErrorAction SilentlyContinue
if (-not $demoNativePassing -or $demoNativePassing -eq "Legacy") {
    $demoArguments = @($args | ForEach-Object {
        '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
    })
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & python -m demo_cli.cli @demoArguments
    $demoExitCode = $LASTEXITCODE
}
catch {
    Write-Host "Unable to invoke Python. Activate your Python environment and try again."
}
finally {
    Pop-Location
}
exit $demoExitCode
