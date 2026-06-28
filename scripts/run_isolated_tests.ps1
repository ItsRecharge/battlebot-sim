#!/usr/bin/env pwsh
# Run each test file in its OWN Python process to dodge the intermittent native
# DLL load/teardown crash (NumPy + SciPy + VTK + MuJoCo + Pillow in one process —
# see CONTRIBUTING.md).
#
# Pass/fail is judged from each run's JUnit XML, NOT the process exit code: the
# native stack can crash on *shutdown* with a non-zero code AFTER every test has
# already passed, so trusting the exit code would report false failures.

$ErrorActionPreference = "Stop"
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$env:QT_QPA_PLATFORM = "offscreen"

$reportDir = Join-Path $env:TEMP "bbsim_junit"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$failed = @()
Get-ChildItem -Path "tests" -Filter "test_*.py" -Recurse | ForEach-Object {
    $name = $_.BaseName
    $xml = Join-Path $reportDir "$name.xml"
    Write-Host "=== $($_.Name) ===" -ForegroundColor Cyan
    & $py -m pytest $_.FullName -q "--junitxml=$xml" | Out-Host

    if (-not (Test-Path $xml)) {
        $failed += "$name (no JUnit report — collection error?)"
        return
    }
    [xml]$doc = Get-Content $xml
    $suite = if ($doc.testsuites) { $doc.testsuites.testsuite } else { $doc.testsuite }
    $bad = ([int]$suite.failures) + ([int]$suite.errors)
    $total = [int]$suite.tests
    if ($bad -gt 0) {
        $failed += "$name ($bad of $total failing)"
    } else {
        Write-Host "    $total passed" -ForegroundColor Green
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`nFAILED: $($failed -join '; ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll test files passed." -ForegroundColor Green
exit 0
