$ErrorActionPreference = "Stop"
$python = if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else {
    "python"
}
$tests = @(
    "tests.test_emergency_system",
    "tests.test_audio_pipeline",
    "tests.test_benchmark_runner",
    "tests.test_benchmark_metrics",
    "tests.test_speech_recognizer"
)
foreach ($test in $tests) {
    Write-Host "Running $test"
    & $python -m $test
    if ($LASTEXITCODE -ne 0) { throw "$test failed with exit code $LASTEXITCODE" }
}
Write-Host "All software tests passed. External model/network tests were not run."
