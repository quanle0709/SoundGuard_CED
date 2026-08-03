$ErrorActionPreference = "Stop"
$python = if (Test-Path -LiteralPath ".\.venv\Scripts\python.exe") {
    ".\.venv\Scripts\python.exe"
} else {
    "python"
}
$tests = @(
    "test_emergency_system.py",
    "test_audio_pipeline.py",
    "test_benchmark_runner.py",
    "test_benchmark_metrics.py",
    "test_speech_recognizer.py"
)
foreach ($test in $tests) {
    Write-Host "Running $test"
    & $python $test
    if ($LASTEXITCODE -ne 0) { throw "$test failed with exit code $LASTEXITCODE" }
}
Write-Host "All software tests passed. External model/network tests were not run."
