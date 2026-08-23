# Emergency V3 recovery

Known-good V2 commit: 24346d879634f65eeb8a647bdba0ac84e249c63d; V2 branch: improve-ced-safety-v2. The source/snapshot evidence is under benchmark_results/improvement and was SHA-256 verified before V3 work.

Disable V3 immediately in the current shell:

    Remove-Item Env:SOUNDGUARD_ENABLE_EMERGENCY_V3 -ErrorAction SilentlyContinue

The --emergency-v3 CLI flag must also be omitted. With both absent, no worker starts and behavior matches frozen V2.

Return to the V2 branch while preserving the working tree:

    git switch improve-ced-safety-v2

Remove only the ignored experimental environment/checkpoint if disk recovery is needed (this is destructive and must be run intentionally from the repository root):

    Remove-Item -LiteralPath "benchmark_data\external\efficientsed_venv" -Recurse -Force
    Remove-Item -LiteralPath "benchmark_data\external\efficientsed_repo" -Recurse -Force

Removing those paths does not remove CED-Tiny. If V3 is accidentally enabled afterward, the adapter logs one failure and continues with V2.
