Write-Host "======================================================"
Write-Host "      🏁 F1 PIPELINE MASTER EXECUTION SCRIPT 🏁       "
Write-Host "======================================================"

$env:PYTHONPATH="."

Write-Host "`n[1/6] Fetching Telemetry Data (Downloading FastF1 cache)..."
python scripts\download_data.py
if ($LASTEXITCODE -ne 0) { Write-Host "Error in Step 1"; exit 1 }

Write-Host "`n[2/6] Fitting ML Degradation Models..."
python scripts\fit_degradation.py
if ($LASTEXITCODE -ne 0) { Write-Host "Error in Step 2"; exit 1 }

Write-Host "`n[3/6] Training PPO Agent (This will take a few minutes)..."
python scripts\train_agent.py
if ($LASTEXITCODE -ne 0) { Write-Host "Error in Step 3"; exit 1 }

Write-Host "`n[4/6] Evaluating Model vs Baselines..."
python scripts\evaluate_agent.py
if ($LASTEXITCODE -ne 0) { Write-Host "Error in Step 4"; exit 1 }

Write-Host "`n[5/6] Generating Visualizations..."
python scripts\visualize_race.py
if ($LASTEXITCODE -ne 0) { Write-Host "Error in Step 5"; exit 1 }

Write-Host "`n[6/6] Launching Final Demo!"
python scripts\demo.py

Write-Host "`n✅ Pipeline completed successfully!"
