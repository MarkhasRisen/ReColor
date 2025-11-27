param(
    [string]$PythonExe = "python"
)

Write-Host "== Phase 4b: Flickr scrape + Formal Test ==" -ForegroundColor Cyan

# --------------- Configuration ---------------
# Ensure Flickr API credentials are available
if (-not $env:FLICKR_KEY -or -not $env:FLICKR_SECRET) {
    Write-Host "ERROR: Please set environment variables FLICKR_KEY and FLICKR_SECRET before running." -ForegroundColor Red
    Write-Host "Example:" -ForegroundColor Yellow
    Write-Host "$env:FLICKR_KEY='YOUR_KEY'; $env:FLICKR_SECRET='YOUR_SECRET'" -ForegroundColor Yellow
    exit 1
}

$Classes = @(
    'beige_tan','black','blue','brown','cyan','gold','gray','green','lime',
    'magenta','navy','orange','pink','purple','red','silver','teal','white','yellow'
)

$RepoRoot      = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptsDir    = Join-Path $RepoRoot 'Color-Classification-CNN\scripts'
$ScraperScript = Join-Path $ScriptsDir 'flickr_scraper.py'
$TesterScript  = Join-Path $ScriptsDir 'run_formal_test.py'
$SummaryScript = Join-Path $ScriptsDir 'summarize_results.py'

$ModelPath     = Join-Path $RepoRoot 'Color-Classification-CNN\export\color_model_effb0_19cls_weighted_final_20251114_210537_fp32.tflite'
$LabelsPath    = Join-Path $RepoRoot 'Color-Classification-CNN\export\labels.json'

$TestOutDir    = Join-Path $RepoRoot 'real_world_phone_test'
$ResultsDir    = Join-Path $RepoRoot 'real_world_phone_results'

$ImageCountPerClass = 60
# CC licenses (exclude All Rights Reserved=0). Includes 1..6,9,10. See Flickr docs for details.
$Licenses = '1,2,3,4,5,6,9,10'
$PhoneTags = 'iphone,samsung,galaxy,pixel,mobile'

New-Item -ItemType Directory -Force -Path $TestOutDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

# --------------- Step 1: Scrape Test Set ---------------
Write-Host "[1/2] Starting Flickr scrape into: $TestOutDir" -ForegroundColor Cyan

foreach ($c in $Classes) {
    $classOut = Join-Path $TestOutDir $c
    New-Item -ItemType Directory -Force -Path $classOut | Out-Null
    Write-Host ("Scraping class: {0}" -f $c) -ForegroundColor Green

    & $PythonExe $ScraperScript `
        --search_term $c `
        --num_images $ImageCountPerClass `
        --licenses $Licenses `
        --tags $PhoneTags `
        --output_folder $classOut

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Scrape exited with code $LASTEXITCODE for class '$c'" -ForegroundColor Yellow
    }
}

# --------------- Step 2: Run Formal Test ---------------
Write-Host "[2/2] Scraping complete. Running formal test..." -ForegroundColor Cyan

& $PythonExe $TesterScript `
    --model $ModelPath `
    --labels $LabelsPath `
    --test_dir $TestOutDir `
    --results_dir $ResultsDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Formal test failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# --------------- Step 3: Summary ---------------
Write-Host "[3/3] Generating focused summary..." -ForegroundColor Cyan
& $PythonExe $SummaryScript `
    --results_dir $ResultsDir `
    --navy_blue_thresh 0.10 `
    --gold_yellow_thresh 0.08 `
    --silver_gray_thresh 0.10

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Summary exited with code $LASTEXITCODE" -ForegroundColor Yellow
}

Write-Host "All done. Results in: $ResultsDir (summary.txt included)" -ForegroundColor Cyan
