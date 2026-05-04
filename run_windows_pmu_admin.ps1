param(
  [int]$Cpu = 0,
  [int]$Outer = 500000,
  [int]$Repeats = 15
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$principal = [Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  throw "Run this script from an Administrator PowerShell. Windows blocks kernel ETW/PMU trace start at medium integrity."
}

$exe = Join-Path $PSScriptRoot "uops_research.exe"
$profile = Join-Path $PSScriptRoot "windows_pmu_sampling.wprp"
if (-not (Test-Path $exe)) { throw "Missing $exe" }
if (-not (Test-Path $profile)) { throw "Missing $profile" }

$status = (& wpr -status) -join "`n"
if ($status -notmatch "WPR is not recording") {
  throw "WPR is already recording. Stop the existing WPR session before running this script."
}

& wpr -pmcsources | Tee-Object -FilePath "pmu_sources_admin.txt"

function Invoke-WprBenchmark {
  param(
    [string]$ProfileName,
    [string]$Tag
  )

  $csv = "pmu_${Tag}_cpu${Cpu}.csv"
  $meta = "pmu_${Tag}_cpu${Cpu}_meta.txt"
  $etl = "pmu_${Tag}_cpu${Cpu}.etl"
  $started = $false

  try {
    & wpr -start "$profile!$ProfileName.Verbose" -filemode
    $started = $true
    & $exe $Cpu $Outer $Repeats > $csv 2> $meta
    & wpr -stop $etl "uops microbenchmark $Tag" -skipPdbGen
    $started = $false
  } finally {
    if ($started) {
      & wpr -cancel | Out-Null
    }
  }
}

Invoke-WprBenchmark -ProfileName "PMUCSwitch" -Tag "cswitch"
Invoke-WprBenchmark -ProfileName "PMUSampling" -Tag "sampling"

Write-Host "Wrote pmu_cswitch_cpu${Cpu}.etl, pmu_sampling_cpu${Cpu}.etl, and matching CSV/meta files."
