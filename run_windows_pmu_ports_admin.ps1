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
$profile = Join-Path $PSScriptRoot "windows_pmu_ports_pcore_experimental.wprp"
if (-not (Test-Path $exe)) { throw "Missing $exe" }
if (-not (Test-Path $profile)) { throw "Missing $profile" }
if ($Cpu -ne 0) {
  Write-Warning "This profile uses P-core port events. CPU 0 was detected as a P-core in our run; other CPU ids may not be P-cores."
}

$status = (& wpr -status) -join "`n"
if ($status -notmatch "WPR is not recording") {
  throw "WPR is already recording. Stop the existing WPR session before running this script."
}

$csv = "pmu_pcore_ports_cpu${Cpu}.csv"
$meta = "pmu_pcore_ports_cpu${Cpu}_meta.txt"
$etl = "pmu_pcore_ports_cpu${Cpu}.etl"
$started = $false

try {
  & wpr -start "$profile!PCorePorts.Verbose" -filemode
  $started = $true
  & $exe $Cpu $Outer $Repeats > $csv 2> $meta
  & wpr -stop $etl "uops microbenchmark P-core port counters" -skipPdbGen
  $started = $false
} finally {
  if ($started) {
    & wpr -cancel | Out-Null
  }
}

Write-Host "Wrote $etl and matching CSV/meta files."
