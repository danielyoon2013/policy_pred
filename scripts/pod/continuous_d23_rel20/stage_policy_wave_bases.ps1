param(
  [Parameter(Mandatory = $true)]
  [int[]]$Years,
  [string]$HostName = "103.207.149.128",
  [int]$Port = 14317,
  [string]$KeyPath = "C:\Users\danielyoon\.ssh\id_ed25519",
  [string]$KnownHosts = "C:\tmp\runpod_known_hosts",
  [string]$RemoteRoot = "/workspace/continuous_d23_rel20",
  [string]$CheckpointDir = "D:\hist_LLM\continual\model\year_checkpoints\d23",
  [string]$TokenizerDir = "D:\hist_LLM\continual\data\tokenizer",
  [int]$FirstCheckpointYear = 1850,
  [int]$Throttle = 4,
  [switch]$ClearRemoteBases
)

$ErrorActionPreference = "Stop"

$models = Get-ChildItem -Path $CheckpointDir -Filter "model_*.pt" | Sort-Object Name
$metas = Get-ChildItem -Path $CheckpointDir -Filter "meta_*.json" | Sort-Object Name
if ($models.Count -ne $metas.Count) {
  throw "Mismatched model/meta counts: $($models.Count) models vs $($metas.Count) metas"
}

$stepByYear = @{}
for ($i = 0; $i -lt $models.Count; $i++) {
  $year = $FirstCheckpointYear + $i
  $step = [regex]::Match($models[$i].BaseName, "model_(\d+)").Groups[1].Value
  $metaStep = [regex]::Match($metas[$i].BaseName, "meta_(\d+)").Groups[1].Value
  if ($step -ne $metaStep) {
    throw "Model/meta step mismatch at sorted index ${i}: $step vs $metaStep"
  }
  $stepByYear[$year] = $step
}

$tokFiles = @(
  (Join-Path $TokenizerDir "tokenizer.pkl"),
  (Join-Path $TokenizerDir "token_bytes.pt")
)
foreach ($path in $tokFiles) {
  if (!(Test-Path -Path $path)) { throw "Missing tokenizer file: $path" }
}

$sshBase = @(
  "-T",
  "-i", $KeyPath,
  "-p", "$Port",
  "-o", "BatchMode=yes",
  "-o", "StrictHostKeyChecking=no",
  "-o", "UserKnownHostsFile=$KnownHosts",
  "root@$HostName"
)
$scpBase = @(
  "-i", $KeyPath,
  "-P", "$Port",
  "-o", "BatchMode=yes",
  "-o", "StrictHostKeyChecking=no",
  "-o", "UserKnownHostsFile=$KnownHosts"
)

$clear = if ($ClearRemoteBases) { "rm -rf '$RemoteRoot/bases';" } else { "" }
$yearList = ($Years | ForEach-Object { "$_" }) -join " "
$remotePrep = @"
set -e
$clear
mkdir -p '$RemoteRoot/shared_tokenizer' '$RemoteRoot/bases'
for y in $yearList; do
  mkdir -p '$RemoteRoot/bases/'"`$y"'/base_checkpoints/d23'
  ln -sfn '$RemoteRoot/shared_tokenizer' '$RemoteRoot/bases/'"`$y"'/tokenizer'
done
"@

ssh @sshBase $remotePrep
if ($LASTEXITCODE -ne 0) { throw "remote prep failed" }

$tokDest = "root@${HostName}:$RemoteRoot/shared_tokenizer/"
& scp @scpBase $tokFiles[0] $tokFiles[1] $tokDest
if ($LASTEXITCODE -ne 0) { throw "tokenizer upload failed" }

$jobs = @()
foreach ($year in $Years) {
  if (!$stepByYear.ContainsKey($year)) { throw "No checkpoint step mapped for year $year" }
  $step = $stepByYear[$year]
  $model = Join-Path $CheckpointDir "model_$step.pt"
  $meta = Join-Path $CheckpointDir "meta_$step.json"
  if (!(Test-Path -Path $model)) { throw "Missing $model" }
  if (!(Test-Path -Path $meta)) { throw "Missing $meta" }

  while (($jobs | Where-Object { $_.State -eq "Running" }).Count -ge $Throttle) {
    $finished = Wait-Job -Job $jobs -Any
    Receive-Job -Job $finished
    if ($finished.State -ne "Completed") {
      throw "Upload job failed for $($finished.Name)"
    }
  }

  $dest = "root@${HostName}:$RemoteRoot/bases/$year/base_checkpoints/d23/"
  $args = @($scpBase + @($model, $meta, $dest))
  $jobs += Start-Job -Name "stage_$year" -ScriptBlock {
    param($scpArgs, $year, $step)
    Write-Output "START $year step=$step"
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) {
      throw "scp failed for $year step=$step"
    }
    Write-Output "DONE $year step=$step"
  } -ArgumentList $args, $year, $step
}

while (($jobs | Where-Object { $_.State -eq "Running" }).Count -gt 0) {
  $finished = Wait-Job -Job $jobs -Any
  Receive-Job -Job $finished
  if ($finished.State -ne "Completed") {
    throw "Upload job failed for $($finished.Name)"
  }
}

$jobs | Where-Object { $_.State -ne "Running" } | ForEach-Object {
  Receive-Job -Job $_
  if ($_.State -ne "Completed") {
    throw "Upload job failed for $($_.Name)"
  }
}

$verifyRows = ($Years | ForEach-Object { "{0}:{1}" -f $_,$stepByYear[$_] }) -join " "
$verify = @"
set -e
missing=0
for row in $verifyRows; do
  y=`${row%%:*}
  s=`${row##*:}
  test -f '$RemoteRoot/bases/'"`$y"'/base_checkpoints/d23/model_'"`$s"'.pt' || { echo missing model "`$row"; missing=1; }
  test -f '$RemoteRoot/bases/'"`$y"'/base_checkpoints/d23/meta_'"`$s"'.json' || { echo missing meta "`$row"; missing=1; }
  test -f '$RemoteRoot/bases/'"`$y"'/tokenizer/tokenizer.pkl' || { echo missing tokenizer "`$row"; missing=1; }
done
exit `$missing
"@
ssh @sshBase $verify
if ($LASTEXITCODE -ne 0) { throw "remote verify failed" }

Write-Output "STAGING_COMPLETE years=$($Years.Count) $($Years -join ',')"
