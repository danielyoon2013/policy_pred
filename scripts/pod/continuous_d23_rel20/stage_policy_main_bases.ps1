param(
  [string]$HostName = "103.207.149.128",
  [int]$Port = 14317,
  [string]$KeyPath = "C:\Users\danielyoon\.ssh\id_ed25519",
  [string]$KnownHosts = "C:\tmp\runpod_known_hosts",
  [string]$RemoteRoot = "/workspace/continuous_d23_rel20",
  [string]$CheckpointDir = "D:\hist_LLM\continual\model\year_checkpoints\d23",
  [int]$Throttle = 4
)

$ErrorActionPreference = "Stop"

$pairs = @(
  @{ year = 1931; step = "150975" },
  @{ year = 1935; step = "155694" },
  @{ year = 1939; step = "159519" },
  @{ year = 1943; step = "163268" },
  @{ year = 1947; step = "167490" },
  @{ year = 1951; step = "170683" },
  @{ year = 1955; step = "174444" },
  @{ year = 1959; step = "177756" },
  @{ year = 1963; step = "181337" },
  @{ year = 1967; step = "184141" },
  @{ year = 1971; step = "187795" },
  @{ year = 1975; step = "191532" },
  @{ year = 1979; step = "195328" },
  @{ year = 1983; step = "199822" },
  @{ year = 1987; step = "204738" },
  @{ year = 1991; step = "211381" },
  @{ year = 1995; step = "219010" },
  @{ year = 1999; step = "226640" },
  @{ year = 2003; step = "234269" },
  @{ year = 2007; step = "241898" },
  @{ year = 2011; step = "249528" },
  @{ year = 2015; step = "257157" },
  @{ year = 2019; step = "264787" }
)

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

$yearList = ($pairs | ForEach-Object { $_.year }) -join " "
$remotePrep = @"
set -e
mkdir -p '$RemoteRoot/shared_tokenizer'
cp -f '$RemoteRoot/bases/1950/tokenizer/tokenizer.pkl' '$RemoteRoot/shared_tokenizer/tokenizer.pkl'
cp -f '$RemoteRoot/bases/1950/tokenizer/token_bytes.pt' '$RemoteRoot/shared_tokenizer/token_bytes.pt'
for y in $yearList; do
  mkdir -p '$RemoteRoot/bases/'"`$y"'/base_checkpoints/d23'
  ln -sfn '$RemoteRoot/shared_tokenizer' '$RemoteRoot/bases/'"`$y"'/tokenizer'
done
"@

ssh @sshBase $remotePrep

$jobs = @()
foreach ($pair in $pairs) {
  while (($jobs | Where-Object { $_.State -eq "Running" }).Count -ge $Throttle) {
    $finished = Wait-Job -Job $jobs -Any
    Receive-Job -Job $finished
    if ($finished.State -ne "Completed") {
      throw "Upload job failed for $($finished.Name)"
    }
  }

  $year = $pair.year
  $step = $pair.step
  $model = Join-Path $CheckpointDir "model_$step.pt"
  $meta = Join-Path $CheckpointDir "meta_$step.json"
  if (!(Test-Path -LiteralPath $model)) { throw "Missing $model" }
  if (!(Test-Path -LiteralPath $meta)) { throw "Missing $meta" }

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

$verifyRows = ($pairs | ForEach-Object { "{0}:{1}" -f $_.year,$_.step }) -join " "
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

Write-Output "STAGING_COMPLETE years=$($pairs.Count)"
