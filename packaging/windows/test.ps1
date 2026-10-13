# The installer on a real Windows: a silent install into a scratch folder,
# what it put there and where Windows lists it, the command line run from
# it, then a silent uninstall that leaves a user's own tool in tools\.
#
#   pwsh packaging/windows/test.ps1 -Setup <setup.exe> -Version 0.1.0
param(
    [Parameter(Mandatory)][string]$Setup,
    [Parameter(Mandatory)][string]$Version
)
$ErrorActionPreference = 'Stop'

function Check([bool]$ok, [string]$what) {
    if (-not $ok) { throw "FAILED: $what" }
    Write-Host "ok  $what"
}

$scratch = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$dir = Join-Path $scratch 'omnifrog-install-test'
if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
$link = Join-Path ([Environment]::GetFolderPath('Programs')) 'Omnifrog.lnk'
$key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Omnifrog'

# /D= is last and unquoted, as NSIS reads it.
$install = Start-Process -FilePath $Setup -ArgumentList '/S', "/D=$dir" -Wait -PassThru
Check ($install.ExitCode -eq 0) "silent install exits 0 (got $($install.ExitCode))"

foreach ($name in 'omnifrog.exe', 'omnifrog-gui.exe', 'README.md', 'uninstall.exe', 'tools') {
    Check (Test-Path (Join-Path $dir $name)) "$name is installed"
}
Check (Test-Path $link) 'the Start-menu shortcut is there'
Check ((Get-ItemProperty $key).DisplayVersion -eq $Version) "Windows lists version $Version"
Check ((Get-ItemProperty $key).InstallLocation -eq $dir) 'Windows knows where it is installed'

$said = & (Join-Path $dir 'omnifrog.exe') --version
Check ($LASTEXITCODE -eq 0) 'omnifrog.exe --version runs'
Check ("$said" -match [regex]::Escape($Version)) "it says $Version (said: $said)"

# What someone put in tools\ is theirs.
$kept = Join-Path $dir 'tools\kept.txt'
Set-Content -Path $kept -Value 'a tool of the user'

# The uninstaller copies itself to a temporary folder and returns at once:
# wait for the programs to go rather than for the process.
Start-Process -FilePath (Join-Path $dir 'uninstall.exe') -ArgumentList '/S' -Wait
$deadline = (Get-Date).AddSeconds(60)
while ((Test-Path (Join-Path $dir 'omnifrog.exe')) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
}
Start-Sleep -Seconds 2

foreach ($name in 'omnifrog.exe', 'omnifrog-gui.exe', 'README.md', 'uninstall.exe') {
    Check (-not (Test-Path (Join-Path $dir $name))) "$name is removed"
}
Check (-not (Test-Path $link)) 'the Start-menu shortcut is removed'
Check (-not (Test-Path $key)) 'the uninstall entry is removed'
Check (Test-Path $kept) "the user's tool in tools\ is still there"

Remove-Item -Recurse -Force $dir
Write-Host 'installer test passed'
