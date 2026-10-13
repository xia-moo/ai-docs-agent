$ErrorActionPreference = 'Stop'
$taskPidFile = Join-Path $PSScriptRoot 'logs\server.pid'
if (-not (Test-Path -LiteralPath $taskPidFile)) { Write-Output 'No tracked application process.'; exit 0 }
$taskOwnedPid = [int](Get-Content -LiteralPath $taskPidFile)
$taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $taskOwnedPid"
if (-not $taskProcess) { Write-Output 'Application is already stopped.'; exit 0 }
$taskCommand = $taskProcess.CommandLine.Replace('/', '\')
$taskIsApplication = $taskCommand.Contains('uvicorn ai_docs_agent.api:app') -or $taskCommand.Contains((Join-Path $PSScriptRoot 'app.py'))
if ($taskProcess.ExecutablePath -ne 'D:\anaconda\envs\AIDocsAgent_py312\python.exe' -or -not $taskIsApplication) {
    throw 'The recorded process does not belong to this application; nothing was stopped.'
}
Stop-Process -Id $taskOwnedPid -Force
Write-Output 'AI Docs Agent stopped.'
