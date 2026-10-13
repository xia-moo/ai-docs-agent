$ErrorActionPreference = 'Stop'
$taskPython = 'D:\anaconda\envs\AIDocsAgent_py312\python.exe'
Set-Location -LiteralPath $PSScriptRoot
& $taskPython (Join-Path $PSScriptRoot 'app.py')

