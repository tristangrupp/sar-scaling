# Stops experiment 10's own GPU trainer at the end of the overnight window (only train.py --res ... --device cuda).
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'train\.py --res' -and $_.CommandLine -match '--device cuda' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
