@echo off
REM Run this once to create a desktop shortcut for the bot launcher.
cd /d "%~dp0"

powershell -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$desktop = [System.Environment]::GetFolderPath('Desktop');" ^
  "$shortcut = $ws.CreateShortcut($desktop + '\Kalshi AI Bot.lnk');" ^
  "$shortcut.TargetPath = '%~dp0launch_bot.bat';" ^
  "$shortcut.WorkingDirectory = '%~dp0';" ^
  "$shortcut.WindowStyle = 1;" ^
  "$shortcut.IconLocation = '%SystemRoot%\system32\SHELL32.dll,13';" ^
  "$shortcut.Description = 'Launch Kalshi AI Trading Bot (Beast + Lean + Dashboard)';" ^
  "$shortcut.Save();"

echo.
echo  Desktop shortcut created: "Kalshi AI Bot"
echo  Double-click it anytime to start the bot.
echo.
pause
