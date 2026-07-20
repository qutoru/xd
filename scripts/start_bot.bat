@echo off
REM ============================================================
REM  Crypto Signal Bot launcher.
REM  While this window is open, the bot is LIVE: it answers
REM  /start, delivers owner Accept/Ignore signals, and places
REM  PAPER orders when you tap Accept. Close the window to stop.
REM ============================================================
title Crypto Signal Bot
cd /d "C:\Users\DVU\Desktop\xd"

:loop
echo(
echo ============================================================
echo   Crypto Signal Bot is RUNNING (listener).
echo   Keep this window open. Close it to stop the bot.
echo ============================================================
echo(
py main.py bot
echo(
echo Listener stopped (exit code %errorlevel%). Restarting in 5s...
echo Press Ctrl+C now to quit instead.
timeout /t 5 >nul
goto loop
