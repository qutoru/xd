@echo off
REM ============================================================
REM  Crypto Signal Bot - launcher (Phase 8, lightweight)
REM  Double-click to start the bot in live mode.
REM  Stop: press Ctrl+C or just close this window.
REM ============================================================

REM Run from the folder this .bat lives in, so relative paths work.
cd /d "%~dp0"
title Crypto Signal Bot (live)

echo ============================================
echo   Crypto Signal Bot - LIVE mode
echo   Sends a Telegram signal after each 15m bar
echo   Stop: Ctrl+C or close this window
echo ============================================
echo.

py main.py live

echo.
echo Bot stopped (exit code %errorlevel%).
pause
