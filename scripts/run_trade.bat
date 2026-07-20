@echo off
REM One daily semi-auto trade cycle: sends the owner buttoned Accept/Ignore signals.
REM Run from the repo root so .env loads and data/ (pending, ledger, risk state) is shared.
cd /d "C:\Users\DVU\Desktop\xd"
py main.py trade --signals alx >> "data\trade_cron.log" 2>&1
