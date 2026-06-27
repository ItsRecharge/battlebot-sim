@echo off
REM Launch the BattleBot Damage Simulator from source (no compile needed).
REM Double-click this file, or run it from a terminal.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m battlebot_sim
