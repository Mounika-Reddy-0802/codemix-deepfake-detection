@echo off
rem Start the final-review demo: the server on the final recipe, three detector lanes,
rem the demo calls and the stage player, then open the guided flow in the browser.
rem
rem   scripts\run_demo.bat            (from the project root, or double-click it)
rem
rem Everything the demo needs lives under DFD_DATA (default C:\dfdata), outside git:
rem   checkpoints\final_s1234_best.pt, checkpoints\lora_norm_rvc_channel_best.pt,
rem   demo_calls_v2\ (or demo_clips_final\), stage_player\
rem The English-only checkpoint comes from Git LFS: checkpoints\baseline\best.pt.
rem Close this window to stop the server.

setlocal
cd /d "%~dp0.."

if not defined DFD_DATA set "DFD_DATA=C:\dfdata"
if not defined DFD_PORT set "DFD_PORT=8000"

rem ---------------------------------------------------------------- Python
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" if exist "%DFD_DATA%\venv\Scripts\python.exe" set "PY=%DFD_DATA%\venv\Scripts\python.exe"

rem ---------------------------------------------------------------- the live model
set "DFD_CHECKPOINT=%DFD_DATA%\checkpoints\final_s1234_best.pt"
set "DFD_THRESHOLD_FILE=configs\threshold_final.yaml"
set "DFD_DEVICE=cpu"
set "CUDA_VISIBLE_DEVICES="

rem ---------------------------------------------------------------- the demo calls
set "DEMO_CLIPS_DIR=%DFD_DATA%\demo_calls_v2"
if not exist "%DEMO_CLIPS_DIR%\demo_clips.json" set "DEMO_CLIPS_DIR=%DFD_DATA%\demo_clips_final"
set "STAGE_PLAYER_DIR=%DFD_DATA%\stage_player"
set "TUNNEL_URL_FILE=%DFD_DATA%\tunnel_url.txt"

rem ---------------------------------------------------------------- scene 2: three detectors
set "DFD_LANES=S1 (English only)=checkpoints\baseline\best.pt=configs\threshold.yaml,Deployed adapter=%DFD_DATA%\checkpoints\lora_norm_rvc_channel_best.pt=configs\threshold.yaml,Final recipe=%DFD_DATA%\checkpoints\final_s1234_best.pt=configs\threshold_final.yaml"

rem the larger encoder (SSL-AASIST, #63) joins as a fourth lane when its weights are on this laptop;
rem its threshold was chosen on the same dev clips by the same rule (configs	hreshold_ssl_aasist.yaml)
set "SSL_CKPT=%DFD_DATA%\checkpoints\ssl_aasist_final_s1234_fp16.pt"
if not exist "%SSL_CKPT%" set "SSL_CKPT=%DFD_DATA%\checkpoints\ssl_aasist_final_s1234.pt"
if exist "%SSL_CKPT%" set "DFD_LANES=%DFD_LANES%,SSL-AASIST (next candidate)=%SSL_CKPT%=configs	hreshold_ssl_aasist.yaml"
if not exist "%SSL_CKPT%" echo note: no SSL-AASIST weights in %DFD_DATA%\checkpoints, scene 2 runs three lanes

rem wav2vec2-base is cached after the first run: never wait on the network in the room
if exist "%USERPROFILE%\.cache\huggingface\hub\models--facebook--wav2vec2-base" set "HF_HUB_OFFLINE=1"

rem ---------------------------------------------------------------- checks
if not exist "%DFD_CHECKPOINT%" echo MISSING: %DFD_CHECKPOINT%
if not exist "%DFD_DATA%\checkpoints\lora_norm_rvc_channel_best.pt" echo MISSING: %DFD_DATA%\checkpoints\lora_norm_rvc_channel_best.pt
if not exist "%DEMO_CLIPS_DIR%\demo_clips.json" echo MISSING: demo calls in %DFD_DATA%\demo_calls_v2 or demo_clips_final
if not exist "%STAGE_PLAYER_DIR%\stage_player.json" echo NOTE: no stage player yet: python scripts\prepare_stage_player.py

echo.
echo   Final-review demo on http://localhost:%DFD_PORT%/demo
echo   model      %DFD_CHECKPOINT%
echo   threshold  %DFD_THRESHOLD_FILE%
echo   calls      %DEMO_CLIPS_DIR%
echo   python     %PY%
echo.

rem open the guided flow as soon as the server answers (the models finish loading meanwhile);
rem set DFD_NO_BROWSER=1 to start the server alone
if not defined DFD_NO_BROWSER start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='http://localhost:%DFD_PORT%'; for($i=0;$i -lt 120;$i++){ try { Invoke-RestMethod \"$u/api/health\" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }; $c=@(\"$env:ProgramFiles\Google\Chrome\Application\chrome.exe\", \"${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe\", \"$env:LocalAppData\Google\Chrome\Application\chrome.exe\") | Where-Object { Test-Path $_ } | Select-Object -First 1; if ($c) { Start-Process $c \"$u/demo\" } else { Start-Process \"$u/demo\" }"

"%PY%" -m uvicorn live_call.server:app --host 0.0.0.0 --port %DFD_PORT%
endlocal
