#!/bin/sh
# wingpu App entry (wingpu-app.yaml run): start the bot PAUSED, as /work/scripts/run_bot.sh did.
# The user presses 開始 in the web UI (or writes {"run": true} to /work/control.json).
# exec: the bot is the process wgct started, so its stop signal (SIGTERM) reaches it and it lets go of the keys.
set -eu
cd "$(dirname "$0")/.."
# the python: MSBOT_PYTHON, else the App's own venv (an installed App: wgct sets VIRTUAL_ENV), else the venv made
# by hand for the linked checkout (/work/venv-msbot)
if [ -n "${MSBOT_PYTHON:-}" ]; then
    PY=$MSBOT_PYTHON
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY=$VIRTUAL_ENV/bin/python
else
    PY=/work/venv-msbot/bin/python
fi
if [ ! -x "$PY" ]; then
    echo "no $PY: make the venv first (README.wingpu.md, 第一次安裝 venv), or install the App from git" >&2
    exit 2
fi
echo "{\"run\": false}" > /work/control.json
exec "$PY" -m src.engine.MapleStoryAutoLevelUp --cfg "${MSBOT_CFG:-wingpu_maple}"
