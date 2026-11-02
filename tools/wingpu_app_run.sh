#!/bin/sh
# wingpu App entry (wingpu-app.yaml run): start the bot PAUSED, as /work/scripts/run_bot.sh did.
# The user presses 開始 in the web UI (or writes {"run": true} to /work/control.json).
# exec: the bot is the process wgct started, so its stop signal (SIGTERM) reaches it and it lets go of the keys.
set -eu
cd "$(dirname "$0")/.."
PY=${MSBOT_PYTHON:-/work/venv-msbot/bin/python}
if [ ! -x "$PY" ]; then
    echo "no $PY: make the venv first (README.wingpu.md, 第一次安裝 venv)" >&2
    exit 2
fi
echo "{\"run\": false}" > /work/control.json
exec "$PY" -m src.engine.MapleStoryAutoLevelUp --cfg "${MSBOT_CFG:-wingpu_maple}"
