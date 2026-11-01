#!/bin/sh
# wingpu_sync.sh - put a commit of this repo into a wingpu script container.
#
#   tools/wingpu_sync.sh [REF] [CONTAINER] [SSH_HOST]
#     REF        commit / branch to deploy          (default: HEAD)
#     CONTAINER  docker container on the wingpu host (default: wingpu-ct-maple)
#     SSH_HOST   wingpu host; "" = docker is local   (default: test@192.168.66.63)
#
# The container's /work/MapleStoryAutoLevelUp becomes a git checkout of REF
# (detached HEAD): tracked files exactly as in the commit, untracked files
# (log/, screenshot/) kept. A plain copy from before is moved aside to
# MapleStoryAutoLevelUp.old-<time> the first time. User state lives outside the
# code directory (/work/msbot_config.yaml, /work/control.json, /work/debug/),
# so deploying never touches it. The venv (/work/venv-msbot) is not changed.
# Commit first: uncommitted changes are not deployed.
set -eu
REF=${1:-HEAD}
CT=${2:-wingpu-ct-maple}
HOST=${3-test@192.168.66.63}
cd "$(dirname "$0")/.."
SHA=$(git rev-parse --verify "$REF^{commit}")
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "note: uncommitted changes are NOT deployed (deploying $SHA)" >&2
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git branch -f wingpu-sync-tmp "$SHA" >/dev/null
git bundle create "$TMP/msbot.bundle" wingpu-sync-tmp >/dev/null 2>&1
git branch -D wingpu-sync-tmp >/dev/null

REMOTE='set -eu
D=/work/MapleStoryAutoLevelUp
B=/work/.msbot_sync.bundle
cat > "$B"
if [ ! -d "$D/.git" ]; then
    if [ -d "$D" ]; then
        mv "$D" "$D.old-$(date +%Y%m%d-%H%M%S)"
    fi
    mkdir -p "$D"
    git -C "$D" init -q
fi
git -C "$D" fetch -q "$B" "wingpu-sync-tmp:refs/remotes/sync/deployed" --force
git -C "$D" -c advice.detachedHead=false checkout -q -f --detach refs/remotes/sync/deployed
rm -f "$B"
echo "deployed $(git -C "$D" rev-parse --short HEAD): $(git -C "$D" log -1 --format=%s)"'

if [ -n "$HOST" ]; then
    ssh "$HOST" "docker exec -i $CT sh -c '$REMOTE'" < "$TMP/msbot.bundle"
else
    docker exec -i "$CT" sh -c "$REMOTE" < "$TMP/msbot.bundle"
fi
echo "restart the bot to load it (web UI: 重新啟動, or POST ./restart)"
