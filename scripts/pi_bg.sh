#!/bin/bash
# pi_bg.sh - start pi in the background on a server (non-interactive), logging to a file
# usage: pi_bg.sh <prompt_base64> [timeout_seconds] [workdir]
# prints: TASK=<id>  PID=<pid>  LOG=<path>
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WD="${3:-$REPO_ROOT}"
cd "$WD" || exit 1
mkdir -p results/pi_logs
TS=$(date +%Y%m%d_%H%M%S)
TF="results/pi_logs/task_${TS}.txt"
LF="results/pi_logs/pi_${TS}.log"
printf '%s' "$1" | base64 -d > "$TF"
TMO="${2:-3600}"
export PATH="$HOME/.npm-global/bin:$PATH"

# detached: setsid leaves the session, nohup ignores SIGHUP, stdin from /dev/null
# (required when there is no controlling TTY)
setsid nohup bash -c "timeout $TMO pi -p --no-session \"\$(cat '$TF')\" < /dev/null 2>&1; echo \"[pi] exit=\$? at \$(date +%F_%T)\"" > "$LF" 2>&1 < /dev/null &
PID=$!
echo "TASK=$TS"
echo "PID=$PID"
echo "LOG=$LF"
sleep 2
echo "ALIVE=$(ps -p $PID > /dev/null 2>&1 && echo yes || echo no)"
