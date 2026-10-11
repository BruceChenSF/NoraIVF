#!/bin/bash
# pi_bg.sh - 在服务器上以后台方式启动 pi（非交互），输出写入日志文件
# 用法: pi_bg.sh <prompt_base64> [timeout_seconds] [workdir]
# 输出: TASK=<id>  PID=<pid>  LOG=<path>
set -u
WD="${3:-/home/csf/projects/NoraIVF}"
cd "$WD" || exit 1
mkdir -p results/pi_logs
TS=$(date +%Y%m%d_%H%M%S)
TF="results/pi_logs/task_${TS}.txt"
LF="results/pi_logs/pi_${TS}.log"
printf '%s' "$1" | base64 -d > "$TF"
TMO="${2:-3600}"
export PATH="$HOME/.npm-global/bin:$PATH"

# 后台运行：setsid 脱离会话，nohup 忽略挂断，stdin 重定向 /dev/null（非 TTY 必需）
setsid nohup bash -c "timeout $TMO pi -p --no-session \"\$(cat '$TF')\" < /dev/null 2>&1; echo \"[pi] exit=\$? at \$(date +%F_%T)\"" > "$LF" 2>&1 < /dev/null &
PID=$!
echo "TASK=$TS"
echo "PID=$PID"
echo "LOG=$LF"
sleep 2
echo "ALIVE=$(ps -p $PID > /dev/null 2>&1 && echo yes || echo no)"
