#!/system/bin/sh
# MCFI 守护进程：直接从模块目录运行，崩溃自动重启；入口加载日志写入 daemon.log
MODDIR=${0%/*}
DAEMON="$MODDIR/mcfi/bin/mcfid"
LOG="$MODDIR/daemon.log"

[ -f "$DAEMON" ] || exit 0
chmod 755 "$DAEMON" 2>/dev/null

# 等待开机完成
until [ "$(getprop sys.boot_completed)" = "1" ]; do sleep 2; done
sleep 3

echo "==== MCFI service.sh 拉起守护进程 $(date '+%F %T') ====" >> "$LOG"
echo "守护进程路径: $DAEMON" >> "$LOG"
echo "配置路径: $MODDIR/config.conf（升级自动保留）" >> "$LOG"

while true; do
  "$DAEMON" >> "$LOG" 2>&1
  echo "!! mcfid 退出(rc=$?)，3 秒后重启 $(date '+%F %T')" >> "$LOG"
  sleep 3
done
