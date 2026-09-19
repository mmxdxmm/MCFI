#!/system/bin/sh
# MCFI 卸载清理
pkill -f mcfid 2>/dev/null
rm -f /data/local/tmp/mcfi_config.conf
rm -f /data/local/tmp/framegen_config.conf   # 兼容清理旧版 FrameGen 镜像
# 模块目录 /data/adb/modules/mcfi 由 Magisk 自动清理
