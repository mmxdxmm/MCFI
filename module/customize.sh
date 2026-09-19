#!/system/bin/sh
# MCFI 安装脚本：文件全部保留在模块目录，不额外建 /data/adb 子目录
# 升级时自动保留旧配置（含旧版 FrameGen 模块 /data/adb/modules/framegen）

ui_print "- 设备 ABI: $ARCH"
if [ "$ARCH" != "arm64" ]; then
  ui_print "! 本模块仅支持 arm64-v8a 设备"
  abort "! 安装中止"
fi

NEW=/data/adb/modules/mcfi
OLD=/data/adb/modules/framegen

# 升级安装：优先保留 MCFI 自身旧配置，其次兼容旧 FrameGen 配置
if [ -f "$NEW/config.conf" ]; then
  ui_print "- 检测到旧配置，升级保留"
  cp -f "$NEW/config.conf" "$MODPATH/config.conf"
elif [ -f "$OLD/config.conf" ]; then
  ui_print "- 检测到旧版 FrameGen 配置，迁移保留"
  cp -f "$OLD/config.conf" "$MODPATH/config.conf"
fi

chmod 755 "$MODPATH/mcfi/bin/mcfid"
chmod 644 "$MODPATH/config.conf" "$MODPATH/mcfi/panel/index.html"
chmod 755 "$MODPATH/service.sh" "$MODPATH/post-fs-data.sh" "$MODPATH/uninstall.sh" "$MODPATH/customize.sh"

# 运行时事件目录：app 进程（非 root）直接写事件文件到这里，daemon 汇总到面板
mkdir -p "$MODPATH/runtime"
chmod 0777 "$MODPATH/runtime"

# 兜底配置镜像（守护进程启动后也会自动刷新）
cp -f "$MODPATH/config.conf" /data/local/tmp/mcfi_config.conf 2>/dev/null
chmod 644 /data/local/tmp/mcfi_config.conf 2>/dev/null

ui_print "- 已就位: zygisk/arm64-v8a.so + mcfi/bin/mcfid + 控制面板（均在模块目录内）"
ui_print "- 守护进程直接从模块目录运行，升级时自动保留配置"
ui_print "- 重启后生效，状态见 Magisk 模块页描述行"
ui_print "- 控制面板: 手机浏览器打开 http://127.0.0.1:4400"
ui_print "  (端口被占用时自动顺延，实际端口以模块描述为准)"
ui_print " "
ui_print "  注意: 需在 Magisk 设置中启用 Zygisk；KernelSU 环境配合 Zygisk Next 同样可用"
