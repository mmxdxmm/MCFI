#!/bin/bash
# MCFI 一键构建脚本
set -e
NDK=${NDK:-/opt/android-ndk-r27c}
ROOT=$(cd "$(dirname "$0")" && pwd)
BUILD=/tmp/mcfi-build
OUT=/tmp/mcfi-out
MODULE=$OUT/module

rm -rf "$BUILD" "$OUT"
mkdir -p "$BUILD" "$MODULE/zygisk" "$MODULE/mcfi/bin" "$MODULE/mcfi/panel"

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

# 组装 Magisk 模块目录
cp "$BUILD/arm64-v8a.so"   "$MODULE/zygisk/arm64-v8a.so"
cp "$BUILD/mcfid"      "$MODULE/mcfi/bin/mcfid"
cp "$ROOT/panel/index.html" "$MODULE/mcfi/panel/index.html"
cp "$ROOT/module/"module.prop "$ROOT/module/"*.sh "$ROOT/module/config.conf" "$MODULE/"
cp "$ROOT/README.md" "$MODULE/README.md" 2>/dev/null || true

cd "$OUT/module"
ZIPNAME="MCFI-补帧模块-v2.6.15-arm64.zip"
rm -f "../$ZIPNAME"
zip -q -r -9 "../$ZIPNAME" .
echo ""
echo "构建完成: $OUT/$ZIPNAME"
ls -la "$OUT/$ZIPNAME"
unzip -l "$OUT/$ZIPNAME"
