#!/bin/bash
# MCFI Vulkan shader → SPIR-V 头文件 重编脚本
# 用法：在装有 glslangValidator（Vulkan SDK 或 NDK 自带）的环境执行
#   cd src/shaders && ./gen_spv.sh
# 产物：覆盖本目录下的 *.spv 与 *_spv.h（C++ #include 这些头文件）
set -e
cd "$(dirname "$0")"

GV=$(command -v glslangValidator || true)
if [ -z "$GV" ]; then
  echo "未找到 glslangValidator；请安装 Vulkan SDK 或 NDK 后重试。" >&2
  exit 1
fi

gen() {
  local src="$1" stage="$2" outspv="$3" outh="$4"
  "$GV" --target-env vulkan1.1 -S "$stage" -V "$src" -o "$outspv"
  # 生成 C 头文件（unsigned char 数组）
  {
    echo "// 自动生成：$outh（glslangValidator 编译，勿手改）"
    echo "#pragma once"
    echo "#include <cstddef>"
    echo "static const unsigned char ${outspv%.spv}_spv[] = {"
    xxd -i < "$outspv" | sed 's/^  //'
    echo "};"
    echo "static const size_t ${outspv%.spv}_spv_len = sizeof(${outspv%.spv}_spv);"
  } > "$outh"
  echo "OK $src -> $outh"
}

# 图形管线
gen fs_interp.frag frag fs_interp.spv fs_interp_spv.h
gen fs_triangle.vert vert fs_triangle.spv fs_triangle_spv.h
gen mci.frag     frag mci.spv     mci_spv.h
# compute：fp32 与 fp16 两版（fp16 需 #version 450 + enable extension）
gen me0.comp     comp me0.spv     me0_spv.h
gen me1.comp     comp me1.spv     me1_spv.h
gen me0_f16.comp comp me0_f16.spv me0_f16_spv.h
gen me1_f16.comp comp me1_f16.spv me1_f16_spv.h
gen gmv0.comp    comp gmv0.spv    gmv0_spv.h
gen gmv1.comp    comp gmv1.spv    gmv1_spv.h
# 多尺度 L2 与 subgroup 加速（v2.5.0 新增）
gen me0_l2.comp  comp me0_l2.spv  me0_l2_spv.h
gen me1_l2.comp  comp me1_l2.spv  me1_l2_spv.h
gen me1_sub.comp comp me1_sub.spv me1_sub_spv.h
# L2 的 fp16 半精度版（设备支持 VK_KHR_shader_float16_int8 时优先）
gen me0_l2_f16.comp  comp me0_l2_f16.spv  me0_l2_f16_spv.h
gen me1_l2_f16.comp  comp me1_l2_f16.spv  me1_l2_f16_spv.h
echo "全部重编完成。"
