# MCFI v2.4.4 优化实施记录

## 一、已完成（本机改源码，待 NDK r27c 环境 `./build.sh` 编译 + 真机 logcat 验证）

### 步骤A — GLES 视频模式 L2：1/16 → 1/8 ✅
- `payload.cpp`：把 L2 相对 L1 的下采样倍率从硬编码 ×4 改为运行时参数 `uLv2`。
  - `run_motion_estimation(...)` 新增尾参 `int lv2 = 4`；`R2 = max(1, range/lv2)`；
    给 `progME[0]/[1]` 传 `uLv2`。
  - `CS_ME0_SRC`/`CS_ME1_SRC` 里所有 `g*4`、`dsW2=(dsW+3)/4`、`center*4.0` 改为按 `uLv2`。
  - 游戏异步路径 `AsyncCtx` 不传（默认 lv2=4，保持 1/16，省功耗）。
  - 视频 `VideoCtx`：`dsW2=dsW/2`（1/8），调用传 `lv2=2`。
- 要点：覆盖范围不变（L2 位移统一放大到 ±range L1 像素），只是粗层分辨率更高；
  游戏/视频两套资源尺寸不串。

### 步骤B — 视频模式补 Vulkan API ✅
- 关键修复：`=video` 原本 `want_vk=false`，不挂 Vulkan hook → Vulkan 渲染的视频 App 完全不插帧。
  改为 `want_vk = (g_backend==0 || ==2 || ==3)`，视频模式同时挂 GLES+Vulkan，哪个渲染走哪个。
- Vulkan 侧 `=video` 已有小 surface 守卫（width<500 禁用），复用现有单尺度 1/4 异步管线。

### 步骤C — SurfaceFlinger 时间戳注入 ✅
- GLES：加载 `eglPresentationTimeANDROID`（EGL_ANDROID_presentation_time）。
  生成帧 swap 前 `pts=+1 vsync`；真实帧 swap（`mcfi_swap_real`）前 `pts=+2 vsync` 并采样周期。
  vsync 周期用相邻真实帧到达时间差分平滑估计（默认 120Hz=8.33ms）。
- Vulkan：`my_CreateDevice` 枚举并启用 `VK_GOOGLE_display_timing`；
  worker 生成帧 present 挂 `VkPresentTimesInfoGOOGLE`（+1 vsync）；
  `my_QueuePresentKHR` 末尾真实帧 present 挂 +2 vsync（按 gen_seq 配对）。
- 扩展/函数缺失 → 静默退回原行为，不影响稳定性。
- 新配置开关 `pts_enable`（默认 1），面板可关。

### 步骤E — 配置/面板适配 ✅
- `config.h`：新增 `pts_enable`（parse/dump/clamp 齐全，老配置缺字段默认 1）。
- `panel/index.html`：新增「帧时间戳对齐」开关，render/collect 同步。

### 步骤D（部分）— subgroup 探测 ✅
- `gles_subgroup_ok()` 运行时探测 `GL_KHR_shader_subgroup_arithmetic/shuffle`，
  视频 init 时打日志上报（fp16/subgroup 能力）。仅探测，未替换算法。

## 二、需在「有 glslangValidator + NDK」环境完成（本机无工具链）

### 步骤B' — Vulkan L2 多尺度（用户要求"真正多维度"）
- 现状：Vulkan 后端单尺度 1/4（me0/me1）。要加 L2 粗层需：
  1. 新写 `me0_l2.comp`（L2 粗搜索）+ `me1_l2.comp`（L1 细化，读 L2 粗场放大），
     逻辑参照 GLES 的 `CS_ME0_SRC/CS_ME1_SRC`（已参数化 uLv2，可直接移植）。
  2. `payload_vk.cpp` 加 dsL2 图像、mvA_L2 图像、新 compute 管线/descriptor/dispatch 顺序。
  3. 用 `src/shaders/gen_spv.sh` 重编 SPV → `*_spv.h`。
- 建议：默认 lv2=2（1/8）与 GLES 视频对齐；加配置开关，失败回退单尺度。

### 步骤D（完整）— L1 SAD subgroup/warp 归约
- GLES：`gles_subgroup_ok()` 已探测；实际加速版需把 8×8 block 的 16 个采样点分给 lane、
  `subgroupAdd` 归约，重写 `CS_ME1_SRC` 的内层循环。需真机验证 subgroupSize（Adreno 64/Mali 16~32）。
- Vulkan：用 `VK_KHR_shader_subgroup_shuffle`/arithmetic，改 `me1.comp` 后用 gen_spv.sh 重编。

## 三、明确不做
- ❌ 直接读引擎深度图/运动矢量（数据驱动 MV）：外挂 hook 层无此语义，需引擎 SDK 集成，另立项目。
- 🟡 硬件采样器 warp、FP16：现状已完成（`texture()` LINEAR / mediump / f16 shader + 探测）。

## 四、编译与验证
- **本机已用 NDK r27c 编译打包为 v2.5.0**（arm64-v8a.so + mcfid 实测通过）。
- 改 shader 后：`cd src/shaders && ./gen_spv.sh` 重编 SPV（本机无 glslangValidator，需有工具链的环境跑）。
- 真机验证：`logcat -s MCFI` 看
  - `eglPresentationTimeANDROID 时间戳注入: 可用`
  - `设备启用 VK_GOOGLE_display_timing`
  - `GLES 能力探测: fp16=.. subgroup=..`
  - 视频模式下快速运动（如舞蹈/横移）是否更稳、两帧停留是否均匀。

## 五、v2.5.0 版本说明
- version=v2.5.0 / versionCode=250（module.prop / build.sh ZIPNAME 已同步）。
- 已编译进包：GLES 视频 L2=1/8、视频补 Vulkan hook、SurfaceFlinger 时间戳注入（pts_enable 开关）、subgroup 探测。
- 待重编 SPV 后接入：Vulkan L2 多尺度、Vulkan subgroup me1（需 glslangValidator）。
