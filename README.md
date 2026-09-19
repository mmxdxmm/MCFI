# MCFI 补帧模块 v2.5.4（运动补偿插帧 · Zygisk · arm64-v8a）

基于 Zygisk 的**运动补偿插帧（MCI）**模块，**同时支持 OpenGL ES 3.2+ 与 Vulkan 1.1+ 渲染的游戏与视频**。
在相邻两个真实帧之间用 GPU 做块匹配运动估计，按运动矢量合成中间帧并插入呈现，画面连贯流畅、拖影显著低于普通混合。

- **GLES 路径**：Hook `eglSwapBuffers`，异步 worker 线程（共享 EGL 上下文）完成运动估计 + 合成；
  hook 线程只做拷贝与生成帧呈现，**不占用游戏渲染线程**。
- **Vulkan 路径**：Hook `vkGetInstanceProcAddr` 一处入口，跟踪交换链，拦截 `vkQueuePresentKHR`；
  compute 运动估计 + 图形管线合成在异步 worker 线程执行，拷贝后严格 barrier 回 `PRESENT_SRC_KHR`。

## 一、运动估计算法（v2.5.x）

两后端统一采用**多尺度块匹配 + 子像素细化 + 置信度回退**：

```
L2 粗层（1/16 游戏 / 1/8 视频）  →  9 候选 3DRS 搜索，得粗运动场
        ↓ 放大 ×L
L1 细层（1/4）                  →  以粗场为中心 ±1 + 时间预测 + GMV 候选 + 半像素细化
        ↓
中值滤波 + 低置信统计            →  隔离错误矢量 / 整帧降级判定
```

性能优化（按设备能力自动启用，不支持自动回退）：

| 优化 | GLES | Vulkan | 说明 |
|---|---|---|---|
| L2 多尺度 | ✅ `uLv2` 参数化（游戏4/视频2） | ✅ 双缓冲 `mv_img_l2[2]` | 粗层先定大位移，细层局部求精 |
| subgroup 归约 | ✅ `CS_ME1_SUB_SRC`（16 lane） | ✅ `me1_sub.comp` | 16 采样点 `subgroupAdd` 一次归约，省循环开销 |
| FP16 | ✅ `mediump` | ✅ `*_f16.comp`（`VK_KHR_shader_float16_int8`） | 不支持自动回退 FP32 |
| 硬件采样器 | ✅ `texture()` LINEAR | ✅ blit + `sampler` LINEAR | 插值交给 TMU，不手写双线性 |
| 帧时间戳注入 | ✅ `eglPresentationTimeANDROID` | ✅ `VK_GOOGLE_display_timing` | 让 A / A.5 / B 各占一个 vsync，避免帧停留不均 |

subgroup 能力判定：Vulkan 查 core `VkPhysicalDeviceSubgroupProperties.supportedOperations & ARITHMETIC_BIT`（不依赖 KHR 扩展名，Adreno 也能识别）；GLES 查 `GL_KHR_shader_subgroup_arithmetic`。

## 二、安装

1. 确认 Magisk 已启用 **Zygisk**（Magisk App → 设置 → Zygisk 开关），设备为 arm64-v8a；
   KernelSU / APatch 环境请配合 **Zygisk Next** 使用（本模块只使用标准 Zygisk API 与 root companion，可兼容）。
2. 刷入 `MCFI-补帧模块-v2.5.4-arm64.zip`，重启。

## 三、控制面板

手机浏览器直接打开：**http://127.0.0.1:4399**（端口被占用自动顺延，实际端口看 Magisk 模块页描述行）。

（也可用电脑：`adb forward tcp:4399 tcp:4399` 后浏览器打开同一地址）

| 参数 | 说明 |
|---|---|
| 启用补帧 | 总开关 |
| 目标模式 | 白名单 / 黑名单 / 全部应用（系统关键进程永不注入） |
| 包名列表 | 逗号分隔，可带 `=gles` / `=vulkan` / `=video` / `=off` 后缀 |
| 插帧算法 | MCI 运动补偿（默认，推荐）/ 普通混合（兼容）/ 运动自适应混合 |
| 运动估计质量 | 0~100：搜索半径与细化级数，越高越清晰、开销越高 |
| 拖影抑制强度 | 0~100：遮挡区域保守程度 |
| 插帧间隔 | 每 N 个真实帧插入 1 个生成帧；N=1 即帧率翻倍 |
| 帧时间戳对齐 | 开（默认）：让生成帧与真实帧各占一个 vsync；关：退回原 present 节奏 |

**面板为手动刷新**：配置与日志都只在点击「刷新」按钮时重新读取，不会自动轮询覆盖你正在编辑的配置。

配置文件：`/data/adb/modules/mcfi/config.conf`（key=value，升级模块自动保留）。

## 四、工作原理

```
游戏进程                            守护进程(root)
   │ eglSwapBuffers / vkQueuePresentKHR 被 hook     ▲
   │  ① 真实帧 N+1 拷贝/入队（GLES 环形槽 / Vulkan 队列）│
   │  ② 异步 worker：L2粗搜索→L1细化→中值滤波→双向warp │ HTTP :4399 控制面板
   │     合成中间帧，先行呈现生成帧（带时间戳注入）     │ unix socket 配置服务
   │  ③ 真实帧 N+1 照常呈现（+2 vsync）              ▲
   ▼                                               │
zygisk companion（root 中继，规避 SELinux）───────────┘
```

- Zygisk 入口在 `preAppSpecialize` 阶段读取配置、匹配包名，命中则安装 hook。
- 配置热更新：载荷每 ~120 次呈现通过 companion 通道向守护进程拉取一次配置，面板保存后约 2 秒生效。
- 守护进程由 `service.sh` 从模块目录直接拉起，崩溃自动重启；版本号编译时从 `module.prop` 注入，
  启动日志为 `======== MCFI 守护进程启动 vX.Y.Z ========`。
- **模块页一眼可见**：守护进程把运行状态（面板端口、补帧开关、名单模式与条数、算法）实时写进
  `module.prop` 的 description。

## 五、查看日志

**主要用 logcat**（实时、最直接）：

```bash
logcat -s MCFI MCFID
```

关键日志行：

| 日志 | 含义 |
|---|---|
| `MCFI 守护进程启动 vX.Y.Z` | 守护进程版本（应与模块版本一致） |
| `subgroup: arithmetic=1 subgroupSize=64` | Vulkan subgroup 能力探测结果 |
| `GLES 能力探测: fp16=是 subgroup=支持(已接入ME1归约)` | GLES 能力 |
| `GLES ME1 已接入 subgroup 归约版` | GLES subgroup 实际生效 |
| `L2 pipeline 就绪: subgroup`（或 fp16/fp32） | Vulkan 选用的运动估计管线 |
| `eglPresentationTimeANDROID 时间戳注入: 可用` | GLES 时间戳注入能力 |
| `设备启用 VK_GOOGLE_display_timing` | Vulkan 时间戳注入能力 |

守护进程自身日志落盘在模块目录 `daemon.log`（面板「守护进程日志」卡片可读尾部）；
各应用命中事件经 companion 上报后由守护进程环形保留在 `app.log`（最新 200 行），
面板「最近生效记录」按 GLES（绿）/ Vulkan（蓝）着色区分。

## 六、性能与稳定性设计

1. **异步不抢渲染**：GLES hook 线程只做「拷帧 + 必要时画生成帧并 swap」两件廉价操作，运动估计/合成全部在
   共享上下文的 worker 线程；Vulkan 的 GPU 操作全部在 worker 线程。
2. **防堆积**：GLES 用 4 个环形槽 + 栅栏同步 + 消费握手，worker 繁忙时直接丢弃当前帧；
   Vulkan 队列只保留最新任务，丢弃积压旧任务。
3. **能力回退**：L2 / subgroup / fp16 / 时间戳注入任一环节探测或创建失败，自动降级到可用子集，
   绝不改动游戏自身呈现路径；`logcat -s MCFI` 可见原因。
4. **视频模式**：`=video` 后缀走 GLES 同步路径 + 按 EGLContext 资源池（上限 3 套，防内存爆满），
   过小 surface（弹幕/小窗/浮层，宽 < 500）直接透传，只对主画面插帧。Vulkan 视频也会 hook
   （`want_vk` 含 backend 0/2/3）。

## 七、故障排查

1. **守护进程**：Magisk 模块页看描述行是否显示「运行中 · 面板 http://127.0.0.1:xxxx」；`ps -A | grep mcfid`。
2. **Zygisk 加载**：重启后 `logcat -s MCFI` 应出现「zygisk 入口已加载」；没有则检查 Magisk 的 Zygisk 开关。
3. **版本不对**：守护进程启动日志若仍显示旧版本号，说明模块未更新到位；重新刷入并重启。
4. **subgroup 没生效**：看 `subgroup: arithmetic=` 是否为 1、`GLES ME1 已接入 subgroup 归约版` 是否出现；
   为 0 说明该设备驱动未暴露对应能力，已自动走普通版，不影响功能。
5. **面板 404**：描述行里的端口才是实际端口。

## 八、卸载

Magisk 中移除模块并重启（`uninstall.sh` 会清理 `/data/local/tmp` 下的配置镜像）。

## 九、源码与重新编译

- `src/entry.cpp` — Zygisk 入口（companion 通道、包名匹配）
- `src/payload.cpp` — GLES 后端（异步 worker + 运动估计/合成，视频模式资源池；内嵌 `CS_ME1_SUB_SRC` subgroup 版）
- `src/payload_vk.cpp` — Vulkan 后端（入口 hook + 交换链跟踪 + L2/subgroup/fp16 compute 管线 + MCI 合成）
- `src/shaders/` — GLSL 源码与预编译 SPIR-V 头文件（`*_spv.h`，由 glslangValidator 14.3 生成）
- `src/daemon.cpp` — root 守护进程（配置服务 + 面板 HTTP + module.prop 状态回写；版本号编译注入）
- `src/config.h` — 配置解析
- `panel/index.html` — 控制面板（手动刷新）
- `module/` — Magisk 模块脚本

构建：
```bash
NDK=/home/user/android-ndk-r27c ./build.sh
```
产物：`/tmp/mcfi-out/MCFI-补帧模块-vX.Y.Z-arm64.zip`。

修改 Vulkan shader 后重编 SPV：
```bash
cd src/shaders && PATH=/home/user/glslang:$PATH bash gen_spv.sh
```

## 十、已知限制（务必阅读）

1. **延迟**：补帧需要「看到未来一帧」，引入约 1 帧显示延迟；竞技类游戏请自行斟酌。
2. **算法**：端侧实时多尺度块匹配运动估计（无光流、不读引擎内部 G-Buffer），快速规则运动效果最佳；
   极端遮挡/快速缩放场景可能仍有轻微伪影，可调高「拖影抑制强度」或改「普通混合」。
3. **兼容范围**：
   - GLES：要求 OpenGL ES 3.2+（compute 运动估计）；无 compute 时自动降级为普通混合。
   - Vulkan：要求 Vulkan 1.1+、交换链图像带 `TRANSFER_SRC_BIT`、图像数 ≥ 3；
     subgroup / fp16 / 时间戳注入按设备能力自动启停。
4. **帧率翻倍的前提**：设备刷新率需高于游戏帧率（建议 120Hz 屏 + 解锁帧率，或选 ×1.5 档）。
5. 部分使用自定义帧缓冲/MSAA 特殊路径的引擎可能出现首帧拷贝异常，遇到花屏请移出白名单。
6. 与「排除列表/隐藏 Root」类模块共存一般无碍，但请在补帧目标应用中关闭严格隐藏检测。
