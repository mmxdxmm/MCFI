# MCFI 补帧模块 v2.0.0（运动补偿插帧 · Zygisk · arm64-v8a）

基于 Zygisk 的**运动补偿插帧（MCI）**模块，**同时支持 OpenGL ES 3.2+ 与 Vulkan 1.1+ 渲染的游戏与视频**。
在相邻两个真实帧之间用 GPU 做块匹配运动估计，按运动矢量合成中间帧并插入呈现，画面连贯流畅、拖影显著低于普通混合。

- **GLES 路径**：Hook `eglSwapBuffers`，异步 worker 线程（共享 EGL 上下文）完成 运动估计 + 合成；
  hook 线程只做拷贝与生成帧呈现，**不占用游戏渲染线程**。
- **Vulkan 路径**：Hook `vkGetInstanceProcAddr` 一处入口，跟踪交换链，拦截 `vkQueuePresentKHR`；
  compute 运动估计 + 图形管线合成在异步 worker 线程执行，拷贝后严格 barrier 回 `PRESENT_SRC_KHR`。
- **运动补偿算法**：1/4 分辨率降采样 → 8×8 块 SAD 全搜索（可细化）→ 双向 warp 合成，
  带置信度回退（场景切换/新内容自动退化为混合）与遮挡/拖影抑制。

## 一、安装

1. 确认 Magisk 已启用 **Zygisk**（Magisk App → 设置 → Zygisk 开关），设备为 arm64-v8a；
   KernelSU / APatch 环境请配合 **Zygisk Next** 使用（本模块只使用标准 Zygisk API 与 root companion，可兼容）。
2. 刷入 `MCFI-补帧模块-v2.0.0-arm64.zip`，重启。

## 二、控制面板

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

**面板为手动刷新**：配置与日志都只在点击「刷新」按钮时重新读取，不会自动轮询覆盖你正在编辑的配置。

配置文件：`/data/adb/modules/mcfi/config.conf`（key=value，升级模块自动保留）。

## 三、工作原理

```
游戏进程                            守护进程(root)
   │ eglSwapBuffers / vkQueuePresentKHR 被 hook     ▲
   │  ① 真实帧 N+1 拷贝/入队（GLES 环形槽 / Vulkan 队列）│
   │  ② 异步 worker：降采样→块匹配运动估计→双向 warp   │ HTTP :4399 控制面板
   │     合成中间帧，先行呈现生成帧                    │ unix socket 配置服务
   │  ③ 真实帧 N+1 照常呈现                          ▲
   ▼                                               │
zygisk companion（root 中继，规避 SELinux）───────────┘
```

- Zygisk 入口在 `preAppSpecialize` 阶段读取配置、匹配包名，命中则安装 hook。
- 配置热更新：载荷每 ~120 次呈现通过 companion 通道向守护进程拉取一次配置，面板保存后约 2 秒生效。
- 守护进程由 `service.sh` 从模块目录直接拉起，崩溃自动重启；日志 `daemon.log`、`app.log` 均在模块目录内。
- **模块页一眼可见**：守护进程把运行状态（面板端口、补帧开关、名单模式与条数、算法）实时写进
  `module.prop` 的 description。
- **接口日志**：命中应用按 `包名(pid): GLES 接口 eglSwapBuffers 首次命中 WxH` / `Vulkan 接口
  vkQueuePresentKHR 首次被调用` 上报到 `app.log`（环形保留最新 200 行）与面板「最近生效记录」，
  面板按 GLES（绿）/ Vulkan（蓝）着色区分。

## 四、性能与稳定性设计

1. **异步不抢渲染**：GLES 游戏模式 hook 线程只做「拷帧 + 必要时画生成帧并 swap」两件廉价操作，
   运动估计/合成全部在共享上下文的 worker 线程执行；Vulkan 的 GPU 操作全部在 worker 线程。
2. **防堆积**：GLES 用 4 个环形槽 + 栅栏同步 + 消费握手，worker 繁忙时直接丢弃当前帧（不阻塞、不积压）；
   Vulkan 队列只保留最新任务，丢弃积压旧任务。
3. **老驱动兼容**：Vulkan 拷贝后强制 barrier 回 `PRESENT_SRC_KHR`；资源创建/能力检查失败只降级为
   普通混合（或跳过插帧），绝不改动游戏自身呈现路径；`logcat -s MCFI` 可见原因。
4. **视频模式**：`=video` 后缀走 GLES 同步路径 + 按 EGLContext 资源池（上限 3 套，防内存爆满），
   过小 surface（弹幕/小窗/浮层，宽 < 500）直接透传，只对主画面插帧；部分视频 App 因多 Context
   或私有呈现路径可能不插帧，属设计取舍，不会导致卡死或花屏。

## 五、故障排查

1. **守护进程**：Magisk 模块页看描述行是否显示「运行中 · 面板 http://127.0.0.1:xxxx」；`ps -A | grep mcfid`。
2. **Zygisk 加载**：重启后 `logcat -s MCFI` 应出现「zygisk 入口已加载」；没有则检查 Magisk 的 Zygisk 开关。
3. **命中应用**：打开目标应用后 `logcat -s MCFI` 应出现「目标进程命中: 包名，后端=…」与接口命中日志。
4. **面板 404**：描述行里的端口才是实际端口；面板「守护进程日志」卡片可看 daemon.log 尾部。

## 六、卸载

Magisk 中移除模块并重启（`uninstall.sh` 会清理 `/data/local/tmp` 下的配置镜像）。

## 七、源码与重新编译

- `src/entry.cpp` — Zygisk 入口（companion 通道、包名匹配）
- `src/payload.cpp` — GLES 后端（异步 worker + 运动估计/合成，视频模式资源池）
- `src/payload_vk.cpp` — Vulkan 后端（入口 hook + 交换链跟踪 + compute 运动估计 + MCI 管线）
- `src/shaders/` — GLSL 源码与预编译 SPIR-V（`*.spv` + `*_spv.h`，由 glslangValidator 生成）
- `src/daemon.cpp` — root 守护进程（配置服务 + 面板 HTTP + module.prop 状态回写）
- `src/config.h` — 配置解析
- `panel/index.html` — 控制面板（手动刷新）
- `module/` — Magisk 模块脚本

用 Android NDK r25+ 执行 `./build.sh` 即可重新打包（第三方依赖 Dobby 静态链接，源码已含于 `third_party/`）。
如需修改 Vulkan 着色器，改 `src/shaders/*.frag/*.comp` 后用 glslangValidator 重新生成 `*_spv.h`。

## 八、已知限制（务必阅读）

1. **延迟**：补帧需要「看到未来一帧」，引入约 1 帧显示延迟；竞技类游戏请自行斟酌。
2. **算法**：端侧实时块匹配运动估计（无光流），快速规则运动效果最佳；极端遮挡/快速缩放场景
   可能仍有轻微伪影，可调高「拖影抑制强度」或改「普通混合」。
3. **兼容范围**：
   - GLES：要求 OpenGL ES 3.2+（compute 运动估计）；无 compute 时自动降级为普通混合。
   - Vulkan：要求交换链图像带 `VK_IMAGE_USAGE_TRANSFER_SRC_BIT`（绝大多数引擎默认开启）、
     图像数 ≥ 3 才插帧；交换链格式需支持 blit 与 `R8G8B8A8_UNORM` storage（几乎全部支持），
     否则降级为普通混合。
   - 双后端自动识别，同一参数配置对两者都生效。
4. **帧率翻倍的前提**：设备刷新率需高于游戏帧率（建议 120Hz 屏 + 解锁帧率，或选 ×1.5 档）。
5. 部分使用自定义帧缓冲/MSAA 特殊路径的引擎可能出现首帧拷贝异常，遇到花屏请移出白名单。
6. 与「排除列表/隐藏 Root」类模块共存一般无碍，但请在补帧目标应用中关闭严格隐藏检测。
7. Vulkan 插帧会自行 acquire/呈现一张额外交换链图像，个别老驱动（如骁龙 855 早期 Adreno）上如遇
   闪烁或帧序异常，请将该应用切回 GLES 渲染（`=gles`）或移出白名单。
