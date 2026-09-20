// MCFI 载荷核心：Hook eglSwapBuffers，运动补偿插帧（MCI）并插入呈现
//
// 算法：
//   1. 把相邻真实帧降采样到 1/4 分辨率（纹理采样线性缩放，兼顾预滤波）
//   2. compute 着色器做块匹配运动估计（SAD 全搜索 + 可选细化），输出运动矢量场（RGBA8 编码）
//   3. fragment 着色器按运动矢量双向 warp 合成中间帧，带置信度回退与遮挡/拖影抑制
//
// 执行模型：
//   - 游戏模式（默认）：异步。hook 线程只做两件廉价事（把新帧拷贝进环形槽、必要时把生成帧
//     画到默认帧缓冲并先行 swap），运动估计/合成全部在共享上下文的 worker 线程执行，
//     不占用游戏渲染线程；环形槽 4 个 + 栅栏同步 + 消费握手，天然防堆积。
//   - 视频模式（=video）：同步 + 按 EGLContext 资源池（延续原版防内存爆满思路），
//     小 surface 直接透传不做插帧。

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <GLES3/gl31.h>   // GLES 3.1+ compute（运动估计）
#include <android/log.h>
#include <dlfcn.h>
#include <dobby.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <time.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <queue>
#include <string>
#include <vector>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <map>
#include <sys/socket.h>
#include <sys/un.h>

#include "config.h"

#define LOG_TAG "MCFI"
#define LOGI(...) do { if (g_cfg.log_level > 0) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__); } while (0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static McfiConfig g_cfg;
static std::mutex g_cfg_mtx;
static int g_companion_fd = -1;
static std::string g_pkg;   // 命中包名（由 entry 传入，用于事件上报）
static int g_backend = 0;   // McfiConfig::Backend，由 entry 传入
static std::mutex g_gl_mtx; // 串行化 hook 内 GL 操作（app 可能多线程 swap）

// 供 worker / hook 使用的跨文件接口
extern "C" void mcfi_get_config(McfiConfig *out);
extern "C" void mcfi_refresh_config();
extern "C" void mcfi_vk_install(int log_level, int backend);

using eglSwapBuffers_t = EGLBoolean (*)(EGLDisplay, EGLSurface);
static eglSwapBuffers_t orig_eglSwapBuffers = nullptr;

// ---------------- SurfaceFlinger 时间戳注入（EGL_ANDROID_presentation_time） ----------------
// 目的：连续两次 swap 时，让生成帧 A.5 上屏于 T+1 vsync、真实帧 B 上屏于 T+2 vsync，
// 两帧各占一个完整 vsync 周期，避免连发两次 swap 导致前帧被丢弃/停留时间不均。
// 扩展缺失或函数指针加载失败 → 静默退回原行为（不设时间戳）。
typedef EGLBoolean (*PFNEGLPRESENTATIONTIMEANDROIDPROC)(EGLDisplay, EGLSurface, int64_t);
static PFNEGLPRESENTATIONTIMEANDROIDPROC g_eglPts = nullptr;
static std::atomic<int64_t> g_vsync_period_ns{16666667};   // 兜底 60Hz=16.67ms；配置 vsync_us>0 时用配置值，否则自适应
static int64_t g_last_real_swap_ns = 0;
static bool g_pts_pending = false;   // 本 hook 调用是否已插入生成帧（需给真实帧设 +2 vsync）

static int64_t mcfi_now_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
// 用相邻真实帧 swap 间隔平滑估计 vsync 周期（合理区间 3~20ms）；配置 vsync_hz>0 时用配置值
static void mcfi_vsync_sample(int64_t now_ns) {
    if (g_cfg.vsync_hz > 0) { g_vsync_period_ns.store(1000000000LL / g_cfg.vsync_hz); return; }
    if (g_last_real_swap_ns > 0) {
        int64_t d = now_ns - g_last_real_swap_ns;
        if (d >= 3000000 && d <= 20000000) {
            int64_t cur = g_vsync_period_ns.load();
            g_vsync_period_ns.store((cur * 7 + d) / 8);
        }
    }
    g_last_real_swap_ns = now_ns;
}
// 把 surface 的期望上屏时间设为对齐到 vsync 边界的时间
// k=1: 下一个 vsync 上屏（生成帧）
// k=2: 下下个 vsync 上屏（真实帧）
static void mcfi_set_pts(EGLDisplay dpy, EGLSurface surf, int k) {
    if (!g_eglPts || !g_cfg.gles_pts_enable) return;
    int64_t period = g_vsync_period_ns.load();
    int64_t now = mcfi_now_ns();
    // 对齐到下一个 vsync 边界（向上取整）
    int64_t next_vsync = ((now + period - 1) / period) * period;
    g_eglPts(dpy, surf, next_vsync + (k - 1) * period);
}
// 真实帧上屏：若本帧插入了生成帧，则把真实帧期望上屏时间设为 +2 vsync，并采样周期
static inline EGLBoolean mcfi_swap_real(EGLDisplay dpy, EGLSurface surf) {
    if (g_pts_pending) {
        mcfi_set_pts(dpy, surf, 2);
        mcfi_vsync_sample(mcfi_now_ns());
        g_pts_pending = false;
    }
    return orig_eglSwapBuffers(dpy, surf);
}

// ---------------- 事件日志 ----------------
// 打 logcat，同时通过 companion 通道发给 daemon 落盘到 app.log（环形保留最新若干行）
static void mcfi_send_raw(const std::string &msg) {
    LOGI("%s", msg.c_str());
    if (g_companion_fd < 0) return;
    std::string out = "EVT " + msg + "\n";
    struct pollfd p{g_companion_fd, POLLOUT, 0};
    if (poll(&p, 1, 5) <= 0) return;
    ssize_t w = write(g_companion_fd, out.data(), out.size());
    if (w <= 0) return;
    char ack[64];
    p.fd = g_companion_fd; p.events = POLLIN; p.revents = 0;
    if (poll(&p, 1, 1) > 0) read(g_companion_fd, ack, sizeof(ack));
}

static void mcfi_send_event(const char *fmt, ...) {
    char body[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    mcfi_send_raw(body);
}

extern "C" void mcfi_set_pkg(const char *pkg);
extern "C" void mcfi_set_backend(int be) { g_backend = be; }
// 供 Vulkan 后端上报已格式化好的事件字符串
extern "C" void mcfi_send_event(const char *msg) {
    if (!msg || !*msg) return;
    mcfi_send_raw(msg);
}

static bool is_video_mode() {
    return g_backend == McfiConfig::B_VIDEO;
}

// ---------------- 配置热更新（通过 companion 通道向守护进程请求） ----------------

static std::string mcfi_request_config(int fd) {
    if (fd < 0) return "";
    struct pollfd p{fd, POLLOUT, 0};
    if (poll(&p, 1, 10) <= 0) return "";
    if (write(fd, "CFG\n", 4) != 4) return "";
    std::string acc;
    char buf[4096];
    for (int i = 0; i < 16; i++) {
        p.fd = fd; p.events = POLLIN; p.revents = 0;
        if (poll(&p, 1, 20) <= 0) break;
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        acc.append(buf, (size_t)n);
        if (acc.find("@EOF@") != std::string::npos) {
            size_t e = acc.find("@EOF@");
            return acc.substr(0, e);
        }
    }
    return "";
}

// ---------------- GLES 着色器 ----------------

static const char *VS_SRC = R"(#version 300 es
const vec2 pos[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
out vec2 vUV;
void main() {
    vec2 p = pos[gl_VertexID];
    gl_Position = vec4(p, 0.0, 1.0);
    vUV = p * 0.5 + 0.5;
}
)";

static const char *FS_COPY_SRC = R"(#version 300 es
precision mediump float;
uniform sampler2D uTex;
in vec2 vUV;
out vec4 fragColor;
void main() { fragColor = texture(uTex, vUV); }
)";

// 运动补偿插帧：
//   uMode 0=MCI（推荐） 1=普通混合(兼容) 2=运动自适应混合
//   uMV 为 RGBA8 运动矢量场：r/g 编码矢量(-range..range -> 0..1)，b=置信度
static const char *FS_INTERP_SRC = R"(#version 300 es
precision highp float;
uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform sampler2D uMV;
uniform int uMode;
uniform float uStrength;
uniform float uRange;
uniform float uSmooth;     // 平滑 0~100：静止/近静止保护阈值
uniform vec2 uDsTexel;
in vec2 vUV;
out vec4 fragColor;
void main() {
    vec4 c = texture(uCur, vUV);
    vec4 p = texture(uPrev, vUV);
    vec4 mid = mix(p, c, 0.5);
    if (uMode == 1) { fragColor = mid; return; }
    if (uMode == 2) {
        float d = abs(dot(c.rgb - p.rgb, vec3(0.299, 0.587, 0.114)));
        float w = clamp(d * uStrength * 4.0, 0.0, 1.0);
        fragColor = mix(mid, c, w);
        return;
    }
    // MCI：按运动矢量双向 warp
    vec4 mv = texture(uMV, vUV);                    // 双线性采样矢量场
    vec2 v = (mv.xy * 2.0 - 1.0) * uRange * uDsTexel;
    float conf = mv.z;
    // 静止/近静止保护：两帧亮度差低于阈值 → 直接取当前帧（零 warp 零混合）。
    // 内容几乎相同时取哪帧视觉无差，彻底消除低纹理/静止区域的矢量抖动与拖影。
    float ld = abs(dot(c.rgb - p.rgb, vec3(0.299, 0.587, 0.114)));
    if (ld < uSmooth * 0.0004) { fragColor = c; return; }
    vec4 cp = texture(uPrev, vUV - v * 0.5);
    vec4 cc = texture(uCur,  vUV + v * 0.5);
    vec4 mci = mix(cp, cc, 0.5);
    // 统一中间帧合成（SVP 式）：高置信 = 运动补偿双向 warp（t=0.5 位置采样）；
    // 低置信 = 原始两帧 50% 混合（同样是 t=0.5 时间点）——两者在运动正确的区域是
    // 同一画面，用置信度连续渐变过渡：无跳帧、无区域硬边界 → 无边缘撕裂。
    float fall = clamp((1.0 - conf) * 3.0, 0.0, 1.0);
    vec4 blend = mix(p, c, 0.5);
    vec4 outC = mix(blend, mci, 1.0 - fall);
    // 遮挡/拖影抑制：warp 后两侧采样亮度差异大（一侧被遮挡）→ 向当前帧轻微渐变（软回退，不硬跳）
    float lp = dot(cp.rgb, vec3(0.299, 0.587, 0.114));
    float lc = dot(cc.rgb, vec3(0.299, 0.587, 0.114));
    float g = clamp(abs(lp - lc) * uStrength * 8.0, 0.0, 1.0);
    outC = mix(outC, c, g * 0.35);
    fragColor = outC;
}
)";

// 运动估计 compute 着色器（GLES 3.1+）——3D 递归搜索（3DRS）
// 注意：部分驱动（如 Adreno 部分型号）只支持 image 的 writeonly/readonly，不支持 read/write 混合访问。
// 因此拆成两个 pass、两个独立 program：
//   pass0（CS_ME0_SRC）：3DRS 粗搜索（零/时间/空间/更新 ~9 候选），结果写 mvA（writeonly image）
//   pass1（CS_ME1_SRC）：以 mvA（sampler2D）为中心 ±1 细化 + 半像素子像素细化，结果写 mvB
// 时间预测（上一帧同块矢量）保证运动场跨帧连续 → 中间帧稳定不闪；候选少 → GPU 占用低。
// 输出纹理由 run_motion_estimation 返回（refine 开启返回 mvB，否则 mvA）。
// 运动估计 compute 着色器（GLES 3.1+）——多尺度 3D 递归搜索（金字塔 3DRS）+ 半精度（fp16/mediump）
// 说明：部分驱动（Adreno 部分型号）只支持 image 的 writeonly/readonly，不支持混合读写，
// 因此全部 pass 用 writeonly image + sampler 读；SAD 累加用 mediump（fp16 硬件上即半精度，
// 快速运动大范围搜索的 ALU 开销减半；不支持 fp16 的驱动自动按 fp32 处理，无副作用）。
// 多尺度：
//   pass0（CS_ME0_SRC，L2=1/16 层）：候选=零/时间/空间/更新（9 个），半径 R2=max(1,range/4)
//     → 覆盖 ±range 全分辨率像素的快速运动，写 mvA（L2 场）
//   pass1（CS_ME1_SRC，L1=1/4 层）：以 mvA 放大 4 倍为中心 ±1 + 时间预测 + 半像素细化，
//     写 mvB（L1 场）。运动矢量统一编码为 L1 基准（±range L1 像素），合成 frag 直接可用。
static const char *CS_ME0_SRC = R"(#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0) uniform highp sampler2D uPrev;    // dsL2 prev
layout(binding = 1) uniform highp sampler2D uCur;     // dsL2 cur
layout(binding = 2) uniform highp sampler2D uMvPrev;  // 上一帧 L1 场（mvC，尺寸 mvW1×mvH1）
layout(binding = 3, rgba32f) uniform writeonly highp image2D uMV;  // L2 场（mvA）
uniform highp ivec4 uP0;   // (R2, block, dsW2, dsH2)
uniform highp ivec4 uP1;   // (R1, mvW1, mvH1, mvW2)
uniform highp int uLv2;    // L2 相对 L1 的下采样倍率：4=1/16(游戏) 2=1/8(视频)
void main() {
    ivec2 g = ivec2(gl_GlobalInvocationID.xy);
    int L = uLv2;
    int dsW = uP0.z, dsH = uP0.w, block = uP0.y, R = uP0.x;
    int mvW1 = uP1.y, mvH1 = uP1.z;
    int gw = (dsW + block - 1) / block;
    int gh = (dsH + block - 1) / block;
    if (g.x >= gw || g.y >= gh) return;
    vec2 dsSize = vec2(float(dsW), float(dsH));
    vec2 base = (vec2(g * block) + vec2(0.5 * float(block))) / dsSize;
    vec2 mvSize1 = vec2(float(mvW1), float(mvH1));
    // L2 块 g 对应 L1 场位置 g*L（1/L 层 = 1/4 层再缩 L 倍）：时间/空间预测直接采样 L1 场
    vec2 cT = (vec2(g * L) + vec2(0.5 * float(L))) / mvSize1;
    vec2 tm = texture(uMvPrev, cT).xy;
    vec2 spL = texture(uMvPrev, (vec2(max(g.x * L - L, 0), g.y * L + L / 2) + vec2(0.5)) / mvSize1).xy;
    vec2 spU = texture(uMvPrev, (vec2(g.x * L + L / 2, max(g.y * L - L, 0)) + vec2(0.5)) / mvSize1).xy;
    vec2 spR = texture(uMvPrev, (vec2(min(g.x * L + L, mvW1 - 1), max(g.y * L - L, 0)) + vec2(0.5)) / mvSize1).xy;
    float jit = 1.0 / (2.0 * float(R));   // 更新预测步长：±1 L2 像素
    vec2 cand[9];
    cand[0] = vec2(0.5);
    cand[1] = tm;
    cand[2] = spL; cand[3] = spU; cand[4] = spR;
    cand[5] = tm + vec2( jit, 0.0);
    cand[6] = tm + vec2(-jit, 0.0);
    cand[7] = tm + vec2(0.0,  jit);
    cand[8] = tm + vec2(0.0, -jit);
    float bestSad = 1e30;
    float zeroSad = 1e30;
    vec2 bestOff = cand[0];
    for (int k = 0; k < 9; k++) {
        vec2 cp = clamp(cand[k], vec2(0.0), vec2(1.0));
        vec2 off = (cp * 2.0 - 1.0) * float(R) / dsSize;
        mediump float sadA = 0.0, sadB = 0.0;
        for (int by = 0; by < block; by += 2) {
            for (int bx = 0; bx < block; bx += 2) {
                vec2 pp = base + (vec2(float(bx), float(by)) + 0.5 - 0.5 * float(block)) / dsSize;
                mediump vec3 c = texture(uCur, pp).rgb;
                mediump vec3 q = texture(uPrev, pp - off).rgb;
                mediump float d = abs(c.r - q.r) + abs(c.g - q.g) + abs(c.b - q.b);
                if ((bx & 2) != 0) sadB += d; else sadA += d;
            }
        }
        float sad = float(sadA) + float(sadB);
        zeroSad = (k == 0) ? sad : zeroSad;
        // 无分支选择：min 更新最优 SAD，mix 按 step 选择最优偏移（编译器→select 指令）
        float better = step(sad, bestSad);
        bestSad = min(bestSad, sad);
        bestOff = mix(bestOff, cp, better);
    }
    float vx = 0.5, vy = 0.5, conf = 0.25;
    // 可信度门限无分支化：ok=1 才采用（bestSad < zeroSad*0.70）
    float ok = 1.0 - step(zeroSad * 0.70, bestSad);
    vx = mix(0.5, bestOff.x, ok);
    vy = mix(0.5, bestOff.y, ok);
    conf = mix(0.25, clamp(1.0 - bestSad / (float(block * block) * 0.375), 0.0, 1.0), ok);
    imageStore(uMV, g, vec4(vx, vy, conf, 1.0));
}
)";

// 多尺度 pass1（CS_ME1_SRC，L1=1/4 层）：粗场放大为中心 ±1 整像素 + 时间预测 + 半像素细化。
// 运动矢量统一编码 L1 基准（±R1 L1 像素）：vx=(off+R1)/(2R1)，合成 frag 的 uRange=R1、uDsTexel=1/dsW 直接成立。
// GMV（全局运动向量）统计 pass0：L2 场 mvA 直方图（轻量，原子累加）。
// 逻辑：远景运动≈相机运动；3DRS 在平坦区找不到可靠 MV，统计全画面主导运动向量强塞给低置信块。
static const char *CS_GMV0_SRC = R"(#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0) uniform highp sampler2D uMV;     // L2 场（mvA）
layout(std430, binding = 1) buffer Hist { int bins[1024]; int total; } hist;
uniform highp ivec4 uP0;   // (R2, mvW2, mvH2, binW)
void main() {
    ivec2 g = ivec2(gl_GlobalInvocationID.xy);
    if (g.x >= uP0.y || g.y >= uP0.z) return;
    vec2 mv = texture(uMV, (vec2(g) + 0.5) / vec2(float(uP0.y), float(uP0.z))).xy;
    vec2 disp = (mv * 2.0 - 1.0) * float(uP0.x);   // L2 像素位移（±R2）
    int binW = uP0.w;
    int qx = int(round(disp.x)) + uP0.x + 1;
    int qy = int(round(disp.y)) + uP0.x + 1;
    qx = clamp(qx, 0, binW - 1);
    qy = clamp(qy, 0, binW - 1);
    atomicAdd(hist.bins[qy * binW + qx], 1);
    atomicAdd(hist.total, 1);
}
)";

// GMV 统计 pass1：直方图归约找峰值（单 workgroup），写 1×1 GMV 纹理（x,y=归一化矢量 z=峰值占比 w=0）
static const char *CS_GMV1_SRC = R"(#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) buffer Hist { int bins[1024]; int total; } hist;
layout(binding = 1, rgba32f) uniform writeonly highp image2D uOut;  // 1×1 GMV
uniform highp ivec4 uP0;   // (binW, R2, 0, 0)
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    int binW = uP0.x, R2 = uP0.y;
    int best = 0; float bestC = -1.0;
    int n = binW * binW;
    int tot = hist.total > 0 ? hist.total : 1;
    for (int i = 0; i < n; i++) {
        float c = float(hist.bins[i]);
        float oldC = bestC;                     // 无分支峰值归约（max + 三元 select）
        bestC = max(bestC, c);
        best = (c > oldC) ? i : best;
    }
    int qx = best % binW, qy = best / binW;
    vec2 disp = vec2(float(qx - R2 - 1), float(qy - R2 - 1));
    vec2 gmv = disp / float(R2) * 0.5 + 0.5;   // 归一化 0~1（与运动场同编码）
    float w = bestC / float(tot);
    w *= step(0.20, w);                          // 峰值占比过低 → 无 GMV（无分支）
    imageStore(uOut, ivec2(0, 0), vec4(gmv, w, 0.0));
}
)";

static const char *CS_ME1_SRC = R"(#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0) uniform highp sampler2D uPrev;      // dsL1 prev (1/4)
layout(binding = 1) uniform highp sampler2D uCur;       // dsL1 cur
layout(binding = 2) uniform highp sampler2D uMVPrev;    // L2 粗场（mvA，尺寸 mvW2×mvH2）
layout(binding = 3) uniform highp sampler2D uMVPrevL1;  // 上一帧 L1 场（mvC，时间预测）
layout(binding = 4, rgba32f) uniform writeonly highp image2D uMV;  // L1 场（mvB）
layout(binding = 5) uniform highp sampler2D uGMV;  // 1×1 GMV 纹理（x,y=归一化 z=峰值占比）
uniform highp ivec4 uP0;   // (R1, block, dsW1, dsH1)
uniform highp ivec4 uP1;   // (R2, mvW1, mvH1, mvW2)
uniform highp int uRefine; // 1=细化和半像素 0=仅转写粗场中心（快速档）
uniform highp float uSmoothW; // 平滑 0~1：低置信向时间预测收缩（时域稳定）
uniform highp int uLv2;    // L2 相对 L1 的下采样倍率：4=1/16(游戏) 2=1/8(视频)
void main() {
    ivec2 g = ivec2(gl_GlobalInvocationID.xy);
    int L = uLv2;
    int dsW = uP0.z, dsH = uP0.w, block = uP0.y, R = uP0.x;
    int R2 = uP1.x, mvW1 = uP1.y, mvH1 = uP1.z;
    // L2 场尺寸由 L1 ds 尺寸按 L 推导（uP1 只有 4 个 int，装不下 mvH2）
    int dsW2 = (dsW + L - 1) / L;
    int dsH2 = (dsH + L - 1) / L;
    int mvW2 = (dsW2 + 7) / 8;
    int mvH2 = (dsH2 + 7) / 8;
    int gw = (dsW + block - 1) / block;
    int gh = (dsH + block - 1) / block;
    if (g.x >= gw || g.y >= gh) return;
    vec2 dsSize = vec2(float(dsW), float(dsH));
    vec2 base = (vec2(g * block) + vec2(0.5 * float(block))) / dsSize;
    float zeroSad = 0.0;
    for (int by = 0; by < block; by += 2) {
        for (int bx = 0; bx < block; bx += 2) {
            vec2 pp = base + (vec2(float(bx), float(by)) + 0.5 - 0.5 * float(block)) / dsSize;
            mediump vec3 c = texture(uCur, pp).rgb;
            mediump vec3 q = texture(uPrev, pp).rgb;
            zeroSad += float(abs(c.r - q.r)) + float(abs(c.g - q.g)) + float(abs(c.b - q.b));
        }
    }
    // L2 粗场（归一化）→ L1 像素中心：×L 放大（L=4 游戏/L=2 视频）
    vec2 c0 = texture(uMVPrev, (vec2(g) + 0.5) / vec2(float(mvW2), float(mvH2))).xy;
    vec2 center = (c0 * 2.0 - 1.0) * float(R2) * float(L);
    // 时间预测（上一帧 L1 场同块）
    vec2 tm = texture(uMVPrevL1, (vec2(g) + 0.5) / vec2(float(mvW1), float(mvH1))).xy;
    vec2 tmPix = (tm * 2.0 - 1.0) * float(R);
    float bestSad = 1e30;
    vec2 bestOff = center;
    vec2 cand[15];
    int n = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            cand[n++] = center + vec2(float(dx), float(dy));
        }
    }
    cand[n++] = tmPix;
    cand[n++] = tmPix + vec2( 1.0, 0.0);
    cand[n++] = tmPix + vec2(-1.0, 0.0);
    cand[n++] = tmPix + vec2(0.0,  1.0);
    cand[n++] = tmPix + vec2(0.0, -1.0);
    // GMV 候选：画面存在主导运动（相机运动）时强塞给平坦/低置信块。
    // 记录 SAD_GMV（GMV 候选的匹配误差）作为深度判定依据：GMV 能完美解释的块=远景（区域 A）。
    vec4 gmvData = texture(uGMV, vec2(0.5));
    int gmvIdx = -1;
    if (gmvData.z > 0.05) { gmvIdx = n; cand[n++] = (gmvData.xy * 2.0 - 1.0) * float(R); }
    int candN = (uRefine != 0) ? n : 1;   // 快速档只测粗场中心（转写放大）
    float sadGmv = 1e30;
    for (int k = 0; k < candN; k++) {
        vec2 cp = clamp(cand[k], vec2(-float(R)), vec2(float(R)));
        vec2 off = cp / dsSize;
        // 2 路 fp16 子累加 + fp32 合并：单累加器到 ~6000 时 ULP≈4，分路后相对误差 <7e-4，
        // 避免大数累加低位截断 → SAD 比较噪声（运动估计乱选 → 拖影/模糊）
        mediump float sadA = 0.0, sadB = 0.0;
        for (int by = 0; by < block; by += 2) {
            for (int bx = 0; bx < block; bx += 2) {
                vec2 pp = base + (vec2(float(bx), float(by)) + 0.5 - 0.5 * float(block)) / dsSize;
                mediump vec3 c = texture(uCur, pp).rgb;
                mediump vec3 q = texture(uPrev, pp - off).rgb;
                mediump float d = abs(c.r - q.r) + abs(c.g - q.g) + abs(c.b - q.b);
                if ((bx & 2) != 0) sadB += d; else sadA += d;
            }
        }
        float sad = float(sadA) + float(sadB);
        sadGmv = (k == gmvIdx) ? sad : sadGmv;
        float better = step(sad, bestSad);
        bestSad = min(bestSad, sad);
        bestOff = mix(bestOff, cp, better);
    }
    if (uRefine != 0) {
        // 半像素子像素细化（8 候选：4 轴向 + 4 对角）
        vec2 sub[8];
        sub[0] = bestOff + vec2( 0.5,  0.0);
        sub[1] = bestOff + vec2(-0.5,  0.0);
        sub[2] = bestOff + vec2( 0.0,  0.5);
        sub[3] = bestOff + vec2( 0.0, -0.5);
        sub[4] = bestOff + vec2( 0.5,  0.5);
        sub[5] = bestOff + vec2(-0.5,  0.5);
        sub[6] = bestOff + vec2( 0.5, -0.5);
        sub[7] = bestOff + vec2(-0.5, -0.5);
        for (int i = 0; i < 8; i++) {
            vec2 cp = clamp(sub[i], vec2(-float(R)), vec2(float(R)));
            vec2 off = cp / dsSize;
            mediump float sadA = 0.0, sadB = 0.0;
            for (int by = 0; by < block; by += 2) {
                for (int bx = 0; bx < block; bx += 2) {
                    vec2 pp = base + (vec2(float(bx), float(by)) + 0.5 - 0.5 * float(block)) / dsSize;
                    mediump vec3 c = texture(uCur, pp).rgb;
                    mediump vec3 q = texture(uPrev, pp - off).rgb;
                    mediump float d = abs(c.r - q.r) + abs(c.g - q.g) + abs(c.b - q.b);
                    if ((bx & 2) != 0) sadB += d; else sadA += d;
                }
            }
            float sad = float(sadA) + float(sadB);
            float better = step(sad, bestSad);
            bestSad = min(bestSad, sad);
            bestOff = mix(bestOff, cp, better);
        }
    }
    float vx = 0.5, vy = 0.5, conf = 0.25;
    // 可信度门限无分支化：ok=1 才采用（bestSad < zeroSad*0.70）
    float ok = 1.0 - step(zeroSad * 0.70, bestSad);
    vec2 vn = (bestOff + vec2(float(R))) / (2.0 * vec2(float(R)));
    vx = mix(0.5, vn.x, ok);
    vy = mix(0.5, vn.y, ok);
    conf = mix(0.25, clamp(1.0 - bestSad / (float(block * block) * 0.375), 0.0, 1.0), ok);
    // 低置信向时间预测收缩：conf<0.5 时向上一帧同块矢量靠拢（w 由 max 自然截止，无分支）
    float w = max(0.5 - conf, 0.0) * 2.0 * uSmoothW;
    vx = mix(vx, tm.x, w);
    vy = mix(vy, tm.y, w);
    // 全局运动覆盖：平坦/低置信块若 GMV 匹配好（残差 ≤ 零位移误差），直接把运动矢量
    // 替换为 GMV——远景跟随相机运动。三个条件 → step 乘积（无分支），在运动场内部完成。
    float gm = (1.0 - step(0.35, conf)) * step(0.05, gmvData.z)
             * (1.0 - step(zeroSad * 1.05, sadGmv));
    vec2 gvN = gmvData.xy * 2.0 - 1.0;
    vx = mix(vx, (gvN.x + 1.0) * 0.5, gm);
    vy = mix(vy, (gvN.y + 1.0) * 0.5, gm);
    conf = mix(conf, min(conf + 0.15, 0.5), gm);   // 提升置信，合成端走 warp 而非帧混合
    imageStore(uMV, g, vec4(vx, vy, conf, 1.0));
}
)";

// GLES subgroup 版 pass1：16 lane 协作一个 block，每 lane 一个采样点，subgroupAdd 归约 SAD。
// dispatch 维度与原版相同 (gw,gh,1)——原版每线程一个 block，本版每 workgroup(16 lane) 一个 block。
// 需 GL_KHR_shader_subgroup_arithmetic；编译失败由 C++ 回退原版。
static const char *CS_ME1_SUB_SRC = R"(#version 310 es
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 16, local_size_y = 1) in;
layout(binding = 0) uniform highp sampler2D uPrev;
layout(binding = 1) uniform highp sampler2D uCur;
layout(binding = 2) uniform highp sampler2D uMVPrev;
layout(binding = 3) uniform highp sampler2D uMVPrevL1;
layout(binding = 4, rgba32f) uniform writeonly highp image2D uMV;
layout(binding = 5) uniform highp sampler2D uGMV;
uniform highp ivec4 uP0;
uniform highp ivec4 uP1;
uniform highp int uRefine;
uniform highp float uSmoothW;
uniform highp int uLv2;
void main() {
    ivec2 g = ivec2(gl_WorkGroupID.xy);
    int L = uLv2;
    int dsW = uP0.z, dsH = uP0.w, block = uP0.y, R = uP0.x;
    int R2 = uP1.x, mvW1 = uP1.y, mvH1 = uP1.z;
    int dsW2 = (dsW + L - 1) / L, dsH2 = (dsH + L - 1) / L;
    int mvW2 = (dsW2 + 7) / 8, mvH2 = (dsH2 + 7) / 8;
    int gw = (dsW + block - 1) / block;
    int gh = (dsH + block - 1) / block;
    if (g.x >= gw || g.y >= gh) return;
    vec2 dsSize = vec2(float(dsW), float(dsH));
    vec2 base = (vec2(g * block) + vec2(0.5 * float(block))) / dsSize;
    int lane = int(gl_SubgroupInvocationID);
    bool valid = lane < 16;
    int by = (lane >> 2) * 2, bx = (lane & 3) * 2;
    vec2 pp = base + (vec2(float(bx), float(by)) + 0.5 - 0.5 * float(block)) / dsSize;
    mediump vec3 c = texture(uCur, pp).rgb;
    mediump vec3 q0 = texture(uPrev, pp).rgb;
    float d0 = valid ? float(abs(c.r-q0.r)+abs(c.g-q0.g)+abs(c.b-q0.b)) : 0.0;
    float zeroSad = subgroupAdd(d0);
    vec2 c0 = texture(uMVPrev, (vec2(g) + 0.5) / vec2(float(mvW2), float(mvH2))).xy;
    vec2 center = (c0 * 2.0 - 1.0) * float(R2) * float(L);
    vec2 tm = texture(uMVPrevL1, (vec2(g) + 0.5) / vec2(float(mvW1), float(mvH1))).xy;
    vec2 tmPix = (tm * 2.0 - 1.0) * float(R);
    float bestSad = 1e30;
    vec2 bestOff = center;
    vec2 cand[15];
    int n = 0;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) cand[n++] = center + vec2(float(dx), float(dy));
    cand[n++] = tmPix;
    cand[n++] = tmPix + vec2( 1.0, 0.0); cand[n++] = tmPix + vec2(-1.0, 0.0);
    cand[n++] = tmPix + vec2(0.0,  1.0); cand[n++] = tmPix + vec2(0.0, -1.0);
    vec4 gmvData = texture(uGMV, vec2(0.5));
    int gmvIdx = -1;
    if (gmvData.z > 0.05) { gmvIdx = n; cand[n++] = (gmvData.xy * 2.0 - 1.0) * float(R); }
    int candN = (uRefine != 0) ? n : 1;
    float sadGmv = 1e30;
    for (int k = 0; k < candN; k++) {
        vec2 cp = clamp(cand[k], vec2(-float(R)), vec2(float(R)));
        vec2 off = cp / dsSize;
        vec3 q = texture(uPrev, pp - off).rgb;
        float d = valid ? float(abs(c.r-q.r)+abs(c.g-q.g)+abs(c.b-q.b)) : 0.0;
        float sad = subgroupAdd(d);
        sadGmv = (k == gmvIdx) ? sad : sadGmv;
        float better = step(sad, bestSad);
        bestSad = min(bestSad, sad);
        bestOff = mix(bestOff, cp, better);
    }
    if (uRefine != 0) {
        vec2 sub[8];
        sub[0]=bestOff+vec2( 0.5,0.0); sub[1]=bestOff+vec2(-0.5,0.0);
        sub[2]=bestOff+vec2(0.0, 0.5); sub[3]=bestOff+vec2(0.0,-0.5);
        sub[4]=bestOff+vec2( 0.5, 0.5); sub[5]=bestOff+vec2(-0.5, 0.5);
        sub[6]=bestOff+vec2( 0.5,-0.5); sub[7]=bestOff+vec2(-0.5,-0.5);
        for (int i = 0; i < 8; i++) {
            vec2 cp = clamp(sub[i], vec2(-float(R)), vec2(float(R)));
            vec2 off = cp / dsSize;
            vec3 q = texture(uPrev, pp - off).rgb;
            float d = valid ? float(abs(c.r-q.r)+abs(c.g-q.g)+abs(c.b-q.b)) : 0.0;
            float sad = subgroupAdd(d);
            float better = step(sad, bestSad);
            bestSad = min(bestSad, sad);
            bestOff = mix(bestOff, cp, better);
        }
    }
    if (lane == 0) {
        float vx = 0.5, vy = 0.5, conf = 0.25;
        float ok = 1.0 - step(zeroSad * 0.70, bestSad);
        vec2 vn = (bestOff + vec2(float(R))) / (2.0 * vec2(float(R)));
        vx = mix(0.5, vn.x, ok);
        vy = mix(0.5, vn.y, ok);
        conf = mix(0.25, clamp(1.0 - bestSad / (float(block * block) * 0.375), 0.0, 1.0), ok);
        float w = max(0.5 - conf, 0.0) * 2.0 * uSmoothW;
        vx = mix(vx, tm.x, w);
        vy = mix(vy, tm.y, w);
        float gm = (1.0 - step(0.35, conf)) * step(0.05, gmvData.z)
                 * (1.0 - step(zeroSad * 1.05, sadGmv));
        vec2 gvN = gmvData.xy * 2.0 - 1.0;
        vx = mix(vx, (gvN.x + 1.0) * 0.5, gm);
        vy = mix(vy, (gvN.y + 1.0) * 0.5, gm);
        conf = mix(conf, min(conf + 0.15, 0.5), gm);
        imageStore(uMV, g, vec4(vx, vy, conf, 1.0));
    }
}
)";

// 平滑 pass（CS_ME2_SRC）：对 mv 图做 3×3 中值滤波（按运动幅度），孤立错误矢量被邻居
// 拉平——视频/直播噪声区的矢量随机跳变（闪烁主因）大幅减少；conf 取 9 邻最大值，
// 局部一致则提升置信，孤立低置信被邻居救回。
static const char *CS_ME2_SRC = R"(#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0) uniform highp sampler2D uMV;
layout(binding = 1, rgba32f) uniform writeonly highp image2D uOut;
uniform highp ivec2 uSize;
void main() {
    ivec2 g = ivec2(gl_GlobalInvocationID.xy);
    ivec2 S = uSize;
    if (g.x >= S.x || g.y >= S.y) return;
    mediump vec2 v[9];
    mediump float conf = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 p = clamp(g + ivec2(dx, dy), ivec2(0), S - 1);
            mediump vec4 t = texelFetch(uMV, p, 0);
            v[(dy + 1) * 3 + (dx + 1)] = t.xy;
            conf = max(conf, t.z);
        }
    }
    // 按运动幅度升序冒泡 5 轮：v[4] 即中值矢量（对离群矢量鲁棒）
    for (int r = 0; r < 5; r++) {
        for (int i = 0; i < 8; i += 2) {
            if (length(v[i]) > length(v[i + 1])) { mediump vec2 t = v[i]; v[i] = v[i + 1]; v[i + 1] = t; }
        }
        for (int i = 1; i < 8; i += 2) {
            if (length(v[i]) > length(v[i + 1])) { mediump vec2 t = v[i]; v[i] = v[i + 1]; v[i + 1] = t; }
        }
    }
    // 中值 + 邻域均值混合：消除块边界矢量跳变（warp 后块状撕裂/折线）
    highp vec2 vavg = (vec2(v[0]) + vec2(v[1]) + vec2(v[2]) + vec2(v[3]) + vec2(v[4])
                     + vec2(v[5]) + vec2(v[6]) + vec2(v[7]) + vec2(v[8])) / 9.0;
    mediump vec2 vs = mix(v[4], vec2(vavg), 0.35);
    imageStore(uOut, g, vec4(vs, conf, 1.0));
}
)";

// 统计 pass（CS_ME3_SRC）：低置信（conf<0.5）块计数写入 SSBO，CPU 读回做整帧降级判定
static const char *CS_ME3_SRC = R"(#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0) uniform highp sampler2D uMV;
layout(binding = 1, std430) buffer Stats { uint lowCount; };
uniform highp ivec2 uSize;
void main() {
    ivec2 g = ivec2(gl_GlobalInvocationID.xy);
    if (g.x >= uSize.x || g.y >= uSize.y) return;
    // 无分支：conf<0.5 → 计 1，否则计 0（step(0.5, conf)=1 当 conf>=0.5）
    uint confOk = uint(step(0.5, texelFetch(uMV, g, 0).z));
    atomicAdd(lowCount, 1u - confOk);
}
)";

// ---------------- GL 工具 ----------------

static GLuint mcfi_compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        LOGE("shader compile fail: %s", log);
    }
    return s;
}

static GLuint mcfi_link_program(const char *vs_src, const char *fs_src) {
    GLuint vs = mcfi_compile(GL_VERTEX_SHADER, vs_src);
    GLuint fs = mcfi_compile(GL_FRAGMENT_SHADER, fs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        LOGE("program link fail: %s", log);
        glDeleteProgram(p);
        return 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

static GLuint mcfi_link_compute(const char *cs_src) {
    GLuint cs = mcfi_compile(GL_COMPUTE_SHADER, cs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, cs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        LOGE("compute link fail: %s", log);
        glDeleteProgram(p);
        return 0;
    }
    glDeleteShader(cs);
    return p;
}

static bool gles_subgroup_ok();
// ME1 选择：探测 GL_KHR_shader_subgroup_arithmetic 可用时链接 subgroup 版，
// 编译/链接失败或扩展不支持 → 回退原版（每线程一个 block）。
static GLuint link_me1() {
    if (gles_subgroup_ok()) {
        GLuint p = mcfi_link_compute(CS_ME1_SUB_SRC);
        if (p) { LOGI("GLES ME1 已接入 subgroup 归约版"); return p; }
    }
    return mcfi_link_compute(CS_ME1_SRC);
}

static bool gles3_compute_ok() {
    GLint major = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    return major >= 3;   // GLES 3.1+ 必有 compute；按 3.2 要求运行
}

static void make_tex(GLuint &tex, int w, int h, bool f32 = false) {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, f32 ? GL_RGBA32F : GL_RGBA8, w, h, 0, GL_RGBA,
                 f32 ? GL_FLOAT : GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// 插帧算法名（用于日志/面板区分当前运行的是哪种算法）
static const char *mcfi_algo_name(int mode) {
    return mode == 0 ? "MCI运动补偿" : (mode == 2 ? "运动自适应" : "普通混合");
}

// 探测 GLES mediump 是否为真 fp16：fp16 尾数 10 位；fp32 模拟时 precision≥23
// （GLES 精度模型与 compute 一致，用 fragment 精度查询即可）
static bool gles_mediump_fp16() {
    GLint range[2] = {0, 0}, prec = 0;
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_MEDIUM_FLOAT, range, &prec);
    return prec > 0 && prec <= 13;
}

// 探测 GLES subgroup（warp-level）支持：SAD 像素求和可在 lane 间并行 + subgroupAdd 归约。
// 仅探测上报，不强制启用——不同厂商 subgroupSize 差异大，实际加速版需真机验证后接入。
static bool gles_subgroup_ok() {
    const GLubyte *ext = glGetString(GL_EXTENSIONS);
    if (!ext) return false;
    std::string s((const char *)ext);
    return s.find("GL_KHR_shader_subgroup_arithmetic") != std::string::npos
        || s.find("GL_KHR_shader_subgroup_shuffle") != std::string::npos;
}

static void make_fbo(GLuint &fbo, GLuint tex) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// 用 copy 程序把纹理画满目标视口（线性缩放）
static void draw_texture(GLuint progCopy, GLuint vao, GLuint tex) {
    glUseProgram(progCopy);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(progCopy, "uTex"), 0);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

// 运行多尺度运动估计（金字塔 3DRS）：prev/cur 为全分辨率纹理。
//   pass0（L2=1/16 层）：读 mvPrevL1（上一帧 L1 平滑场，时间/空间预测源）+ dsL2，粗搜索写 mvA
//   pass1（L1=1/4 层）：以 mvA 放大 4 倍为中心 ±1 + 时间预测 + 半像素细化，写 mvB
// 返回 mvB（L1 场）。R2 = max(1, range/4)：L2 半径 1 对应 ±4 L1 像素 = ±range（range=4 时）全覆盖。
// GMV（全局运动向量）资源：直方图统计两 pass + 1×1 输出纹理 + 直方图 SSBO
struct GmvRes {
    GLuint prog[2] = {0, 0};   // CS_GMV0（直方图）/ CS_GMV1（归约）
    GLuint tex = 0;            // 1×1 rgba32f（x,y=归一化 GMV z=峰值占比）
    GLuint hist = 0;           // SSBO：1024 bins + total（std430）
};
static bool gmv_init(GmvRes &g) {
    g.prog[0] = mcfi_link_compute(CS_GMV0_SRC);
    g.prog[1] = mcfi_link_compute(CS_GMV1_SRC);
    if (!g.prog[0] || !g.prog[1]) { LOGE("GMV 程序创建失败"); return false; }
    glGenTextures(1, &g.tex);
    glBindTexture(GL_TEXTURE_2D, g.tex);
    const float zeroGmv[4] = {0.5f, 0.5f, 0.0f, 0.0f};   // 零运动 + 无 GMV，防降级路径读到垃圾矢量
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, zeroGmv);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenBuffers(1, &g.hist);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g.hist);
    static const int kHistSize = (1024 + 1) * sizeof(int);
    glBufferData(GL_SHADER_STORAGE_BUFFER, kHistSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return true;
}

static void gmv_free(GmvRes &g) {
    if (g.prog[0]) glDeleteProgram(g.prog[0]);
    if (g.prog[1]) glDeleteProgram(g.prog[1]);
    if (g.tex) glDeleteTextures(1, &g.tex);
    if (g.hist) glDeleteBuffers(1, &g.hist);
    g = GmvRes{};
}

static GLuint run_motion_estimation(GLuint progME[2], GLuint progCopy, GLuint vao,
                                    GLuint prevFull, GLuint curFull,
                                    GLuint dsTex[4], GLuint dsFbo[4],
                                    GLuint mvPrevL1, GLuint mvA, GLuint mvB,
                                    int w, int h, int dsW, int dsH, int dsW2, int dsH2,
                                    int mvW1, int mvH1, int mvW2, int mvH2,
                                    int range, bool refine, float smoothW = 0.6f,
                                    GmvRes *gmv = nullptr, int lv2 = 4) {
    // 1. 降采样 4 张：prev/cur → 1/4、L2 层（纹理采样 LINEAR 缩放，等效预滤波）
    //    lv2=4 → L2=1/16（游戏）；lv2=2 → L2=1/8（视频，粗搜索更准）
    for (int i = 0; i < 4; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, dsFbo[i]);
        glViewport(0, 0, (i < 2) ? dsW : dsW2, (i < 2) ? dsH : dsH2);
        draw_texture(progCopy, vao, ((i & 1) == 0) ? prevFull : curFull);
    }
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    int R2 = (range / lv2 < 1) ? 1 : range / lv2;

    // 2. pass0（L2）：3DRS 粗搜索（读 mvPrevL1 + dsL2，writeonly 写 mvA）
    glUseProgram(progME[0]);
    glUniform4i(glGetUniformLocation(progME[0], "uP0"), R2, 8, dsW2, dsH2);
    glUniform4i(glGetUniformLocation(progME[0], "uP1"), range, mvW1, mvH1, mvW2);
    glUniform1i(glGetUniformLocation(progME[0], "uLv2"), lv2);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, dsTex[2]);
    glUniform1i(glGetUniformLocation(progME[0], "uPrev"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, dsTex[3]);
    glUniform1i(glGetUniformLocation(progME[0], "uCur"), 1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mvPrevL1);
    glUniform1i(glGetUniformLocation(progME[0], "uMvPrev"), 2);
    glBindImageTexture(3, mvA, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    int gw2 = (dsW2 + 7) / 8, gh2 = (dsH2 + 7) / 8;
    glDispatchCompute(gw2, gh2, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // 2b. GMV（全局运动向量）统计：L2 场直方图 → 归约找主导运动 → 1×1 纹理。
    //     远景平坦区 3DRS 找不到可靠 MV，由 GMV 候选/合成回退补偿相机运动。
    if (gmv && gmv->prog[0] && gmv->prog[1]) {
        int binW = 2 * R2 + 2;
        static const int kZero[1025] = {0};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gmv->hist);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(kZero), kZero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        // CS_GMV0：直方图
        glUseProgram(gmv->prog[0]);
        glUniform4i(glGetUniformLocation(gmv->prog[0], "uP0"), R2, mvW2, mvH2, binW);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mvA);
        glUniform1i(glGetUniformLocation(gmv->prog[0], "uMV"), 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gmv->hist);
        int gwv = (mvW2 + 7) / 8, ghv = (mvH2 + 7) / 8;
        if (gwv < 1) gwv = 1;
        if (ghv < 1) ghv = 1;
        glDispatchCompute(gwv, ghv, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        // CS_GMV1：归约 → 写 GMV 纹理
        glUseProgram(gmv->prog[1]);
        glUniform4i(glGetUniformLocation(gmv->prog[1], "uP0"), binW, R2, 0, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gmv->hist);
        glBindImageTexture(1, gmv->tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // 3. pass1（L1）：mvA 放大为中心 + 时间预测细化，writeonly 写 mvB（始终跑，输出 L1 场）
    if (progME[1]) {
        glUseProgram(progME[1]);
        glUniform4i(glGetUniformLocation(progME[1], "uP0"), range, 8, dsW, dsH);
        glUniform4i(glGetUniformLocation(progME[1], "uP1"), R2, mvW1, mvH1, mvW2);
        glUniform1i(glGetUniformLocation(progME[1], "uLv2"), lv2);
        glUniform1i(glGetUniformLocation(progME[1], "uRefine"), refine ? 1 : 0);
        glUniform1f(glGetUniformLocation(progME[1], "uSmoothW"), smoothW);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, dsTex[0]);
        glUniform1i(glGetUniformLocation(progME[1], "uPrev"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, dsTex[1]);
        glUniform1i(glGetUniformLocation(progME[1], "uCur"), 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, mvA);
        glUniform1i(glGetUniformLocation(progME[1], "uMVPrev"), 2);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, mvPrevL1);
        glUniform1i(glGetUniformLocation(progME[1], "uMVPrevL1"), 3);
        if (gmv && gmv->tex) {
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, gmv->tex);
            glUniform1i(glGetUniformLocation(progME[1], "uGMV"), 5);
        }
        glBindImageTexture(4, mvB, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        int gw = (dsW + 7) / 8, gh = (dsH + 7) / 8;
        glDispatchCompute(gw, gh, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        return mvB;
    }
    return mvA;
}

// 中值平滑 + 低置信统计：读 mv 输入（mvA/mvB），写 mvC，SSBO 统计低置信块数并读回，
// 返回低置信占比（0~1）。此函数会在 worker 线程同步读回 GPU 结果（4 字节，开销可忽略）。
static float run_motion_smooth_stats(GLuint progME[4], GLuint inMV, GLuint outMV,
                                     GLuint statsBuf, int mvW, int mvH) {
    int gw = (mvW + 7) / 8, gh = (mvH + 7) / 8;
    if (gw <= 0 || gh <= 0) return 0.0f;

    // 平滑：inMV(sampler) -> outMV(writeonly image)
    glUseProgram(progME[2]);
    glUniform2i(glGetUniformLocation(progME[2], "uSize"), mvW, mvH);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inMV);
    glUniform1i(glGetUniformLocation(progME[2], "uMV"), 0);
    glBindImageTexture(1, outMV, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(gw, gh, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // 清统计计数
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuf);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zero), &zero);

    // 统计：outMV(sampler) -> SSBO atomicAdd
    glUseProgram(progME[3]);
    glUniform2i(glGetUniformLocation(progME[3], "uSize"), mvW, mvH);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, outMV);
    glUniform1i(glGetUniformLocation(progME[3], "uMV"), 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, statsBuf);
    glDispatchCompute(gw, gh, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    uint32_t low = 0;
    void *ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(low), GL_MAP_READ_BIT);
    if (ptr) {
        memcpy(&low, ptr, sizeof(low));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return (float)low / (float)(mvW * mvH);
}

// 运动补偿合成：把中间帧画到 dstFbo（0=默认帧缓冲）
static void run_mci_pass(GLuint progInterp, GLuint vao,
                         GLuint prevFull, GLuint curFull, GLuint mvTex,
                         GLuint dstFbo, int w, int h, int dsW, int dsH,
                         int mode, float strength, int range, float smooth = 60.0f) {
    glBindFramebuffer(GL_FRAMEBUFFER, dstFbo);
    glViewport(0, 0, w, h);
    glUseProgram(progInterp);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, prevFull);
    glUniform1i(glGetUniformLocation(progInterp, "uPrev"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, curFull);
    glUniform1i(glGetUniformLocation(progInterp, "uCur"), 1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mvTex);
    glUniform1i(glGetUniformLocation(progInterp, "uMV"), 2);
    glUniform1i(glGetUniformLocation(progInterp, "uMode"), mode);
    glUniform1f(glGetUniformLocation(progInterp, "uStrength"), strength);
    glUniform1f(glGetUniformLocation(progInterp, "uRange"), (float)range);
    glUniform1f(glGetUniformLocation(progInterp, "uSmooth"), smooth);
    glUniform2f(glGetUniformLocation(progInterp, "uDsTexel"),
                1.0f / (float)dsW, 1.0f / (float)dsH);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

// ---------------- GL 状态保存/恢复 ----------------

struct GLState {
    GLint drawFbo, readFbo, prog, viewport[4], activeTex;
    GLint texUnit0, texUnit1, texUnit2, arrayBuffer, vao;
    GLboolean blend, scissor, depthTest, cullFace, depthMask;
};

static void saveState(GLState &s) {
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.drawFbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFbo);
    glGetIntegerv(GL_CURRENT_PROGRAM, &s.prog);
    glGetIntegerv(GL_VIEWPORT, s.viewport);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &s.activeTex);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.texUnit0);
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.texUnit1);
    glActiveTexture(GL_TEXTURE2);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.texUnit2);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &s.arrayBuffer);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
    s.blend = glIsEnabled(GL_BLEND);
    s.scissor = glIsEnabled(GL_SCISSOR_TEST);
    s.depthTest = glIsEnabled(GL_DEPTH_TEST);
    s.cullFace = glIsEnabled(GL_CULL_FACE);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &s.depthMask);
}

static void restoreState(const GLState &s) {
    glUseProgram((GLuint)s.prog);
    glBindVertexArray((GLuint)s.vao);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)s.arrayBuffer);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, (GLuint)s.texUnit0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, (GLuint)s.texUnit1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, (GLuint)s.texUnit2);
    glActiveTexture((GLenum)s.activeTex);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)s.drawFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s.readFbo);
    glViewport(s.viewport[0], s.viewport[1], s.viewport[2], s.viewport[3]);
    if (s.blend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (s.scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    if (s.depthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (s.cullFace) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    glDepthMask(s.depthMask);
}

// ============================================================
// GLES 异步 worker（共享 EGL 上下文）
// 视频模式与游戏模式统一走此流水线：
//   app 线程只拷贝真实帧 + 非阻塞呈现生成帧（零 GPU 重活）；
//   运动估计/中间帧合成全部在 worker 线程。
// ============================================================

#define ASYNC_SLOTS 4

// GMV（全局运动向量）资源：直方图统计两 pass + 1×1 输出纹理 + 直方图 SSBO
struct AsyncCtx {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLConfig cfg = nullptr;
    EGLContext appCtx = EGL_NO_CONTEXT;
    EGLContext workerCtx = EGL_NO_CONTEXT;
    EGLSurface pbuf = EGL_NO_SURFACE;

    GLuint progCopy = 0, progInterp = 0, progME[4] = {0, 0, 0, 0}, vao = 0;
    struct Slot {
        GLuint tex = 0, fbo = 0;
        GLsync copySync = nullptr;
        std::atomic<int> state{0};   // 0=free 1=busy(app 已写入等待 worker)
        int order = 0;               // app 侧帧序号（app 线程写，worker 只读）
    } slots[ASYNC_SLOTS];
    GLuint genTex = 0, genFbo = 0;
    GLsync doneSync = nullptr;
    std::atomic<int> genEpoch{0};
    std::atomic<int> genConsumed{1};
    std::atomic<int> genOrder{0};    // 最近完成生成帧对应的真实帧序号（worker 写）
    int appOrder = 0;                // 已入队最大帧序号（app 线程局部）
    int lastPresented = 0;           // app 线程局部（仅 hook 线程访问）
    GLuint dsTex[4] = {0, 0, 0, 0}, dsFbo[4] = {0, 0, 0, 0};  // [0]=prev 1/4 [1]=cur 1/4 [2]=prev 1/16 [3]=cur 1/16
    GLuint mvTex[3] = {0, 0, 0};     // mvA=L2 粗场(1/16 层) mvB=L1 场(1/4 层) mvC=L1 平滑输出(也是下一帧时间预测源)
    GLuint statsBuf = 0;             // 低置信块计数 SSBO（4 字节）
    GmvRes gmv;                      // GMV 直方图/输出
    int w = 0, h = 0, dsW = 0, dsH = 0, dsW2 = 0, dsH2 = 0;
    int mvW1 = 0, mvH1 = 0, mvW2 = 0, mvH2 = 0;
    std::atomic<int> lastCurSlot{-1};
    std::atomic<bool> hasPrev{false};
    std::atomic<bool> stop{false};
    std::thread worker;
    std::mutex qmtx;
    std::condition_variable qcv;
    std::queue<int> jobs;
    std::mutex cmtx;
    std::condition_variable ccv;
    std::atomic<long> jobCount{0};
    int nextFree = 0;
    bool meOk = true;
    int lastOrder = -1;            // worker 已处理的上一真实帧序号（worker 线程独占；跨帧检测用）
    bool broken = false;           // 初始化失败熔断：本运行不再尝试插帧（防反复失败卡死/资源耗尽）
};

static void async_free_egl_gl(AsyncCtx *a);
static void async_teardown(AsyncCtx *a);
static bool async_init(AsyncCtx *a, EGLContext appCtx, int w, int h);

// 3 套异步资源池（LRU）：UI/视频/弹幕 context 切换不再反复 teardown+init（重建风暴），
// 视频模式防爆内存（最多 3 套上限）。每套约 60~70MB@1080p。
static AsyncCtx g_pool[3];
static EGLContext g_pool_ctx[3] = {EGL_NO_CONTEXT, EGL_NO_CONTEXT, EGL_NO_CONTEXT};
static int g_pool_lru[3] = {0, 0, 0};
static int g_lru_tick = 0;

// 按 EGLContext 获取/创建异步资源：命中直接用；未命中 LRU 淘汰最久未用的一套重建。
// 初始化失败置 broken（该 context 本运行不再插帧）。返回 nullptr 表示不可用（原样交换）。
static AsyncCtx *mcfi_async_acquire(EGLContext ctx, int w, int h) {
    for (int i = 0; i < 3; i++) {
        if (g_pool_ctx[i] == ctx) {
            g_pool_lru[i] = ++g_lru_tick;
            AsyncCtx *a = &g_pool[i];
            if (a->workerCtx == EGL_NO_CONTEXT || a->appCtx != ctx || a->w != w || a->h != h) {
                if (a->broken) return nullptr;
                if (a->workerCtx != EGL_NO_CONTEXT) async_teardown(a);
                if (!async_init(a, ctx, w, h)) {
                    a->broken = true;
                    LOGE("GLES 异步初始化失败，本运行熔断（不再尝试插帧，画面以原始帧呈现）");
                    return nullptr;
                }
            }
            return a;
        }
    }
    int victim = 0;
    for (int i = 1; i < 3; i++) if (g_pool_lru[i] < g_pool_lru[victim]) victim = i;
    AsyncCtx *a = &g_pool[victim];
    if (a->workerCtx != EGL_NO_CONTEXT) async_teardown(a);
    g_pool_ctx[victim] = ctx;
    g_pool_lru[victim] = ++g_lru_tick;
    if (!async_init(a, ctx, w, h)) {
        a->broken = true;
        LOGE("GLES 异步初始化失败，本运行熔断（不再尝试插帧，画面以原始帧呈现）");
        return nullptr;
    }
    return a;
}

static void async_worker_main(AsyncCtx *a) {
    pthread_setname_np(pthread_self(), "MCFI-gles");
    if (!eglMakeCurrent(a->dpy, a->pbuf, a->pbuf, a->workerCtx)) {
        LOGE("GLES worker eglMakeCurrent 失败");
        return;
    }
    LOGI("GLES 异步工作线程已启动 (pid=%d)", getpid());
    McfiConfig cfg;
    mcfi_get_config(&cfg);
    int range = cfg.me_range();
    bool refine = cfg.me_refine() && a->meOk;

    for (;;) {
        int slot = -1;
        {
            std::unique_lock<std::mutex> lk(a->qmtx);
            a->qcv.wait(lk, [&] { return a->stop.load() || !a->jobs.empty(); });
            if (a->stop.load()) break;
            slot = a->jobs.front();
            a->jobs.pop();
        }
        if (slot < 0 || slot >= ASYNC_SLOTS) continue;

        // 等 app 线程的拷贝完成（GPU 侧等待）。16ms 超时：应用切 context/销毁时 GL fence
        // 可能永不 signal，无限等待会让 async_teardown 的 join 卡死 → 黑屏。
        GLsync cs = a->slots[slot].copySync;
        if (cs) {
            GLenum st = glClientWaitSync(cs, 0, 16000000);
            glDeleteSync(cs);
            a->slots[slot].copySync = nullptr;
            if (st == GL_TIMEOUT_EXPIRED) {
                // 拷贝未完成（context 切换/销毁），丢弃本槽不插帧；重置前后帧关系
                a->slots[slot].state = 0;
                a->lastCurSlot = slot;
                a->hasPrev = false;
                a->lastOrder = -1;
                continue;
            }
        }

        int prevSlot = a->lastCurSlot.load();
        int curOrder = a->slots[slot].order;
        long jobNo = ++a->jobCount;
        mcfi_get_config(&cfg);
        range = cfg.me_range();
        refine = cfg.me_refine() && a->meOk;
        // 相邻检查：prev/cur 必须严格相邻（order 差 1）。app 丢帧后若直接合成跨帧中间帧，
        // 运动位置错位 → 画面跳变闪烁（直播场景尤其明显）。跨帧时跳过本帧插帧。
        bool adjacent = (a->lastOrder == -1) || (curOrder - a->lastOrder == 1);
        bool do_insert = a->hasPrev.load() && adjacent && (jobNo % cfg.game_interval) == 0;

        if (do_insert) {
            // 等上一张生成帧被消费（防覆盖 & 防堆积）
            {
                std::unique_lock<std::mutex> lk(a->cmtx);
                a->ccv.wait(lk, [&] { return a->genConsumed.load() != 0 || a->stop.load(); });
            }
            if (a->stop.load()) break;
            a->genConsumed = 0;

            if (a->meOk) {
                // 多尺度 3DRS：mvPrevL1 = 上一帧平滑场 mvC（时间/空间预测源），
                // 粗场写 mvA（L2），细场写 mvB（L1），平滑输出 mvC 作为下一帧预测源
                GLuint mvOut = run_motion_estimation(a->progME, a->progCopy, a->vao,
                                                     a->slots[prevSlot].tex, a->slots[slot].tex,
                                                     a->dsTex, a->dsFbo,
                                                     a->mvTex[2],
                                                     a->mvTex[0],
                                                     a->mvTex[1],
                                                     a->w, a->h, a->dsW, a->dsH,
                                                     a->dsW2, a->dsH2,
                                                     a->mvW1, a->mvH1, a->mvW2, a->mvH2,
                                                     range, refine, cfg.smooth / 100.0f, &a->gmv);
                // 中值平滑 + 低置信整体降级：噪声视频/直播低置信块占比高 → 整帧普通混合（不逐块 warp，杜绝闪烁）
                float lowRatio = run_motion_smooth_stats(a->progME, mvOut, a->mvTex[2],
                                                         a->statsBuf, a->mvW1, a->mvH1);
                int imode = cfg.interp_mode;
                if (lowRatio > 0.40f) imode = 1;
                static long s_deg_cnt = 0, s_tot_cnt = 0;
                s_tot_cnt++;
                if (imode != cfg.interp_mode) s_deg_cnt++;
                if ((s_tot_cnt % 300) == 0) {
                    LOGI("GLES MCI 低置信统计: %ld/%ld 帧整帧降级混合 (占比阈值0.40)", s_deg_cnt, s_tot_cnt);
                }
                run_mci_pass(a->progInterp, a->vao,
                             a->slots[prevSlot].tex, a->slots[slot].tex,
                             a->mvTex[2], a->genFbo,
                             a->w, a->h, a->dsW, a->dsH,
                             imode, cfg.strength / 100.0f, range, (float)cfg.smooth);
            } else {
                run_mci_pass(a->progInterp, a->vao,
                             a->slots[prevSlot].tex, a->slots[slot].tex,
                             a->mvTex[0], a->genFbo,
                             a->w, a->h, a->dsW, a->dsH,
                             1, cfg.strength / 100.0f, range, (float)cfg.smooth);
            }

            if (a->doneSync) { glDeleteSync(a->doneSync); a->doneSync = nullptr; }
            a->doneSync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
            a->genEpoch++;
            a->genOrder.store(a->slots[slot].order, std::memory_order_release);

            // 等 app 呈现完生成帧（此时 prev 槽位的 GPU 读取已全部完成），再释放 prev 槽
            {
                std::unique_lock<std::mutex> lk(a->cmtx);
                a->ccv.wait(lk, [&] { return a->genConsumed.load() != 0 || a->stop.load(); });
            }
            if (a->stop.load()) break;
            if (prevSlot >= 0 && prevSlot < ASYNC_SLOTS)
                a->slots[prevSlot].state = 0;   // 释放 prev 槽
        } else {
            // 跳帧（game_interval>1 时的非插帧帧）：prevSlot 未被本帧 ME/MCI 读取，立即释放，
            // 否则旧 lastCurSlot 会被本帧覆盖后永不归还——4 个环形槽几帧内全部卡死在
            // state=1，app 侧找不到空闲槽而丢帧，表现为 N=2/N=3 插帧完全不生效。
            if (prevSlot >= 0 && prevSlot < ASYNC_SLOTS && prevSlot != slot)
                a->slots[prevSlot].state = 0;
        }
        a->lastCurSlot = slot;
        a->hasPrev = true;
        a->lastOrder = curOrder;
    }

    eglMakeCurrent(a->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

// 创建异步资源（在 app 线程、app 上下文 current 时调用）
static bool async_init(AsyncCtx *a, EGLContext appCtx, int w, int h) {
    a->dpy = eglGetCurrentDisplay();
    a->appCtx = appCtx;

    // 查当前上下文 config（用于创建共享上下文与 pbuffer）
    EGLint cfgId = 0;
    if (!eglQueryContext(a->dpy, appCtx, EGL_CONFIG_ID, &cfgId)) {
        LOGE("eglQueryContext CONFIG_ID 失败");
        return false;
    }
    EGLint n = 0;
    eglGetConfigs(a->dpy, nullptr, 0, &n);
    std::vector<EGLConfig> cfgs((size_t)n);
    eglGetConfigs(a->dpy, cfgs.data(), n, &n);
    EGLConfig cfg = nullptr;
    for (EGLConfig c : cfgs) {
        EGLint id = 0;
        eglGetConfigAttrib(a->dpy, c, EGL_CONFIG_ID, &id);
        if (id == cfgId) { cfg = c; break; }
    }
    if (!cfg) {
        // 兜底：app config 不在全局列表（私有config/no_config_context等）。
        // 按「支持ES3 + 颜色位深最大」选 config（另一个版本验证过的回退策略）。
        EGLint fallbackAttr[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_RED_SIZE, EGL_DONT_CARE, EGL_GREEN_SIZE, EGL_DONT_CARE,
            EGL_BLUE_SIZE, EGL_DONT_CARE, EGL_ALPHA_SIZE, EGL_DONT_CARE,
            EGL_DEPTH_SIZE, EGL_DONT_CARE,
            EGL_NONE
        };
        EGLint fn = 0;
        eglChooseConfig(a->dpy, fallbackAttr, nullptr, 0, &fn);
        if (fn > 0) {
            std::vector<EGLConfig> fc((size_t)fn);
            eglChooseConfig(a->dpy, fallbackAttr, fc.data(), fn, &fn);
            cfg = fc[0];
            LOGI("未匹配到 app EGLConfig，按 ES3 回退（%d 候选）", (int)fn);
        }
    }
    if (!cfg) { LOGE("找不到匹配 EGLConfig"); return false; }
    a->cfg = cfg;

    const EGLint ctxAttrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    a->workerCtx = eglCreateContext(a->dpy, cfg, appCtx, ctxAttrs);
    if (a->workerCtx == EGL_NO_CONTEXT) {
        LOGE("eglCreateContext(shared) 失败: 0x%x", (unsigned)eglGetError());
        return false;
    }
    const EGLint pbAttrs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    a->pbuf = eglCreatePbufferSurface(a->dpy, cfg, pbAttrs);
    if (a->pbuf == EGL_NO_SURFACE) {
        LOGE("eglCreatePbufferSurface 失败: 0x%x", (unsigned)eglGetError());
        eglDestroyContext(a->dpy, a->workerCtx);
        a->workerCtx = EGL_NO_CONTEXT;
        return false;
    }

    // 资源（app 上下文创建，share group 内 worker 可共享）
    a->w = w; a->h = h;
    a->dsW = (w + 3) / 4; a->dsH = (h + 3) / 4;          // L1 降采样（1/4）
    a->dsW2 = (a->dsW + 3) / 4; a->dsH2 = (a->dsH + 3) / 4;  // L2 降采样（1/16）
    a->mvW1 = (a->dsW + 7) / 8; a->mvH1 = (a->dsH + 7) / 8; // L1 运动场
    a->mvW2 = (a->dsW2 + 7) / 8; a->mvH2 = (a->dsH2 + 7) / 8; // L2 运动场
    a->progCopy = mcfi_link_program(VS_SRC, FS_COPY_SRC);
    a->progInterp = mcfi_link_program(VS_SRC, FS_INTERP_SRC);
    a->meOk = gles3_compute_ok();
    if (a->meOk) {
        a->progME[0] = mcfi_link_compute(CS_ME0_SRC);
        a->progME[1] = link_me1();
        a->progME[2] = mcfi_link_compute(CS_ME2_SRC);
        a->progME[3] = mcfi_link_compute(CS_ME3_SRC);
        if (!a->progME[0] || !a->progME[1] || !a->progME[2] || !a->progME[3]) {
            LOGE("compute 链接失败，运动补偿降级为普通混合");
            a->meOk = false;
        }
    } else {
        LOGE("当前 GLES 无 compute 支持，运动补偿降级为普通混合");
    }
    glGenVertexArrays(1, &a->vao);
    for (int i = 0; i < ASYNC_SLOTS; i++) {
        make_tex(a->slots[i].tex, w, h);
        make_fbo(a->slots[i].fbo, a->slots[i].tex);
    }
    make_tex(a->genTex, w, h);
    make_fbo(a->genFbo, a->genTex);
    glGenTextures(4, a->dsTex);
    glGenFramebuffers(4, a->dsFbo);
    for (int i = 0; i < 4; i++) {
        make_tex(a->dsTex[i], (i < 2) ? a->dsW : a->dsW2, (i < 2) ? a->dsH : a->dsH2);
        make_fbo(a->dsFbo[i], a->dsTex[i]);
    }
    // rgba32f + writeonly image：绕开部分驱动不支持 read/write 混合访问的限制
    make_tex(a->mvTex[0], a->mvW2, a->mvH2, true);   // mvA：L2 粗场
    make_tex(a->mvTex[1], a->mvW1, a->mvH1, true);   // mvB：L1 场
    make_tex(a->mvTex[2], a->mvW1, a->mvH1, true);   // mvC：L1 平滑输出（下一帧时间预测源）
    // 3DRS 时间预测源初始化为零矢量（0.5,0.5），避免首帧读到空纹理（全 0 → 错误候选）
    {
        std::vector<float> mv0L2((size_t)a->mvW2 * a->mvH2 * 4, 0.5f);
        for (size_t i = 3; i < mv0L2.size(); i += 4) mv0L2[i] = 1.0f;
        std::vector<float> mv0L1((size_t)a->mvW1 * a->mvH1 * 4, 0.5f);
        for (size_t i = 3; i < mv0L1.size(); i += 4) mv0L1[i] = 1.0f;
        glBindTexture(GL_TEXTURE_2D, a->mvTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a->mvW2, a->mvH2, GL_RGBA, GL_FLOAT, mv0L2.data());
        for (int i = 1; i < 3; i++) {
            glBindTexture(GL_TEXTURE_2D, a->mvTex[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, a->mvW1, a->mvH1, GL_RGBA, GL_FLOAT, mv0L1.data());
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glGenBuffers(1, &a->statsBuf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, a->statsBuf);
    uint32_t zero = 0;
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), &zero, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    if (a->meOk && !gmv_init(a->gmv)) a->meOk = false;   // GMV 失败不致命，仅无 GMV

    if (!a->progCopy || !a->progInterp) {
        LOGE("GLES 程序创建失败");
        async_free_egl_gl(a);
        return false;
    }
    a->genEpoch = 0;
    a->genConsumed = 1;
    a->genOrder = 0;
    a->appOrder = 0;
    a->lastPresented = 0;
    a->lastCurSlot = -1;
    a->hasPrev = false;
    a->jobCount = 0;
    a->lastOrder = -1;
    for (auto &s : a->slots) s.state = 0;
    a->nextFree = 0;

    a->worker = std::thread(async_worker_main, a);
    return true;
}

static void async_free_egl_gl(AsyncCtx *a) {
    if (a->dpy != EGL_NO_DISPLAY) {
        if (a->workerCtx != EGL_NO_CONTEXT) {
            eglDestroyContext(a->dpy, a->workerCtx);
            a->workerCtx = EGL_NO_CONTEXT;
        }
        if (a->pbuf != EGL_NO_SURFACE) {
            eglDestroySurface(a->dpy, a->pbuf);
            a->pbuf = EGL_NO_SURFACE;
        }
    }
    // GL 对象（app 上下文 current 时删除，share group 内任意上下文均可）
    if (a->progCopy) glDeleteProgram(a->progCopy);
    if (a->progInterp) glDeleteProgram(a->progInterp);
    if (a->progME[0]) glDeleteProgram(a->progME[0]);
    if (a->progME[1]) glDeleteProgram(a->progME[1]);
    if (a->progME[2]) glDeleteProgram(a->progME[2]);
    if (a->progME[3]) glDeleteProgram(a->progME[3]);
    if (a->statsBuf) glDeleteBuffers(1, &a->statsBuf);
    if (a->vao) glDeleteVertexArrays(1, &a->vao);
    for (int i = 0; i < ASYNC_SLOTS; i++) {
        if (a->slots[i].tex) glDeleteTextures(1, &a->slots[i].tex);
        if (a->slots[i].fbo) glDeleteFramebuffers(1, &a->slots[i].fbo);
        if (a->slots[i].copySync) { glDeleteSync(a->slots[i].copySync); a->slots[i].copySync = nullptr; }
        a->slots[i].tex = 0; a->slots[i].fbo = 0; a->slots[i].state = 0;
    }
    if (a->genTex) glDeleteTextures(1, &a->genTex);
    if (a->genFbo) glDeleteFramebuffers(1, &a->genFbo);
    if (a->dsTex[0]) glDeleteTextures(4, a->dsTex);
    if (a->dsFbo[0]) glDeleteFramebuffers(4, a->dsFbo);
    if (a->mvTex[0]) glDeleteTextures(3, a->mvTex);
    gmv_free(a->gmv);
    if (a->doneSync) { glDeleteSync(a->doneSync); a->doneSync = nullptr; }
    a->genTex = 0; a->genFbo = 0;
    for (int i = 0; i < 4; i++) { a->dsTex[i] = 0; a->dsFbo[i] = 0; }
    a->mvTex[0] = a->mvTex[1] = a->mvTex[2] = 0;
    a->statsBuf = 0;
    a->progCopy = a->progInterp = 0;
    a->progME[0] = a->progME[1] = a->progME[2] = a->progME[3] = 0;
    a->vao = 0;
    a->dpy = EGL_NO_DISPLAY;
    a->appCtx = EGL_NO_CONTEXT;
    a->genEpoch = 0;
    a->genConsumed = 1;
    a->genOrder = 0;
    a->appOrder = 0;
    a->lastPresented = 0;
    a->lastCurSlot = -1;
    a->lastOrder = -1;
    a->hasPrev = false;
    a->jobCount = 0;
    for (auto &s : a->slots) s.order = 0;
    a->nextFree = 0;
    a->broken = false;
}

static void async_teardown(AsyncCtx *a) {
    {
        std::lock_guard<std::mutex> lk(a->qmtx);
        a->stop = true;
    }
    a->qcv.notify_all();
    a->ccv.notify_all();
    if (a->worker.joinable()) a->worker.join();
    a->stop = false;
    async_free_egl_gl(a);
}

// 主 surface 跟踪：多 surface 应用（bilibili 弹幕/浮层、抖音多窗）只对主画面插帧，
// 其他 surface 原样交换——避免弹幕/浮层动画被插帧（黑长条/异常）或触发初始化失败风暴
static EGLContext s_main_ctx = EGL_NO_CONTEXT;
static EGLSurface s_main_surf = EGL_NO_SURFACE;
static int s_main_w = 0, s_main_h = 0;
static std::chrono::steady_clock::time_point s_main_last_swap{};

// 游戏模式异步插帧（在 hook 内、app 上下文 current 时调用）
static void async_game_path(EGLDisplay dpy, EGLSurface surf, int w, int h) {
    EGLContext ctx = eglGetCurrentContext();
    if (ctx == EGL_NO_CONTEXT) return;

    // 主 surface 判定：首个 surface 为主；更大尺寸的 surface 接管；主 surface 停止交换
    // 超 2 秒（视频切换/销毁）后允许新 surface 接管。其余 surface 一律原样交换。
    {
        auto now = std::chrono::steady_clock::now();
        if (s_main_surf == EGL_NO_SURFACE) {
            s_main_ctx = ctx; s_main_surf = surf; s_main_w = w; s_main_h = h; s_main_last_swap = now;
        } else if (s_main_ctx != ctx || s_main_surf != surf) {
            bool stale = std::chrono::duration_cast<std::chrono::milliseconds>(now - s_main_last_swap).count() > 2000;
            if ((long)w * h > (long)s_main_w * s_main_h || stale) {
                s_main_ctx = ctx; s_main_surf = surf; s_main_w = w; s_main_h = h;
            } else {
                return;   // 非主 surface（弹幕/浮层/小窗）：原样交换
            }
        }
        s_main_last_swap = now;
    }

    // 按 context 取 3 套资源池之一（LRU），失败/熔断则原样交换
    AsyncCtx *a = mcfi_async_acquire(ctx, w, h);
    if (!a) return;

    McfiConfig cfg;
    {
        std::lock_guard<std::mutex> lk(g_cfg_mtx);
        cfg = g_cfg;
    }

    static bool g_gles_reported = false;
    if (!g_gles_reported) {
        g_gles_reported = true;
        mcfi_send_event("%s(pid=%d): MCFI 补帧管线生效 %dx%d [GLES 异步 多尺度3DRS %s %s]",
                        g_pkg.c_str(), getpid(), w, h,
                        mcfi_algo_name(a->meOk ? cfg.interp_mode : 1),
                        a->meOk ? (gles_mediump_fp16() ? "fp16" : "fp32(mediump模拟)") : "普通混合");
    }

    // 找一个空闲槽；没有则丢弃本帧（防堆积）
    int freeSlot = -1;
    for (int k = 0; k < ASYNC_SLOTS; k++) {
        int i = (a->nextFree + k) % ASYNC_SLOTS;
        if (a->slots[i].state.load(std::memory_order_acquire) == 0) {
            freeSlot = i;
            break;
        }
    }
    if (freeSlot < 0) return;   // worker 繁忙，跳过本帧插帧
    a->nextFree = (freeSlot + 1) % ASYNC_SLOTS;
    a->slots[freeSlot].state = 1;
    a->appOrder++;
    a->slots[freeSlot].order = a->appOrder;

    // 1. 拷贝默认帧缓冲(真实新帧) -> 槽纹理
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, a->slots[freeSlot].fbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (a->slots[freeSlot].copySync) glDeleteSync(a->slots[freeSlot].copySync);
    a->slots[freeSlot].copySync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    {
        std::lock_guard<std::mutex> lk(a->qmtx);
        a->jobs.push(freeSlot);
    }
    a->qcv.notify_one();

    // 2. 若 worker 已合成好生成帧且 GPU 完成 → 先呈现生成帧，再呈现真实帧
    //    新鲜度守卫：生成帧对应的真实帧序号必须接近当前帧（appOrder - genOrder <= 2），
    //    过期的中间帧（worker 慢/丢帧时）丢弃不呈现——但必须消费（通知 worker 继续），
    //    避免 worker 永久阻塞；画面只显示真实帧，杜绝跳变闪烁。
    int epoch = a->genEpoch.load(std::memory_order_acquire);
    GLsync ds = a->doneSync;
    int genOrder = a->genOrder.load(std::memory_order_acquire);
    if (epoch != a->lastPresented && ds) {
        bool fresh = (a->appOrder - genOrder) <= 2;
        GLenum st = glClientWaitSync(ds, 0, 0);
        if (st == GL_ALREADY_SIGNALED || st == GL_CONDITION_SATISFIED) {
            a->lastPresented = epoch;
            if (fresh) {
                GLState stt;
                saveState(stt);
                // 画生成帧 -> 默认帧缓冲，先行呈现（时间戳注入：A.5 上屏于下一个 vsync）
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glViewport(0, 0, w, h);
                draw_texture(a->progCopy, a->vao, a->genTex);
                mcfi_set_pts(dpy, surf, 1);
                g_pts_pending = true;
                orig_eglSwapBuffers(dpy, surf);
                // 把真实帧画回默认帧缓冲（外层原始 Swap 呈现）
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glViewport(0, 0, w, h);
                draw_texture(a->progCopy, a->vao, a->slots[freeSlot].tex);
                restoreState(stt);
            }
            a->genConsumed = 1;
            a->ccv.notify_all();
        }
    }
}

// ============================================================
// GLES 视频同步插帧（独立于异步 worker）
// 视频模式在 swap 内同步计算：逻辑简单可控、无异步延迟/上下文切换问题；
// 按 EGLContext 缓存最多 3 套资源（LRU），跳过 <500 小 surface；
// 性能上限：插帧耗时超预算自动降档（减候选→隔帧→熔断停插），
// 避免同步计算拖垮渲染线程（历史教训：同步全速曾 19-20fps/GPU 占满）。
// ============================================================

struct VideoCtx {
    EGLDisplay dpy = EGL_NO_DISPLAY;
    EGLContext appCtx = EGL_NO_CONTEXT;
    GLuint progCopy = 0, progInterp = 0, progME[4] = {0, 0, 0, 0}, vao = 0;
    GLuint dsTex[4] = {0, 0, 0, 0}, dsFbo[4] = {0, 0, 0, 0};
    GLuint mvTex[3] = {0, 0, 0};     // mvA(L2) mvB(L1) mvC(L1 平滑/时间源)
    GLuint statsBuf = 0;
    GmvRes gmv;                      // GMV 直方图/输出
    GLuint prevTex = 0, prevFbo = 0, curTex = 0, curFbo = 0;  // 真实帧缓存（双槽）
    GLuint genTex = 0, genFbo = 0;
    int w = 0, h = 0, dsW = 0, dsH = 0, dsW2 = 0, dsH2 = 0;
    int mvW1 = 0, mvH1 = 0, mvW2 = 0, mvH2 = 0;
    bool meOk = false;
    bool broken = false;
    bool hasPrev = false;
    bool fresh = false;        // 本帧刚完成 init/重建：不计入性能自适应统计（重建耗时是初始化开销，不是插帧开销）
    int lastOrder = -1;
    int perf_level = 0;    // 0=满血 1=减候选(不 refine) 2=隔帧 3=熔断停插
    int perf_hits = 0;
};

static VideoCtx g_vpool[3];
static EGLContext g_vpool_ctx[3] = {EGL_NO_CONTEXT, EGL_NO_CONTEXT, EGL_NO_CONTEXT};
static int g_vpool_lru[3] = {0, 0, 0};
static int g_vlru_tick = 0;

static void video_free(VideoCtx *v) {
    if (v->progCopy) glDeleteProgram(v->progCopy);
    if (v->progInterp) glDeleteProgram(v->progInterp);
    for (int i = 0; i < 4; i++) if (v->progME[i]) glDeleteProgram(v->progME[i]);
    if (v->vao) glDeleteVertexArrays(1, &v->vao);
    if (v->statsBuf) glDeleteBuffers(1, &v->statsBuf);
    if (v->prevTex) glDeleteTextures(1, &v->prevTex);
    if (v->prevFbo) glDeleteFramebuffers(1, &v->prevFbo);
    if (v->curTex) glDeleteTextures(1, &v->curTex);
    if (v->curFbo) glDeleteFramebuffers(1, &v->curFbo);
    if (v->genTex) glDeleteTextures(1, &v->genTex);
    if (v->genFbo) glDeleteFramebuffers(1, &v->genFbo);
    if (v->dsTex[0]) glDeleteTextures(4, v->dsTex);
    if (v->dsFbo[0]) glDeleteFramebuffers(4, v->dsFbo);
    if (v->mvTex[0]) glDeleteTextures(3, v->mvTex);
    gmv_free(v->gmv);
    *v = VideoCtx{};
}

static bool video_init(VideoCtx *v, EGLContext ctx, int w, int h) {
    v->dpy = eglGetCurrentDisplay();
    v->appCtx = ctx;
    v->w = w; v->h = h;
    v->dsW = (w + 3) / 4; v->dsH = (h + 3) / 4;
    // 视频模式 L2=1/8（比游戏的 1/16 高一层，快速运动粗搜索更准）；游戏异步路径仍 1/16
    v->dsW2 = (v->dsW + 1) / 2; v->dsH2 = (v->dsH + 1) / 2;
    v->mvW1 = (v->dsW + 7) / 8; v->mvH1 = (v->dsH + 7) / 8;
    v->mvW2 = (v->dsW2 + 7) / 8; v->mvH2 = (v->dsH2 + 7) / 8;
    v->progCopy = mcfi_link_program(VS_SRC, FS_COPY_SRC);
    v->progInterp = mcfi_link_program(VS_SRC, FS_INTERP_SRC);
    v->meOk = gles3_compute_ok();
    if (v->meOk) {
        v->progME[0] = mcfi_link_compute(CS_ME0_SRC);
        v->progME[1] = link_me1();
        v->progME[2] = mcfi_link_compute(CS_ME2_SRC);
        v->progME[3] = mcfi_link_compute(CS_ME3_SRC);
        if (!v->progME[0] || !v->progME[1] || !v->progME[2] || !v->progME[3]) {
            LOGE("GLES 视频 compute 链接失败，运动补偿降级为普通混合");
            v->meOk = false;
        }
    }
    glGenVertexArrays(1, &v->vao);
    make_tex(v->prevTex, w, h); make_fbo(v->prevFbo, v->prevTex);
    make_tex(v->curTex, w, h);  make_fbo(v->curFbo, v->curTex);
    make_tex(v->genTex, w, h);  make_fbo(v->genFbo, v->genTex);
    glGenTextures(4, v->dsTex);
    glGenFramebuffers(4, v->dsFbo);
    for (int i = 0; i < 4; i++) {
        make_tex(v->dsTex[i], (i < 2) ? v->dsW : v->dsW2, (i < 2) ? v->dsH : v->dsH2);
        make_fbo(v->dsFbo[i], v->dsTex[i]);
    }
    make_tex(v->mvTex[0], v->mvW2, v->mvH2, true);
    make_tex(v->mvTex[1], v->mvW1, v->mvH1, true);
    make_tex(v->mvTex[2], v->mvW1, v->mvH1, true);
    {
        std::vector<float> mv0L2((size_t)v->mvW2 * v->mvH2 * 4, 0.5f);
        for (size_t i = 3; i < mv0L2.size(); i += 4) mv0L2[i] = 1.0f;
        std::vector<float> mv0L1((size_t)v->mvW1 * v->mvH1 * 4, 0.5f);
        for (size_t i = 3; i < mv0L1.size(); i += 4) mv0L1[i] = 1.0f;
        glBindTexture(GL_TEXTURE_2D, v->mvTex[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, v->mvW2, v->mvH2, GL_RGBA, GL_FLOAT, mv0L2.data());
        for (int i = 1; i < 3; i++) {
            glBindTexture(GL_TEXTURE_2D, v->mvTex[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, v->mvW1, v->mvH1, GL_RGBA, GL_FLOAT, mv0L1.data());
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glGenBuffers(1, &v->statsBuf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, v->statsBuf);
    uint32_t zero = 0;
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(zero), &zero, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if (v->meOk && !gmv_init(v->gmv)) v->meOk = false;
    if (!v->progCopy || !v->progInterp) {
        LOGE("GLES 视频程序创建失败");
        video_free(v);
        return false;
    }
    v->hasPrev = false;
    v->lastOrder = -1;
    v->perf_level = 0;
    v->perf_hits = 0;
    LOGI("GLES 能力探测: fp16=%s subgroup=%s",
         gles_mediump_fp16() ? "是" : "否", gles_subgroup_ok() ? "支持(已接入ME1归约)" : "不支持");
    return true;
}

static VideoCtx *video_acquire(EGLContext ctx, int w, int h) {
    for (int i = 0; i < 3; i++) {
        if (g_vpool_ctx[i] == ctx) {
            g_vpool_lru[i] = ++g_vlru_tick;
            VideoCtx *v = &g_vpool[i];
            if (!v->progCopy || v->appCtx != ctx || v->w != w || v->h != h) {
                if (v->broken) return nullptr;
                video_free(v);
                if (!video_init(v, ctx, w, h)) { v->broken = true; return nullptr; }
                v->fresh = true;
            }
            return v;
        }
    }
    int victim = 0;
    for (int i = 1; i < 3; i++) if (g_vpool_lru[i] < g_vpool_lru[victim]) victim = i;
    VideoCtx *v = &g_vpool[victim];
    video_free(v);
    g_vpool_ctx[victim] = ctx;
    g_vpool_lru[victim] = ++g_vlru_tick;
    if (!video_init(v, ctx, w, h)) { v->broken = true; return nullptr; }
    v->fresh = true;
    return v;
}

// 视频同步插帧（hook 内、app 上下文 current 时调用）：
// 当前帧 → curTex；有 prev 时按 video_interval 同步做多尺度 3DRS+合成 → 先呈现中间帧（orig swap）再恢复真实帧；
// 交换 prev/cur 槽（不插帧的帧也保持相邻关系）。性能上限：同步耗时 ≥16ms 累计 3 次 → 升档（1 减候选 / 2 隔帧 / 3 熔断）。
static void video_interp_sync(EGLDisplay dpy, EGLSurface surf, int w, int h, bool is_video) {
    EGLContext ctx = eglGetCurrentContext();
    if (ctx == EGL_NO_CONTEXT) return;

    // 主 surface 判定（与异步共用 s_main_* 跟踪）：首个/更大 surface 为主，其余原样交换
    {
        auto now = std::chrono::steady_clock::now();
        if (s_main_surf == EGL_NO_SURFACE) {
            s_main_ctx = ctx; s_main_surf = surf; s_main_w = w; s_main_h = h; s_main_last_swap = now;
        } else if (s_main_ctx != ctx || s_main_surf != surf) {
            bool stale = std::chrono::duration_cast<std::chrono::milliseconds>(now - s_main_last_swap).count() > 2000;
            if ((long)w * h > (long)s_main_w * s_main_h || stale) {
                s_main_ctx = ctx; s_main_surf = surf; s_main_w = w; s_main_h = h;
            } else {
                return;
            }
        }
        s_main_last_swap = now;
    }

    VideoCtx *v = video_acquire(ctx, w, h);
    if (!v || v->broken) return;
    if (v->perf_level >= 3) return;                          // 熔断：停插

    McfiConfig cfg;
    {
        std::lock_guard<std::mutex> lk(g_cfg_mtx);
        cfg = g_cfg;
    }

    static bool g_video_reported = false;
    if (!g_video_reported) {
        g_video_reported = true;
        int eff_mode = is_video ? cfg.video_interp_mode : cfg.interp_mode;
        mcfi_send_event("%s(pid=%d): MCFI %s补帧管线生效 %dx%d [GLES 同步 多尺度3DRS %s %s]",
                        g_pkg.c_str(), getpid(), is_video ? "视频" : "游戏", w, h,
                        mcfi_algo_name(v->meOk ? eff_mode : 1),
                        v->meOk ? (gles_mediump_fp16() ? "fp16" : "fp32(mediump模拟)") : "普通混合");
    }

    auto t0 = std::chrono::steady_clock::now();
    GLState st;
    saveState(st);

    // 视频插帧间隔（video_interval）：每 N 个真实帧插入 1 个生成帧（N=1 即每帧插，帧率翻倍）；
    // 降档2（隔帧）强制 N=2。不插帧的帧仍拷贝真实帧并交换 prev/cur（保持相邻帧关系），
    // 外层 swap 呈现真实帧——否则间隔帧会跨大间隔合成（prev/cur 拉大，画面跳变闪烁）。
    int gi = cfg.video_interval > 0 ? cfg.video_interval : 1;
    if (v->perf_level >= 2) gi = 2;
    bool do_insert = v->hasPrev && (v->lastOrder % gi) == 0;

    if (!do_insert) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, v->curFbo);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        std::swap(v->prevTex, v->curTex);
        std::swap(v->prevFbo, v->curFbo);
        v->hasPrev = true;
        v->lastOrder++;
        restoreState(st);
        return;
    }

    // 1. 拷贝默认帧缓冲（真实新帧）→ curTex
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, v->curFbo);
    glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    int range = cfg.me_range();
    bool refine = cfg.me_refine() && v->meOk && (v->perf_level < 1);

    // 2. 有上一帧时同步插帧
    if (v->hasPrev) {
        int imode = is_video ? cfg.video_interp_mode : cfg.interp_mode;  // 视频/游戏分别选算法
        if (v->meOk) {
            GLuint mvOut = run_motion_estimation(v->progME, v->progCopy, v->vao,
                                                 v->prevTex, v->curTex,
                                                 v->dsTex, v->dsFbo,
                                                 v->mvTex[2], v->mvTex[0], v->mvTex[1],
                                                 w, h, v->dsW, v->dsH, v->dsW2, v->dsH2,
                                                 v->mvW1, v->mvH1, v->mvW2, v->mvH2,
                                                 range, refine, cfg.smooth / 100.0f, &v->gmv, 2);
            float lowRatio = run_motion_smooth_stats(v->progME, mvOut, v->mvTex[2],
                                                     v->statsBuf, v->mvW1, v->mvH1);
            if (lowRatio > 0.40f) imode = 1;
            run_mci_pass(v->progInterp, v->vao,
                         v->prevTex, v->curTex, v->mvTex[2], v->genFbo,
                         w, h, v->dsW, v->dsH,
                         imode, cfg.strength / 100.0f, range, (float)cfg.smooth);
        } else {
            run_mci_pass(v->progInterp, v->vao,
                         v->prevTex, v->curTex, v->mvTex[0], v->genFbo,
                         w, h, v->dsW, v->dsH,
                         1, cfg.strength / 100.0f, range, (float)cfg.smooth);
        }
        // 呈现中间帧（时间戳注入：A.5 上屏于下一个 vsync）
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        draw_texture(v->progCopy, v->vao, v->genTex);
        mcfi_set_pts(dpy, surf, 1);
        g_pts_pending = true;
        orig_eglSwapBuffers(dpy, surf);
        // 真实帧画回默认帧缓冲（外层原始 Swap 呈现）
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        draw_texture(v->progCopy, v->vao, v->curTex);
    }

    // 3. cur → prev 交换
    std::swap(v->prevTex, v->curTex);
    std::swap(v->prevFbo, v->curFbo);
    v->hasPrev = true;
    v->lastOrder++;

    restoreState(st);

    // 4. 性能上限（同步耗时监控）：≥25ms 累计 10 次升一档（大幅放宽阈值，视频不易误触发）。
    //    刚 init/重建的帧跳过统计（初始化是资源开销不是插帧开销，切视频瞬间不误升档）。
    if (v->fresh) { v->fresh = false; v->perf_hits = 0; }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (ms >= 25) {
        v->perf_hits++;
        if (v->perf_hits >= 10) {
            v->perf_hits = 0;
            if (v->perf_level < 3) {
                v->perf_level++;
                mcfi_send_event("视频同步插帧性能自适应: 升档%d（%s）",
                                v->perf_level,
                                v->perf_level == 1 ? "取消细化" :
                                v->perf_level == 2 ? "隔帧插帧" : "熔断停插");
            }
        }
    } else {
        v->perf_hits = 0;
    }
}

// ---------------- Hook ----------------

static EGLBoolean my_eglSwapBuffers(EGLDisplay dpy, EGLSurface surf) {
    static bool g_gles_first = false;
    if (!g_gles_first) {
        g_gles_first = true;
        int w = 0, h = 0;
        eglQuerySurface(dpy, surf, EGL_WIDTH, &w);
        eglQuerySurface(dpy, surf, EGL_HEIGHT, &h);
        LOGI("%s(pid=%d): GLES 接口 eglSwapBuffers 首次命中 %dx%d",
             g_pkg.c_str(), getpid(), w, h);
        mcfi_send_event("%s(pid=%d): GLES 接口 eglSwapBuffers 首次命中 %dx%d",
                        g_pkg.c_str(), getpid(), w, h);
    }
    {
        McfiConfig cfg;
        {
            std::lock_guard<std::mutex> lk(g_cfg_mtx);
            cfg = g_cfg;
        }
        if (cfg.enabled && orig_eglSwapBuffers) {
            std::lock_guard<std::mutex> lk(g_gl_mtx);

            // 周期性热更新配置（约每 120 次交换一次）
            static long g_swap_count = 0;
            g_swap_count++;
            if ((g_swap_count % 120) == 0 && g_companion_fd >= 0) {
                std::string text = mcfi_request_config(g_companion_fd);
                if (!text.empty()) {
                    McfiConfig nc = McfiConfig::parse(text);
                    {
                        std::lock_guard<std::mutex> lk2(g_cfg_mtx);
                        if (nc.enabled != g_cfg.enabled || nc.interp_mode != g_cfg.interp_mode ||
                            nc.strength != g_cfg.strength || nc.game_interval != g_cfg.game_interval ||
                            nc.video_interval != g_cfg.video_interval || nc.me_quality != g_cfg.me_quality) {
                            LOGI("配置热更新: enabled=%d 算法=%s strength=%d 游戏间隔=%d 视频间隔=%d me=%d",
                                 nc.enabled, mcfi_algo_name(nc.interp_mode),
                                 nc.strength, nc.game_interval, nc.video_interval, nc.me_quality);
                        }
                        g_cfg = nc;
                    }
                    cfg = nc;
                }
            }

            if (cfg.enabled) {
                EGLint w = 0, h = 0;
                eglQuerySurface(dpy, surf, EGL_WIDTH, &w);
                eglQuerySurface(dpy, surf, EGL_HEIGHT, &h);
                if (w <= 0 || h <= 0) return orig_eglSwapBuffers(dpy, surf);

                if (is_video_mode()) {
                    // 视频模式：跳过过小 surface（弹幕/小窗/浮层），只对主画面插帧
                    if (w < 500) return orig_eglSwapBuffers(dpy, surf);
                    // 视频模式独立同步计算（内联呈现中间帧后恢复真实帧，外层再呈现真实帧）
                    video_interp_sync(dpy, surf, w, h, true);
                    return mcfi_swap_real(dpy, surf);
                }
                // 游戏模式异步流水线：app 线程只拷帧 + 呈现生成帧，运动估计在 worker 线程
                async_game_path(dpy, surf, w, h);
            }
        }
    }
    return mcfi_swap_real(dpy, surf);
}

// ---------------- 配置访问接口（供 Vulkan 后端调用） ----------------

extern "C" void mcfi_get_config(McfiConfig *out) {
    std::lock_guard<std::mutex> lk(g_cfg_mtx);
    *out = g_cfg;
}

extern "C" void mcfi_refresh_config() {
    if (g_companion_fd < 0) return;
    std::string text = mcfi_request_config(g_companion_fd);
    if (!text.empty()) {
        std::lock_guard<std::mutex> lk(g_cfg_mtx);
        g_cfg = McfiConfig::parse(text);
    }
}

// ---------------- 初始化入口（由 zygisk entry 调用） ----------------

extern "C" void mcfi_set_pkg(const char *pkg) {
    g_pkg = pkg ? pkg : "";
}

extern "C" void mcfi_payload_init(const char *config_text, int companion_fd) {
    {
        std::lock_guard<std::mutex> lk(g_cfg_mtx);
        g_cfg = McfiConfig::parse(config_text ? config_text : "");
    }
    g_companion_fd = companion_fd;

    McfiConfig cfg;
    mcfi_get_config(&cfg);

    // 根据 per-package 后端指定决定挂哪些 hook
    // B_DEFAULT(0): 两个都装；B_GLES(1): 只 GLES；B_VULKAN(2): 只 Vulkan；
    // B_VIDEO(3): GLES(视频同步) + Vulkan(视频异步) 都装——很多视频 App 走 Vulkan/解码直出，
    //             只挂 GLES 会完全不插帧；哪个渲染就走哪个，Vulkan 侧已有小窗守卫。
    bool want_vk = (g_backend == 0 || g_backend == 2 || g_backend == 3);
    bool want_gles = (g_backend == 0 || g_backend == 1 || g_backend == 3);

    if (want_vk) {
        mcfi_vk_install(cfg.log_level, g_backend);
    } else {
        LOGI("%s(pid=%d): 按配置跳过 Vulkan hook", g_pkg.c_str(), getpid());
    }

    if (!want_gles) {
        LOGI("%s(pid=%d): 按配置跳过 GLES hook", g_pkg.c_str(), getpid());
        mcfi_send_event("%s(pid=%d): 进程已注入（后端=vulkan，仅 Vulkan 生效）",
                        g_pkg.c_str(), getpid());
        return;
    }

    void *egl = dlopen("libEGL.so", RTLD_NOW);
    if (!egl) egl = dlopen("libEGL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!egl) {
        LOGE("dlopen libEGL.so 失败");
        return;
    }
    void *sym = dlsym(egl, "eglSwapBuffers");
    if (!sym) {
        LOGE("dlsym eglSwapBuffers 失败");
        return;
    }
    // 时间戳注入扩展（EGL_ANDROID_presentation_time）：加载失败则静默退回原行为
    g_eglPts = (PFNEGLPRESENTATIONTIMEANDROIDPROC)dlsym(egl, "eglPresentationTimeANDROID");
    LOGI("eglPresentationTimeANDROID 时间戳注入: %s", g_eglPts ? "可用" : "不可用(退回原行为)");
    if (DobbyHook(sym, (void *)my_eglSwapBuffers, (void **)&orig_eglSwapBuffers) != 0) {
        LOGE("DobbyHook eglSwapBuffers 失败");
        return;
    }
    LOGI("eglSwapBuffers hook 安装成功 (pid=%d)", getpid());
    LOGI("%s(pid=%d): 进程已注入，等待首次渲染调用", g_pkg.c_str(), getpid());
    mcfi_send_event("%s(pid=%d): 进程已注入（后端=%s），等待首次渲染调用",
                    g_pkg.c_str(), getpid(),
                    is_video_mode() ? "video(gles)" : "gles");
}
