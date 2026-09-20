// MCFI Vulkan 后端（运动补偿插帧）
//
// 原理：
//   Hook libvulkan.so 的全局入口（vkGetInstanceProcAddr 一处），跟踪设备与交换链，
//   拦截 vkQueuePresentKHR：
//   1. 游戏提交真实帧 N+1 呈现前，把它拷贝到私有纹理 copyCur
//   2. 若满足插帧条件：自行 acquire 一张空闲交换链图像，用图形管线把 (N, N+1) 的中间帧画进去，
//      提交并先行呈现（插入生成帧）
//   3. 再调用原始 vkQueuePresentKHR 呈现真实帧
//
// 运动补偿：
//   插帧管线新增 compute 阶段：把 prev/cur 降采样到 1/4（vkCmdBlitImage），块匹配 SAD
//   运动估计（含可选细化）输出运动矢量场（R8G8B8A8_UNORM storage image），
//   fragment 着色器按矢量双向 warp 合成中间帧（带置信度回退与遮挡/拖影抑制）。
//
// 稳定性要点（务必保持）：
//   - 交换链图像拷贝后必须 barrier 回 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR 再交给呈现引擎；
//   - 所有 GPU 操作在异步 worker 线程执行，present 回调只 push 任务并丢弃积压，绝不阻塞游戏线程；
//   - 生成帧的 submit/present 与游戏 submit 同 queue 隐式串行，额外 acquire 必须等 acquire_sem；
//   - 若资源创建/功能检查失败则只降级为普通混合，绝不改动游戏自身的呈现路径。

#include <vulkan/vulkan.h>
#include <dobby.h>
#include <dlfcn.h>
#include <android/log.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <pthread.h>
#include <mutex>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <string>
#include <cstring>
#include <cstdio>
#include <ctime>

#include "config.h"

#define VKLOG_TAG "MCFI"
#define VKLOGI(...) do { if (g_log_level > 0) __android_log_print(ANDROID_LOG_INFO, VKLOG_TAG, __VA_ARGS__); } while (0)
#define VKLOGE(...) __android_log_print(ANDROID_LOG_ERROR, VKLOG_TAG, __VA_ARGS__)

// 由 payload.cpp 提供
extern "C" void mcfi_get_config(McfiConfig *out);
extern "C" void mcfi_refresh_config();
extern "C" void mcfi_send_event(const char *msg);

static int g_log_level = 1;
static int g_backend_vk = 0;   // McfiConfig::Backend（由 payload_init 传入，用于视频小窗守卫）

#include "shaders/vert_spv.h"
#include "shaders/mci_spv.h"
#include "shaders/me0_spv.h"
#include "shaders/me1_spv.h"
#include "shaders/me0_f16_spv.h"
#include "shaders/me1_f16_spv.h"
#include "shaders/gmv0_spv.h"
#include "shaders/gmv1_spv.h"
#include "shaders/me0_l2_spv.h"
#include "shaders/me1_l2_spv.h"
#include "shaders/me1_sub_spv.h"
#include "shaders/me0_l2_f16_spv.h"
#include "shaders/me1_l2_f16_spv.h"

// ---------------- 真实入口 ----------------

static void *g_vk_lib = nullptr;
static PFN_vkGetDeviceProcAddr g_GetDeviceProcAddr = nullptr;      // dlsym 原始地址（已被 inline hook）
static PFN_vkCreateDevice g_CreateDevice = nullptr;
static PFN_vkGetInstanceProcAddr g_GetInstanceProcAddr = nullptr;
static PFN_vkGetPhysicalDeviceMemoryProperties g_GetPhysDevMemProps = nullptr;
static PFN_vkGetPhysicalDeviceFormatProperties g_GetPhysDevFmtProps = nullptr;
static PFN_vkCreateSwapchainKHR g_CreateSwapchainKHR = nullptr;
static PFN_vkDestroySwapchainKHR g_DestroySwapchainKHR = nullptr;
static PFN_vkQueuePresentKHR g_QueuePresentKHR = nullptr;
static PFN_vkGetDeviceQueue g_GetDeviceQueue = nullptr;
// trampoline：Dobby 保存的原始指令副本，包装函数内部必须调这些（而不是上面的原始地址，
// 因为原始地址的入口已被 inline hook 改写，直接调会跳回包装函数导致无限递归）
static PFN_vkCreateDevice g_OrigCreateDevice = nullptr;
static PFN_vkGetDeviceProcAddr g_OrigGetDeviceProcAddr = nullptr;
static VkInstance g_inst = VK_NULL_HANDLE;      // 记录 instance（my_GetInstanceProcAddr 里捕获）
static bool g_vk_f16 = false;                    // 设备已启用 VK_KHR_shader_float16_int8（fp16 计算可用）
static bool g_vk_subgroup = false;               // 设备 supportedOperations 含 ARITHMETIC_BIT（subgroup 归约可用）
static bool g_vk_cache_ctrl = false;             // 设备已启用 VK_EXT_pipeline_creation_cache_control
static bool g_vk_sgsc = false;                   // VK_EXT_subgroup_size_control 已启用（可钉 subgroup size）
static bool g_vk_sget = false;                   // VK_KHR_shader_subgroup_extended_types 已启用（fp16 subgroup 归约）
static bool g_vk_ts = false;                     // VK_KHR_timeline_semaphore 已启用（槽位 CPU 同步）
static uint32_t g_vk_sg_min = 0, g_vk_sg_max = 0; // 设备支持的 subgroupSize 范围
static std::string g_vk_pkg;                     // 命中包名（由 mcfi_vk_install 传入，用于管线缓存落点）
static PFN_vkGetInstanceProcAddr g_OrigGetInstanceProcAddr = nullptr;

// ---------------- VkPipelineCache 持久化 ----------------
// 进程内 VkPipelineCache 本身即可让"同进程重建交换链"（旋转/切画质）跳过 10 条管线的重编译；
// 附带尝试落盘到 app 私有 files 目录实现跨启动缓存，任何一步失败都静默退回（不影响功能）。
static std::string vk_cache_path() {
    if (g_vk_pkg.empty()) return "";
    return "/data/user/0/" + g_vk_pkg + "/files/mcfi_vk_pipeline.cache";
}
static std::vector<uint8_t> vk_cache_read() {
    std::vector<uint8_t> out;
    std::string p = vk_cache_path();
    if (p.empty()) return out;
    int fd = open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return out;
    char buf[16384];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) out.insert(out.end(), buf, buf + n);
    close(fd);
    return out;
}
static void vk_cache_write(const uint8_t *data, size_t size) {
    std::string p = vk_cache_path();
    if (p.empty() || !data || size == 0) return;
    std::string dir = "/data/user/0/" + g_vk_pkg + "/files";
    mkdir(dir.c_str(), 0700);   // 目录通常已存在（app 私有 files）；失败也无妨
    int fd = open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return;
    size_t off = 0;
    while (off < size) {
        ssize_t n = write(fd, data + off, size - off);
        if (n <= 0) break;
        off += (size_t)n;
    }
    close(fd);
}

// ---------------- SurfaceFlinger 时间戳注入（VK_GOOGLE_display_timing） ----------------
// 让生成帧 A.5 期望上屏于 +1 vsync、真实帧 B 于 +2 vsync，各占一个完整周期。
// 扩展不可用 → g_vk_pts=false，present 不挂 pNext，行为与原版一致。
#define MCFI_PRESENT_TIMES_INFO_GOOGLE 1000093000
typedef struct mcfi_VkPresentTimeGOOGLE {
    uint32_t presentID;
    uint64_t desiredPresentTime;
    uint64_t actualPresentTime;
    uint64_t earliestPresentTime;
    uint64_t presentMargin;
} mcfi_VkPresentTimeGOOGLE;
typedef struct mcfi_VkPresentTimesInfoGOOGLE {
    VkStructureType sType;
    const void* pNext;
    uint32_t swapchainCount;
    const mcfi_VkPresentTimeGOOGLE* pTimes;
} mcfi_VkPresentTimesInfoGOOGLE;

static bool g_vk_pts = false;                     // 设备已启用 VK_GOOGLE_display_timing
static std::atomic<int64_t> g_vk_vsync_ns{8333333};
static std::atomic<uint64_t> g_vk_gen_seq{0};     // worker 已 present 的生成帧计数
static int64_t g_vk_last_real_ns = 0;

// 时间戳对齐 vsync 网格：向上取整到下一边界再偏移 (k-1) 个周期
static int64_t mcfi_vk_align_pts(int64_t now_ns, int64_t period_ns, int k) {
    int64_t next_vsync = ((now_ns + period_ns - 1) / period_ns) * period_ns;
    return next_vsync + (int64_t)(k - 1) * period_ns;
}
static int64_t mcfi_vk_now_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
static void mcfi_vk_vsync_sample(int64_t now_ns) {
    // vsync_hz>0：手动指定刷新率，跳过自动估计（与 GLES 侧同一配置项）
    McfiConfig vkc;
    mcfi_get_config(&vkc);
    if (vkc.vsync_hz > 0) {
        g_vk_vsync_ns.store(1000000000LL / vkc.vsync_hz);
        return;
    }
    if (g_vk_last_real_ns > 0) {
        int64_t d = now_ns - g_vk_last_real_ns;
        if (d >= 3000000 && d <= 20000000) {
            int64_t cur = g_vk_vsync_ns.load();
            g_vk_vsync_ns.store((cur * 7 + d) / 8);
        }
    }
    g_vk_last_real_ns = now_ns;
}

// ---------------- 设备函数表 ----------------

struct DevFns {
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkResetCommandBuffer ResetCommandBuffer;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2;
    PFN_vkCmdCopyImage CmdCopyImage;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDraw CmdDraw;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueuePresentKHR QueuePresentKHR;
    PFN_vkAcquireNextImageKHR AcquireNextImageKHR;
    PFN_vkCreateSemaphore CreateSemaphore;
    PFN_vkDestroySemaphore DestroySemaphore;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkResetFences ResetFences;
    PFN_vkGetSemaphoreCounterValue GetSemaphoreCounterValue;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkCmdFillBuffer CmdFillBuffer;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkGetImageMemoryRequirements2 GetImageMemoryRequirements2;
    PFN_vkGetBufferMemoryRequirements2 GetBufferMemoryRequirements2;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkBindImageMemory2 BindImageMemory2;
    PFN_vkBindBufferMemory2 BindBufferMemory2;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkDestroyRenderPass DestroyRenderPass;
    PFN_vkCreateFramebuffer CreateFramebuffer;
    PFN_vkDestroyFramebuffer DestroyFramebuffer;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreatePipelineCache CreatePipelineCache;
    PFN_vkDestroyPipelineCache DestroyPipelineCache;
    PFN_vkGetPipelineCacheData GetPipelineCacheData;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateSampler CreateSampler;
    PFN_vkDestroySampler DestroySampler;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkDestroyDevice DestroyDevice;
};

struct DeviceCtx {
    VkDevice dev = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    DevFns f{};
    VkPipelineCache cache = VK_NULL_HANDLE;     // 进程内管线缓存（同进程重建交换链时复用编译结果）
    bool sync2 = false;                          // 设备已启用 VK_KHR_synchronization2（barrier2 路径）
    std::map<VkQueue, uint32_t> queue_family;   // queue -> family index
    VkCommandPool cmd_pool = VK_NULL_HANDLE;    // 惰性创建（present queue 的 family）
    uint32_t cmd_pool_family = UINT32_MAX;
};

struct PresentSlot {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquire_sem = VK_NULL_HANDLE;
    VkSemaphore done_sem = VK_NULL_HANDLE;
    VkSemaphore done_ts = VK_NULL_HANDLE;   // timeline 槽（ts 路径 CPU 同步）
    VkFence fence = VK_NULL_HANDLE;         // 回退路径
    uint64_t ts_value = 0;                  // 已提交的 signal 值
};

struct SwapchainCtx {
    DeviceCtx *dc = nullptr;
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{0, 0};
    VkImageUsageFlags usage = 0;
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // ds[k]: prev=copy[k], cur=copy[1-k], mv
    VkSampler sampler = VK_NULL_HANDLE;
    VkImage copy_img[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory copy_mem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView copy_view[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool copy_layout_init[2] = {false, false};

    // ---- 运动补偿资源（1/4 降采样 + 双缓冲运动矢量场 + 双 pass compute 管线） ----
    // 双 pass：pass0 全搜索只写 mvA（writeonly）；pass1 以 sampler 读 mvA、只写 mvB（writeonly）。
    // 与 GLES 同构，绕开该驱动 image 混合读写崩溃/编译失败的问题。
    VkImage ds_img[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory ds_mem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView ds_view[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool ds_layout_init[2] = {false, false};
    VkImage mv_img = VK_NULL_HANDLE;          // mvA：pass0 写（storage），pass1 读（sampler）
    VkDeviceMemory mv_mem = VK_NULL_HANDLE;
    VkImageView mv_view = VK_NULL_HANDLE;
    VkImage mv_imgB = VK_NULL_HANDLE;         // mvB：pass1 写（storage），插帧 frag 读（sampler）
    VkDeviceMemory mv_memB = VK_NULL_HANDLE;
    VkImageView mv_viewB = VK_NULL_HANDLE;
    VkShaderModule cs0 = VK_NULL_HANDLE, cs1 = VK_NULL_HANDLE;
    VkShaderModule cs0f = VK_NULL_HANDLE, cs1f = VK_NULL_HANDLE;  // fp16 版（设备支持时创建）
    VkDescriptorSetLayout cs0_ds_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout cs1_ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool cs0_ds_pool = VK_NULL_HANDLE;
    VkDescriptorPool cs1_ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet cs0_ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE}; // cs0_ds[k]: prev=ds[1-k], cur=ds[k], mvA(storage)
    VkDescriptorSet cs1_ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE}; // cs1_ds[k]: prev, cur, mvA(sampler), mvB(storage)
    VkPipelineLayout cs0_pl = VK_NULL_HANDLE;
    VkPipelineLayout cs1_pl = VK_NULL_HANDLE;
    // GMV（全局运动向量）：统计两 pass（直方图/归约）+ 1×1 输出图 + 直方图 buffer
    VkImage gmv_img = VK_NULL_HANDLE; VkDeviceMemory gmv_mem = VK_NULL_HANDLE; VkImageView gmv_view = VK_NULL_HANDLE;
    bool gmv_layout_init = false;
    VkBuffer hist_buf = VK_NULL_HANDLE; VkDeviceMemory hist_mem = VK_NULL_HANDLE;
    VkShaderModule cs_gmv0 = VK_NULL_HANDLE, cs_gmv1 = VK_NULL_HANDLE;
    VkDescriptorSetLayout gmv0_ds_layout = VK_NULL_HANDLE, gmv1_ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool gmv0_ds_pool = VK_NULL_HANDLE, gmv1_ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet gmv0_ds = VK_NULL_HANDLE, gmv1_ds = VK_NULL_HANDLE;
    VkPipelineLayout gmv0_pl = VK_NULL_HANDLE, gmv1_pl = VK_NULL_HANDLE;
    VkPipeline gmv0_pipeline = VK_NULL_HANDLE, gmv1_pipeline = VK_NULL_HANDLE;
    VkPipeline cs0_pipeline = VK_NULL_HANDLE;
    VkPipeline cs1_pipeline = VK_NULL_HANDLE;
    // ---- L2 多尺度（1/8）----
    VkImage ds_img_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory ds_mem_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView ds_view_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImage mv_img_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory mv_mem_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView mv_view_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    int dsW2 = 0, dsH2 = 0, mvW2 = 0, mvH2 = 0;
    bool l2_ok = false;
    bool subgroup_ok = false;
    bool l2_layout_init[2] = {false, false};
    VkShaderModule cs0_l2 = VK_NULL_HANDLE, cs1_l2 = VK_NULL_HANDLE, cs1_sub = VK_NULL_HANDLE;
    VkShaderModule cs0_l2f = VK_NULL_HANDLE, cs1_l2f = VK_NULL_HANDLE;  // L2 fp16 版
    VkDescriptorSetLayout cs0_l2_ds_layout = VK_NULL_HANDLE, cs1_l2_ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool cs0_l2_ds_pool = VK_NULL_HANDLE, cs1_l2_ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet cs0_l2_ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet cs1_l2_ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout cs0_l2_pl = VK_NULL_HANDLE, cs1_l2_pl = VK_NULL_HANDLE;
    VkPipeline cs0_l2_pipeline = VK_NULL_HANDLE, cs1_l2_pipeline = VK_NULL_HANDLE;
    int dsW = 0, dsH = 0, mvW = 0, mvH = 0;
    bool mci_ok = false;   // 降采样/存储能力检查通过才启用 MCI
    bool mv_b_layout_init = false;  // mvB 首次布局初始化（UNDEFINED→SHADER_READ_ONLY，worker 独占）
    bool mv_a_ro = false;           // mvA 当前处于 SHADER_READ_ONLY（worker 独占）

    PresentSlot slots[4];
    int cur_slot = 0;          // 最新真实帧写入 copy_img[cur_slot]
    bool has_prev = false;
    int observe_left = 0;      // 初始化成功后观察帧数（只 worker 访问）：期间只拷贝不插帧，
                               // 用于隔离"插帧动作本身是否干扰游戏"并避开启动场景剧烈变化期
    long present_count = 0;
    int perf_level = 0;        // 性能自适应档位（worker 独占）：0=满血 1=半径2 2=隔帧 3=停插
    int perf_drops = 0;        // 连续丢帧计数（GPU 队列积压指标，worker 独占）
    std::atomic<bool> resources_ok{false};
    std::atomic<bool> insert_disabled{false};  // usage/图像数/初始化失败时置位
    std::atomic<bool> init_pending{false};     // 初始化任务已入队（去重，防止反复 push）
    std::atomic<bool> dying{false};            // 游戏已销毁该 swapchain（worker 应尽快放弃并清理）
};

static std::mutex g_mtx;
static std::map<VkDevice, DeviceCtx *> g_devices;

// ---------------- 异步插帧工作线程 ----------------
// present 回调只 push 任务到队列，立即返回；GPU 操作（copy/blit/compute/renderpass/额外 present）
// 以及交换链资源创建/销毁全部在 worker 线程做——游戏渲染线程零 GPU 重活，避免首次创建管线卡死黑屏。
struct PendingFrame {
    SwapchainCtx *cx = nullptr;
    VkSwapchainKHR sc = VK_NULL_HANDLE;   // 冗余存句柄：worker 处理前按 sc 查最新 cx，防悬垂
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t idx = 0;
    uint64_t present_count = 0;
    bool init_only = false;               // true = 只做资源初始化（首次 present 时 push）
};
static std::queue<PendingFrame> g_pending;
static std::condition_variable g_cv;
static std::thread g_worker;
static bool g_worker_started = false;
static bool g_worker_stop = false;
static std::mutex g_init_mtx;   // 仅 worker 内部串行化 INIT 与 zombie 清理（游戏线程不触碰，杜绝死锁）
static std::map<VkSwapchainKHR, SwapchainCtx *> g_swapchains;
static std::vector<SwapchainCtx *> g_zombie;   // 游戏已销毁、待 worker 延迟清理的 cx（g_mtx 保护）

// ---------------- 工具 ----------------

static uint32_t find_memory_type(DeviceCtx *dc, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props;
    g_GetPhysDevMemProps(dc->pd, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return UINT32_MAX;
}

static void image_barrier(DeviceCtx *dc, VkCommandBuffer cmd, VkImage img,
                          VkImageLayout oldl, VkImageLayout newl,
                          VkAccessFlags srcacc, VkAccessFlags dstacc,
                          VkPipelineStageFlags srcstage, VkPipelineStageFlags dststage,
                          uint32_t family_src = VK_QUEUE_FAMILY_IGNORED, uint32_t family_dst = VK_QUEUE_FAMILY_IGNORED) {
    // sync2：VkImageMemoryBarrier2 位值与旧版一致（Vulkan 规范保证），直接转换；
    // 单次调用可同时携带图/缓冲 barrier，且 stage/access 更细，驱动可少做全管线 flush。
    if (dc && dc->sync2) {
        VkImageMemoryBarrier2 b2{};
        b2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b2.oldLayout = oldl;
        b2.newLayout = newl;
        b2.srcQueueFamilyIndex = family_src;
        b2.dstQueueFamilyIndex = family_dst;
        b2.image = img;
        b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b2.srcAccessMask = (VkAccessFlags2)srcacc;
        b2.dstAccessMask = (VkAccessFlags2)dstacc;
        b2.srcStageMask = (VkPipelineStageFlags2)srcstage;
        b2.dstStageMask = (VkPipelineStageFlags2)dststage;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b2;
        dc->f.CmdPipelineBarrier2(cmd, &dep);
        return;
    }
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = oldl;
    b.newLayout = newl;
    b.srcQueueFamilyIndex = family_src;
    b.dstQueueFamilyIndex = family_dst;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = srcacc;
    b.dstAccessMask = dstacc;
    vkCmdPipelineBarrier(cmd, srcstage, dststage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static void buffer_barrier(DeviceCtx *dc, VkCommandBuffer cmd, VkBuffer buf,
                           VkAccessFlags srcacc, VkAccessFlags dstacc,
                           VkPipelineStageFlags srcstage, VkPipelineStageFlags dststage) {
    if (dc && dc->sync2) {
        VkBufferMemoryBarrier2 b2{};
        b2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        b2.buffer = buf;
        b2.offset = 0;
        b2.size = VK_WHOLE_SIZE;
        b2.srcAccessMask = (VkAccessFlags2)srcacc;
        b2.dstAccessMask = (VkAccessFlags2)dstacc;
        b2.srcStageMask = (VkPipelineStageFlags2)srcstage;
        b2.dstStageMask = (VkPipelineStageFlags2)dststage;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &b2;
        dc->f.CmdPipelineBarrier2(cmd, &dep);
        return;
    }
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.buffer = buf; b.offset = 0; b.size = VK_WHOLE_SIZE;
    b.srcAccessMask = srcacc;
    b.dstAccessMask = dstacc;
    dc->f.CmdPipelineBarrier(cmd, srcstage, dststage, 0, 0, nullptr, 1, &b, 0, nullptr);
}

// 创建一张可采样的图像（DEVICE_LOCAL）
static bool make_image(DeviceCtx *dc, VkImage *img, VkDeviceMemory *mem, VkImageView *view,
                       VkFormat fmt, int w, int h, VkImageUsageFlags usage) {
    const DevFns &f = dc->f;
    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = fmt;
    ii.extent = {(uint32_t)w, (uint32_t)h, 1};
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (f.CreateImage(dc->dev, &ii, nullptr, img) != VK_SUCCESS) return false;
    // T2-1：优先走 Requirements2/Bind2（1.1 core）：驱动报告 prefersDedicatedAllocation
    // 时挂 VkMemoryDedicatedAllocateInfo（IMAGELESS 场景下某些驱动强制要求 dedicated）。
    // 函数缺失或分配/绑定失败 → 回退经典 Requirements1/Bind1 路径。
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bool dedicated = false;
    if (f.GetImageMemoryRequirements2 && f.BindImageMemory2) {
        VkMemoryDedicatedRequirements ded{};
        ded.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
        VkMemoryRequirements2 mr2{};
        mr2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        mr2.pNext = &ded;
        VkImageMemoryRequirementsInfo2 ir{};
        ir.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
        ir.image = *img;
        f.GetImageMemoryRequirements2(dc->dev, &ir, &mr2);
        ma.allocationSize = mr2.memoryRequirements.size;
        ma.memoryTypeIndex = find_memory_type(dc, mr2.memoryRequirements.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        dedicated = (ded.requiresDedicatedAllocation || ded.prefersDedicatedAllocation) != VK_FALSE;
        if (dedicated) {
            VkMemoryDedicatedAllocateInfo da{};
            da.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
            da.image = *img;
            da.pNext = ma.pNext;
            ma.pNext = &da;
        }
        if (ma.memoryTypeIndex != UINT32_MAX && f.AllocateMemory(dc->dev, &ma, nullptr, &m) == VK_SUCCESS) {
            VkBindImageMemoryInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
            bi.image = *img;
            bi.memory = m;
            if (f.BindImageMemory2(dc->dev, 1, &bi) == VK_SUCCESS) { *mem = m; }
            else { f.FreeMemory(dc->dev, m, nullptr); m = VK_NULL_HANDLE; }
        }
    }
    if (!m) {
        VkMemoryRequirements mr;
        f.GetImageMemoryRequirements(dc->dev, *img, &mr);
        ma.pNext = nullptr;
        ma.allocationSize = mr.size;
        ma.memoryTypeIndex = find_memory_type(dc, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ma.memoryTypeIndex == UINT32_MAX || f.AllocateMemory(dc->dev, &ma, nullptr, &m) != VK_SUCCESS)
            return false;
        f.BindImageMemory(dc->dev, *img, m, 0);
        *mem = m;
    }
    VkImageViewCreateInfo iv{};
    iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv.image = *img;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = fmt;
    iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    iv.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    if (f.CreateImageView(dc->dev, &iv, nullptr, view) != VK_SUCCESS) return false;
    return true;
}

// 设备本地 buffer（GMV 直方图等）：STORAGE + TRANSFER_DST（GPU 侧 CmdFillBuffer 清零）
static bool make_buffer(DeviceCtx *dc, VkBuffer *buf, VkDeviceMemory *mem, VkDeviceSize size) {
    const DevFns &f = dc->f;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (f.CreateBuffer(dc->dev, &bi, nullptr, buf) != VK_SUCCESS) return false;
    // T2-1：buffer 侧统一走 Requirements2/Bind2（非 dedicated，但同一套路径）
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkMemoryAllocateInfo ma{};
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    if (f.GetBufferMemoryRequirements2 && f.BindBufferMemory2) {
        VkMemoryRequirements2 mr2{};
        mr2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        VkBufferMemoryRequirementsInfo2 br{};
        br.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
        br.buffer = *buf;
        f.GetBufferMemoryRequirements2(dc->dev, &br, &mr2);
        ma.allocationSize = mr2.memoryRequirements.size;
        ma.memoryTypeIndex = find_memory_type(dc, mr2.memoryRequirements.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ma.memoryTypeIndex != UINT32_MAX && f.AllocateMemory(dc->dev, &ma, nullptr, &m) == VK_SUCCESS) {
            VkBindBufferMemoryInfo bb{};
            bb.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
            bb.buffer = *buf;
            bb.memory = m;
            if (f.BindBufferMemory2(dc->dev, 1, &bb) == VK_SUCCESS) *mem = m;
            else { f.FreeMemory(dc->dev, m, nullptr); m = VK_NULL_HANDLE; }
        }
    }
    if (!m) {
        VkMemoryRequirements mr;
        f.GetBufferMemoryRequirements(dc->dev, *buf, &mr);
        ma.allocationSize = mr.size;
        ma.memoryTypeIndex = find_memory_type(dc, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ma.memoryTypeIndex == UINT32_MAX || f.AllocateMemory(dc->dev, &ma, nullptr, &m) != VK_SUCCESS)
            return false;
        f.BindBufferMemory(dc->dev, *buf, m, 0);
        *mem = m;
    }
    return true;
}

// ---------------- 交换链资源 ----------------

static void destroy_swapchain_resources(SwapchainCtx *cx) {
    DeviceCtx *dc = cx->dc;
    const DevFns &f = dc->f;
    if (!cx->resources_ok && cx->copy_img[0] == VK_NULL_HANDLE) return;
    for (auto &s : cx->slots) {
        if (s.fence) { f.WaitForFences(dc->dev, 1, &s.fence, VK_TRUE, 1000000000ull); f.DestroyFence(dc->dev, s.fence, nullptr); }
        if (s.done_ts) f.DestroySemaphore(dc->dev, s.done_ts, nullptr);
        if (s.acquire_sem) f.DestroySemaphore(dc->dev, s.acquire_sem, nullptr);
        if (s.done_sem) f.DestroySemaphore(dc->dev, s.done_sem, nullptr);
        s = PresentSlot{};
    }
    if (cx->pipeline) f.DestroyPipeline(dc->dev, cx->pipeline, nullptr);
    if (cx->pl) f.DestroyPipelineLayout(dc->dev, cx->pl, nullptr);
    if (cx->cs0_pipeline) f.DestroyPipeline(dc->dev, cx->cs0_pipeline, nullptr);
    if (cx->cs1_pipeline) f.DestroyPipeline(dc->dev, cx->cs1_pipeline, nullptr);
    if (cx->gmv0_pipeline) f.DestroyPipeline(dc->dev, cx->gmv0_pipeline, nullptr);
    if (cx->gmv1_pipeline) f.DestroyPipeline(dc->dev, cx->gmv1_pipeline, nullptr);
    if (cx->cs0_pl) f.DestroyPipelineLayout(dc->dev, cx->cs0_pl, nullptr);
    if (cx->cs1_pl) f.DestroyPipelineLayout(dc->dev, cx->cs1_pl, nullptr);
    if (cx->gmv0_pl) f.DestroyPipelineLayout(dc->dev, cx->gmv0_pl, nullptr);
    if (cx->gmv1_pl) f.DestroyPipelineLayout(dc->dev, cx->gmv1_pl, nullptr);
    if (cx->vs) f.DestroyShaderModule(dc->dev, cx->vs, nullptr);
    if (cx->fs) f.DestroyShaderModule(dc->dev, cx->fs, nullptr);
    if (cx->cs0) f.DestroyShaderModule(dc->dev, cx->cs0, nullptr);
    if (cx->cs1) f.DestroyShaderModule(dc->dev, cx->cs1, nullptr);
    if (cx->cs0f) f.DestroyShaderModule(dc->dev, cx->cs0f, nullptr);
    if (cx->cs1f) f.DestroyShaderModule(dc->dev, cx->cs1f, nullptr);
    if (cx->cs_gmv0) f.DestroyShaderModule(dc->dev, cx->cs_gmv0, nullptr);
    if (cx->cs_gmv1) f.DestroyShaderModule(dc->dev, cx->cs_gmv1, nullptr);
    if (cx->cs0_ds_pool) f.DestroyDescriptorPool(dc->dev, cx->cs0_ds_pool, nullptr);
    if (cx->cs1_ds_pool) f.DestroyDescriptorPool(dc->dev, cx->cs1_ds_pool, nullptr);
    if (cx->cs0_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->cs0_ds_layout, nullptr);
    if (cx->cs1_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->cs1_ds_layout, nullptr);
    if (cx->gmv0_ds_pool) f.DestroyDescriptorPool(dc->dev, cx->gmv0_ds_pool, nullptr);
    if (cx->gmv1_ds_pool) f.DestroyDescriptorPool(dc->dev, cx->gmv1_ds_pool, nullptr);
    if (cx->gmv0_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->gmv0_ds_layout, nullptr);
    if (cx->gmv1_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->gmv1_ds_layout, nullptr);
    // L2 多尺度清理
    if (cx->cs0_l2_pipeline) f.DestroyPipeline(dc->dev, cx->cs0_l2_pipeline, nullptr);
    if (cx->cs1_l2_pipeline) f.DestroyPipeline(dc->dev, cx->cs1_l2_pipeline, nullptr);
    if (cx->cs0_l2_pl) f.DestroyPipelineLayout(dc->dev, cx->cs0_l2_pl, nullptr);
    if (cx->cs1_l2_pl) f.DestroyPipelineLayout(dc->dev, cx->cs1_l2_pl, nullptr);
    if (cx->cs0_l2) f.DestroyShaderModule(dc->dev, cx->cs0_l2, nullptr);
    if (cx->cs1_l2) f.DestroyShaderModule(dc->dev, cx->cs1_l2, nullptr);
    if (cx->cs1_sub) f.DestroyShaderModule(dc->dev, cx->cs1_sub, nullptr);
    if (cx->cs0_l2f) f.DestroyShaderModule(dc->dev, cx->cs0_l2f, nullptr);
    if (cx->cs1_l2f) f.DestroyShaderModule(dc->dev, cx->cs1_l2f, nullptr);
    if (cx->cs0_l2_ds_pool) f.DestroyDescriptorPool(dc->dev, cx->cs0_l2_ds_pool, nullptr);
    if (cx->cs1_l2_ds_pool) f.DestroyDescriptorPool(dc->dev, cx->cs1_l2_ds_pool, nullptr);
    if (cx->cs0_l2_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->cs0_l2_ds_layout, nullptr);
    if (cx->cs1_l2_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->cs1_l2_ds_layout, nullptr);
    if (cx->ds_pool) f.DestroyDescriptorPool(dc->dev, cx->ds_pool, nullptr);
    if (cx->ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cx->ds_layout, nullptr);
    if (cx->sampler) f.DestroySampler(dc->dev, cx->sampler, nullptr);
    for (auto fb : cx->framebuffers) f.DestroyFramebuffer(dc->dev, fb, nullptr);
    cx->framebuffers.clear();
    if (cx->render_pass) f.DestroyRenderPass(dc->dev, cx->render_pass, nullptr);
    for (auto v : cx->image_views) f.DestroyImageView(dc->dev, v, nullptr);
    cx->image_views.clear();
    for (int i = 0; i < 2; i++) {
        if (cx->copy_view[i]) f.DestroyImageView(dc->dev, cx->copy_view[i], nullptr);
        if (cx->copy_img[i]) f.DestroyImage(dc->dev, cx->copy_img[i], nullptr);
        if (cx->copy_mem[i]) f.FreeMemory(dc->dev, cx->copy_mem[i], nullptr);
        cx->copy_view[i] = VK_NULL_HANDLE; cx->copy_img[i] = VK_NULL_HANDLE; cx->copy_mem[i] = VK_NULL_HANDLE;
        cx->copy_layout_init[i] = false;
    }
    for (int i = 0; i < 2; i++) {
        if (cx->ds_view[i]) f.DestroyImageView(dc->dev, cx->ds_view[i], nullptr);
        if (cx->ds_img[i]) f.DestroyImage(dc->dev, cx->ds_img[i], nullptr);
        if (cx->ds_mem[i]) f.FreeMemory(dc->dev, cx->ds_mem[i], nullptr);
        cx->ds_view[i] = VK_NULL_HANDLE; cx->ds_img[i] = VK_NULL_HANDLE; cx->ds_mem[i] = VK_NULL_HANDLE;
        cx->ds_layout_init[i] = false;
    }
    if (cx->mv_view) f.DestroyImageView(dc->dev, cx->mv_view, nullptr);
    if (cx->mv_img) f.DestroyImage(dc->dev, cx->mv_img, nullptr);
    if (cx->mv_mem) f.FreeMemory(dc->dev, cx->mv_mem, nullptr);
    cx->mv_view = VK_NULL_HANDLE; cx->mv_img = VK_NULL_HANDLE; cx->mv_mem = VK_NULL_HANDLE;
    if (cx->mv_viewB) f.DestroyImageView(dc->dev, cx->mv_viewB, nullptr);
    if (cx->mv_imgB) f.DestroyImage(dc->dev, cx->mv_imgB, nullptr);
    if (cx->mv_memB) f.FreeMemory(dc->dev, cx->mv_memB, nullptr);
    cx->mv_viewB = VK_NULL_HANDLE; cx->mv_imgB = VK_NULL_HANDLE; cx->mv_memB = VK_NULL_HANDLE;
    if (cx->gmv_view) f.DestroyImageView(dc->dev, cx->gmv_view, nullptr);
    if (cx->gmv_img) f.DestroyImage(dc->dev, cx->gmv_img, nullptr);
    if (cx->gmv_mem) f.FreeMemory(dc->dev, cx->gmv_mem, nullptr);
    if (cx->hist_buf) f.DestroyBuffer(dc->dev, cx->hist_buf, nullptr);
    if (cx->hist_mem) f.FreeMemory(dc->dev, cx->hist_mem, nullptr);
    cx->gmv_view = VK_NULL_HANDLE; cx->gmv_img = VK_NULL_HANDLE; cx->gmv_mem = VK_NULL_HANDLE;
    cx->gmv_layout_init = false; cx->hist_buf = VK_NULL_HANDLE; cx->hist_mem = VK_NULL_HANDLE;
    // L2 图像/内存
    for (int i = 0; i < 2; i++) {
        if (cx->ds_view_l2[i]) f.DestroyImageView(dc->dev, cx->ds_view_l2[i], nullptr);
        if (cx->ds_img_l2[i]) f.DestroyImage(dc->dev, cx->ds_img_l2[i], nullptr);
        if (cx->ds_mem_l2[i]) f.FreeMemory(dc->dev, cx->ds_mem_l2[i], nullptr);
        cx->ds_view_l2[i] = VK_NULL_HANDLE; cx->ds_img_l2[i] = VK_NULL_HANDLE; cx->ds_mem_l2[i] = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 2; i++) {
        if (cx->mv_view_l2[i]) f.DestroyImageView(dc->dev, cx->mv_view_l2[i], nullptr);
        if (cx->mv_img_l2[i]) f.DestroyImage(dc->dev, cx->mv_img_l2[i], nullptr);
        if (cx->mv_mem_l2[i]) f.FreeMemory(dc->dev, cx->mv_mem_l2[i], nullptr);
        cx->mv_view_l2[i] = VK_NULL_HANDLE; cx->mv_img_l2[i] = VK_NULL_HANDLE; cx->mv_mem_l2[i] = VK_NULL_HANDLE;
    }
    cx->l2_ok = false;
    cx->mci_ok = false;
    cx->resources_ok = false;
    cx->has_prev = false;
    cx->cur_slot = 0;
}

// worker 线程异步初始化交换链资源（首次 present 时由 INIT 任务触发）。
// 仅在 worker 线程、g_init_mtx 下调用；游戏线程绝不进入。
// 失败路径只置 insert_disabled（不销毁资源）：cx 会随游戏后续 DestroySwapchainKHR 或 zombie 清理统一回收，
// 避免与 worker 的 zombie 清理产生 double-destroy。
static bool create_swapchain_resources(SwapchainCtx *cx);   // 前向声明（定义在下方）
static void init_swapchain_resources(SwapchainCtx *cx, VkQueue queue) {
    DeviceCtx *dc = cx->dc;

    if (dc->cmd_pool == VK_NULL_HANDLE) {
        uint32_t family = dc->queue_family[queue];
        VkCommandPoolCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cp.queueFamilyIndex = family;
        if (dc->f.CreateCommandPool(dc->dev, &cp, nullptr, &dc->cmd_pool) != VK_SUCCESS) {
            VKLOGE("init[1/2] 命令池创建失败，禁用插帧");
            cx->insert_disabled = true;
            cx->init_pending = false;
            return;
        }
        dc->cmd_pool_family = family;
    }
    if (cx->dying) { cx->init_pending = false; return; }
    if (!create_swapchain_resources(cx)) {
        VKLOGE("init[2/2] 交换链资源创建失败，禁用插帧（资源随销毁统一回收）");
        cx->insert_disabled = true;
        cx->init_pending = false;
        return;
    }
    if (cx->images.size() < 3) {
        VKLOGE("init[2/2] 交换链仅 %zu 张图（需要 >=3），禁用插帧以避免额外开销", cx->images.size());
        cx->insert_disabled = true;
        cx->init_pending = false;
        return;
    }
    if (cx->dying) { cx->init_pending = false; return; }
    VKLOGI("Vulkan 补帧就绪: %ux%u, %zu 张交换链图像%s（观察 %d 帧后开始插帧）",
           cx->extent.width, cx->extent.height, cx->images.size(),
           cx->mci_ok ? ", MCI 已启用" : ", MCI 不可用(混合模式)",
           cx->observe_left > 0 ? cx->observe_left : 30);
    // 实际算法名（此时 mci_ok 已确定）
    McfiConfig cfg;
    mcfi_get_config(&cfg);
    const char *algo = (cfg.interp_mode == 0) ? (cx->mci_ok ? "MCI运动补偿" : "普通混合")
                     : (cfg.interp_mode == 2 ? "运动自适应" : "普通混合");
    static bool g_vk_reported = false;
    if (!g_vk_reported) {
        g_vk_reported = true;
        char msg[160];
        snprintf(msg, sizeof(msg), "pid=%d: MCFI 补帧管线生效 %ux%u [Vulkan %s%s]",
                 getpid(), cx->extent.width, cx->extent.height, algo,
                 (cx->mci_ok && cx->cs0f) ? " fp16" : (cx->mci_ok ? " fp32" : ""));
        VKLOGI("%s", msg);
        mcfi_send_event(msg);
    }
    cx->init_pending = false;
}

static bool create_swapchain_resources(SwapchainCtx *cx) {
    DeviceCtx *dc = cx->dc;
    const DevFns &f = dc->f;
    VkResult r;

    uint32_t n = 0;
    r = f.GetSwapchainImagesKHR(dc->dev, cx->sc, &n, nullptr);
    if (r != VK_SUCCESS || n == 0) { VKLOGE("init[1/9] GetSwapchainImages 失败 (%d)", (int)r); return false; }
    cx->images.resize(n);
    r = f.GetSwapchainImagesKHR(dc->dev, cx->sc, &n, cx->images.data());
    if (r != VK_SUCCESS) { VKLOGE("init[1/9] GetSwapchainImages 二次调用失败"); return false; }
    VKLOGI("init[1/9] 交换链图像 %u 张", n);
    if (cx->dying) return false;

    cx->image_views.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = cx->images[i];
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = cx->format;
        iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        iv.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        if (f.CreateImageView(dc->dev, &iv, nullptr, &cx->image_views[i]) != VK_SUCCESS) { VKLOGE("init[2/9] CreateImageView(%u) 失败", i); return false; }
    }
    VKLOGI("init[2/9] 图像视图完成");
    if (cx->dying) return false;

    VkAttachmentDescription att{};
    att.format = cx->format;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments = &att;
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;
    rp.dependencyCount = 1;
    rp.pDependencies = &dep;
    if (f.CreateRenderPass(dc->dev, &rp, nullptr, &cx->render_pass) != VK_SUCCESS) { VKLOGE("init[3/9] CreateRenderPass 失败"); return false; }
    VKLOGI("init[3/9] renderpass 完成");

    VkFramebufferCreateInfo fb{};
    fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb.renderPass = cx->render_pass;
    fb.attachmentCount = 1;
    fb.width = cx->extent.width;
    fb.height = cx->extent.height;
    fb.layers = 1;
    cx->framebuffers.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        fb.pAttachments = &cx->image_views[i];
        if (f.CreateFramebuffer(dc->dev, &fb, nullptr, &cx->framebuffers[i]) != VK_SUCCESS) { VKLOGE("init[4/9] CreateFramebuffer(%u) 失败", i); return false; }
    }
    VKLOGI("init[4/9] framebuffer 完成");
    if (cx->dying) return false;

    // 着色器与管线
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = vert_spv_len; sm.pCode = (const uint32_t *)vert_spv;
    if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->vs) != VK_SUCCESS) { VKLOGE("init[5/9] CreateShaderModule(vert) 失败"); return false; }
    sm.codeSize = mci_spv_len; sm.pCode = (const uint32_t *)mci_spv;
    if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->fs) != VK_SUCCESS) { VKLOGE("init[5/9] CreateShaderModule(frag) 失败"); return false; }
    VKLOGI("init[5/9] shader 模块完成");

    // 插帧图形管线：4 个 combined image sampler（prev/cur/mv/gmv），push constant 32B
    VkDescriptorSetLayoutBinding bindings[4]{};
    for (int b = 0; b < 4; b++)
        bindings[b] = {(uint32_t)b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsl{};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 4;
    dsl.pBindings = bindings;
    if (f.CreateDescriptorSetLayout(dc->dev, &dsl, nullptr, &cx->ds_layout) != VK_SUCCESS) { VKLOGE("init[6/9] CreateDescriptorSetLayout 失败"); return false; }

    VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pll{};
    pll.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pll.setLayoutCount = 1;
    pll.pSetLayouts = &cx->ds_layout;
    pll.pushConstantRangeCount = 1;
    pll.pPushConstantRanges = &pc;
    if (f.CreatePipelineLayout(dc->dev, &pll, nullptr, &cx->pl) != VK_SUCCESS) { VKLOGE("init[6/9] CreatePipelineLayout 失败"); return false; }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = cx->vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = cx->fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0, 0, (float)cx->extent.width, (float)cx->extent.height, 0, 1};
    VkRect2D scissor{{0, 0}, cx->extent};
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount = 1; vp.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cba;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount = 2; gp.pStages = stages;
    gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp; gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms; gp.pColorBlendState = &cb;
    gp.layout = cx->pl; gp.renderPass = cx->render_pass;
    if (f.CreateGraphicsPipelines(dc->dev, dc->cache, 1, &gp, nullptr, &cx->pipeline) != VK_SUCCESS) { VKLOGE("init[6/9] CreateGraphicsPipelines 失败"); return false; }
    VKLOGI("init[6/9] 图形管线完成");
    if (cx->dying) return false;

    // 私有拷贝图像 x2（TRANSFER_DST | SAMPLED）
    for (int i = 0; i < 2; i++) {
        if (!make_image(dc, &cx->copy_img[i], &cx->copy_mem[i], &cx->copy_view[i],
                        cx->format, cx->extent.width, cx->extent.height,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
            VKLOGE("init[7/9] 拷贝图像(%d) 创建失败", i);
            return false;
        }
    }
    VKLOGI("init[7/9] 拷贝图像完成");

    // ---- MCI 资源：降采样 + 双缓冲运动矢量场 + 双 pass compute 管线 ----
    // pass0（cs0）：全搜索，只写 mvA（writeonly storage image）
    // pass1（cs1）：以 sampler 读 mvA 粗矢量 ±sr 细化，只写 mvB（writeonly storage image）
    // 同一张 image 绝不混合读写——该驱动 image 混合读写会导致 GLES 编译失败 / Vulkan 驱动崩溃。
    cx->mci_ok = false;
    McfiConfig vkcfg;
    mcfi_get_config(&vkcfg);
    if (!vkcfg.vk_mci) {
        VKLOGI("Vulkan MCI 未启用（vk_mci=0），普通混合插帧；可在面板开启 MCI");
    } else if (g_GetPhysDevFmtProps) {
        VkFormatProperties fp{};
        g_GetPhysDevFmtProps(dc->pd, cx->format, &fp);
        VkFormatProperties fp8{};
        g_GetPhysDevFmtProps(dc->pd, VK_FORMAT_R8G8B8A8_UNORM, &fp8);
        bool blit_src = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
        bool blit_dst = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
        bool stor_ok  = (fp8.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
        if (blit_src && blit_dst && stor_ok) {
            cx->dsW = ((int)cx->extent.width + 3) / 4;
            cx->dsH = ((int)cx->extent.height + 3) / 4;
            cx->mvW = (cx->dsW + 7) / 8;
            cx->mvH = (cx->dsH + 7) / 8;
            for (int i = 0; i < 2; i++) {
                if (!make_image(dc, &cx->ds_img[i], &cx->ds_mem[i], &cx->ds_view[i],
                                cx->format, cx->dsW, cx->dsH,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                    VKLOGE("init[8a/9] 降采样图像(%d) 创建失败", i);
                    return false;
                }
            }
            VKLOGI("init[8a/9] 降采样图像完成");
            // mvA：pass0 写（storage）+ pass1 读（sampler）
            if (!make_image(dc, &cx->mv_img, &cx->mv_mem, &cx->mv_view,
                            VK_FORMAT_R8G8B8A8_UNORM, cx->mvW, cx->mvH,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                VKLOGE("init[8b/9] 运动矢量图像A 创建失败");
                return false;
            }
            // mvB：pass1 写（storage）+ 插帧 frag 读（sampler）
            if (!make_image(dc, &cx->mv_imgB, &cx->mv_memB, &cx->mv_viewB,
                            VK_FORMAT_R8G8B8A8_UNORM, cx->mvW, cx->mvH,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                VKLOGE("init[8b/9] 运动矢量图像B 创建失败");
                return false;
            }
            // GMV：1×1 RGBA32F（gmv1 写 storage + me1/mci 读 sampler）+ 直方图 storage buffer
            if (!make_image(dc, &cx->gmv_img, &cx->gmv_mem, &cx->gmv_view,
                            VK_FORMAT_R32G32B32A32_SFLOAT, 1, 1,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                VKLOGE("init[8b/9] GMV 图像创建失败");
                return false;
            }
            if (!make_buffer(dc, &cx->hist_buf, &cx->hist_mem, (1024 + 1) * sizeof(int))) {
                VKLOGE("init[8b/9] GMV 直方图 buffer 创建失败");
                return false;
            }
            VKLOGI("init[8b/9] 运动矢量图像完成（A/B 双缓冲）");

            // ---- L2 多尺度（1/8）资源 ----
            cx->dsW2 = (cx->dsW + 1) / 2;
            cx->dsH2 = (cx->dsH + 1) / 2;
            cx->mvW2 = (cx->dsW2 + 7) / 8;
            cx->mvH2 = (cx->dsH2 + 7) / 8;
            bool l2_img_ok = true;
            for (int i = 0; i < 2 && l2_img_ok; i++) {
                if (!make_image(dc, &cx->ds_img_l2[i], &cx->ds_mem_l2[i], &cx->ds_view_l2[i],
                                cx->format, cx->dsW2, cx->dsH2,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                    l2_img_ok = false;
                }
            }
            for (int i = 0; i < 2 && l2_img_ok; i++) {
                if (!make_image(dc, &cx->mv_img_l2[i], &cx->mv_mem_l2[i], &cx->mv_view_l2[i],
                                VK_FORMAT_R8G8B8A8_UNORM, cx->mvW2, cx->mvH2,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                    l2_img_ok = false;
                }
            }
            cx->l2_ok = l2_img_ok;
            cx->subgroup_ok = g_vk_subgroup;
            VKLOGI("L2 多尺度资源: %d (1/8=%dx%d) subgroup=%d", (int)cx->l2_ok, cx->dsW2, cx->dsH2, (int)cx->subgroup_ok);

            // ---- 8c：shader 模块 / 描述符布局 / 管线布局 ----
            sm.codeSize = me0_spv_len; sm.pCode = (const uint32_t *)me0_spv;
            if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs0) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateShaderModule(me0) 失败"); return false; }
            sm.codeSize = me1_spv_len; sm.pCode = (const uint32_t *)me1_spv;
            if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateShaderModule(me1) 失败"); return false; }
            if (g_vk_f16) {
                // fp16（半精度）运动估计：SAD 累加用 float16_t，ALU 开销减半；创建失败静默回退 fp32
                sm.codeSize = me0_f16_spv_len; sm.pCode = (const uint32_t *)me0_f16_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs0f) != VK_SUCCESS) { VKLOGI("init[8c/9] fp16 me0 module 创建失败，回退 fp32"); cx->cs0f = VK_NULL_HANDLE; }
                sm.codeSize = me1_f16_spv_len; sm.pCode = (const uint32_t *)me1_f16_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1f) != VK_SUCCESS) { VKLOGI("init[8c/9] fp16 me1 module 创建失败，回退 fp32"); cx->cs1f = VK_NULL_HANDLE; }
            }
            // cs0（3DRS）：uPrev / uCur / uMvPrev(sampler，上一帧运动场= mvB) / mvA(storage)
            VkDescriptorSetLayoutBinding cb0[4]{};
            cb0[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb0[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb0[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb0[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo cdl0{};
            cdl0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            cdl0.bindingCount = 4;
            cdl0.pBindings = cb0;
            if (f.CreateDescriptorSetLayout(dc->dev, &cdl0, nullptr, &cx->cs0_ds_layout) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateDescriptorSetLayout(cs0) 失败"); return false; }
            // cs1：uPrev / uCur / mvA(sampler) / mvB(storage) / uGMV(sampler)
            VkDescriptorSetLayoutBinding cb1[5]{};
            cb1[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb1[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb1[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb1[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cb1[4] = {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo cdl1{};
            cdl1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            cdl1.bindingCount = 5;
            cdl1.pBindings = cb1;
            if (f.CreateDescriptorSetLayout(dc->dev, &cdl1, nullptr, &cx->cs1_ds_layout) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateDescriptorSetLayout(cs1) 失败"); return false; }
            // cs_gmv0：mvA(sampler) / hist(storage buffer)；cs_gmv1：hist / gmv(storage image)
            sm.codeSize = gmv0_spv_len; sm.pCode = (const uint32_t *)gmv0_spv;
            if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs_gmv0) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateShaderModule(gmv0) 失败"); return false; }
            sm.codeSize = gmv1_spv_len; sm.pCode = (const uint32_t *)gmv1_spv;
            if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs_gmv1) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateShaderModule(gmv1) 失败"); return false; }
            VkDescriptorSetLayoutBinding cbG0[2]{};
            cbG0[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cbG0[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo cdlG0{};
            cdlG0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            cdlG0.bindingCount = 2; cdlG0.pBindings = cbG0;
            if (f.CreateDescriptorSetLayout(dc->dev, &cdlG0, nullptr, &cx->gmv0_ds_layout) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateDescriptorSetLayout(gmv0) 失败"); return false; }
            VkDescriptorSetLayoutBinding cbG1[2]{};
            cbG1[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            cbG1[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo cdlG1{};
            cdlG1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            cdlG1.bindingCount = 2; cdlG1.pBindings = cbG1;
            if (f.CreateDescriptorSetLayout(dc->dev, &cdlG1, nullptr, &cx->gmv1_ds_layout) != VK_SUCCESS) { VKLOGE("init[8c/9] CreateDescriptorSetLayout(gmv1) 失败"); return false; }
            VkPushConstantRange cpc0{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};   // ivec4 uP0
            VkPushConstantRange cpc1{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};   // ivec4 uP0 + int sr
            VkPipelineLayoutCreateInfo cpl0{};
            cpl0.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            cpl0.setLayoutCount = 1;
            cpl0.pSetLayouts = &cx->cs0_ds_layout;
            cpl0.pushConstantRangeCount = 1;
            cpl0.pPushConstantRanges = &cpc0;
            if (f.CreatePipelineLayout(dc->dev, &cpl0, nullptr, &cx->cs0_pl) != VK_SUCCESS) { VKLOGE("init[8c/9] CreatePipelineLayout(cs0) 失败"); return false; }
            VkPipelineLayoutCreateInfo cpl1{};
            cpl1.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            cpl1.setLayoutCount = 1;
            cpl1.pSetLayouts = &cx->cs1_ds_layout;
            cpl1.pushConstantRangeCount = 1;
            cpl1.pPushConstantRanges = &cpc1;
            if (f.CreatePipelineLayout(dc->dev, &cpl1, nullptr, &cx->cs1_pl) != VK_SUCCESS) { VKLOGE("init[8c/9] CreatePipelineLayout(cs1) 失败"); return false; }
            VkPipelineLayoutCreateInfo cplG0{};
            cplG0.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            cplG0.setLayoutCount = 1; cplG0.pSetLayouts = &cx->gmv0_ds_layout;
            cplG0.pushConstantRangeCount = 1; cplG0.pPushConstantRanges = &cpc0;
            if (f.CreatePipelineLayout(dc->dev, &cplG0, nullptr, &cx->gmv0_pl) != VK_SUCCESS) { VKLOGE("init[8c/9] CreatePipelineLayout(gmv0) 失败"); return false; }
            VkPipelineLayoutCreateInfo cplG1{};
            cplG1.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            cplG1.setLayoutCount = 1; cplG1.pSetLayouts = &cx->gmv1_ds_layout;
            cplG1.pushConstantRangeCount = 1; cplG1.pPushConstantRanges = &cpc0;
            if (f.CreatePipelineLayout(dc->dev, &cplG1, nullptr, &cx->gmv1_pl) != VK_SUCCESS) { VKLOGE("init[8c/9] CreatePipelineLayout(gmv1) 失败"); return false; }
            VKLOGI("init[8c/9] compute 布局完成（cs0/cs1/gmv0/gmv1）");

            // ---- L2：shader 模块 / 布局 / 管线布局（创建失败仅置 l2_ok=false，不致命）----
            if (cx->l2_ok) {
                sm.codeSize = me0_l2_spv_len; sm.pCode = (const uint32_t *)me0_l2_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs0_l2) != VK_SUCCESS) { VKLOGI("L2 me0_l2 module 失败"); cx->l2_ok = false; }
            }
            if (cx->l2_ok) {
                sm.codeSize = me1_l2_spv_len; sm.pCode = (const uint32_t *)me1_l2_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1_l2) != VK_SUCCESS) { VKLOGI("L2 me1_l2 module 失败"); cx->l2_ok = false; }
            }
            if (cx->l2_ok && cx->subgroup_ok) {
                sm.codeSize = me1_sub_spv_len; sm.pCode = (const uint32_t *)me1_sub_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1_sub) != VK_SUCCESS) { VKLOGI("me1_sub module 失败，用普通 me1_l2"); cx->cs1_sub = VK_NULL_HANDLE; }
            }
            if (cx->l2_ok && g_vk_f16) {
                sm.codeSize = me0_l2_f16_spv_len; sm.pCode = (const uint32_t *)me0_l2_f16_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs0_l2f) != VK_SUCCESS) cx->cs0_l2f = VK_NULL_HANDLE;
                sm.codeSize = me1_l2_f16_spv_len; sm.pCode = (const uint32_t *)me1_l2_f16_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1_l2f) != VK_SUCCESS) cx->cs1_l2f = VK_NULL_HANDLE;
            }
            if (cx->l2_ok) {
                VkDescriptorSetLayoutBinding cb0l2[4]{};
                cb0l2[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb0l2[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb0l2[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb0l2[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                VkDescriptorSetLayoutCreateInfo cdl0l2{};
                cdl0l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                cdl0l2.bindingCount=4; cdl0l2.pBindings=cb0l2;
                if (f.CreateDescriptorSetLayout(dc->dev, &cdl0l2, nullptr, &cx->cs0_l2_ds_layout)!=VK_SUCCESS) { VKLOGI("cs0_l2 layout 失败"); cx->l2_ok=false; }
            }
            if (cx->l2_ok) {
                VkDescriptorSetLayoutBinding cb1l2[6]{};
                cb1l2[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[4]={4,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[5]={5,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                VkDescriptorSetLayoutCreateInfo cdl1l2{};
                cdl1l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                cdl1l2.bindingCount=6; cdl1l2.pBindings=cb1l2;
                if (f.CreateDescriptorSetLayout(dc->dev, &cdl1l2, nullptr, &cx->cs1_l2_ds_layout)!=VK_SUCCESS) { VKLOGI("cs1_l2 layout 失败"); cx->l2_ok=false; }
            }
            if (cx->l2_ok) {
                VkPushConstantRange cpc0l2{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
                VkPipelineLayoutCreateInfo cpl0l2{};
                cpl0l2.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                cpl0l2.setLayoutCount=1; cpl0l2.pSetLayouts=&cx->cs0_l2_ds_layout;
                cpl0l2.pushConstantRangeCount=1; cpl0l2.pPushConstantRanges=&cpc0l2;
                if (f.CreatePipelineLayout(dc->dev, &cpl0l2, nullptr, &cx->cs0_l2_pl)!=VK_SUCCESS) { VKLOGI("cs0_l2_pl 失败"); cx->l2_ok=false; }
                VkPushConstantRange cpc1l2{VK_SHADER_STAGE_COMPUTE_BIT, 0, 36};
                VkPipelineLayoutCreateInfo cpl1l2{};
                cpl1l2.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                cpl1l2.setLayoutCount=1; cpl1l2.pSetLayouts=&cx->cs1_l2_ds_layout;
                cpl1l2.pushConstantRangeCount=1; cpl1l2.pPushConstantRanges=&cpc1l2;
                if (f.CreatePipelineLayout(dc->dev, &cpl1l2, nullptr, &cx->cs1_l2_pl)!=VK_SUCCESS) { VKLOGI("cs1_l2_pl 失败"); cx->l2_ok=false; }
            }

            // ---- 8d：compute 管线 / 描述符池与集 ----
            VkComputePipelineCreateInfo cpi0{};
            cpi0.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpi0.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpi0.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi0.stage.pName = "main";
            cpi0.layout = cx->cs0_pl;
            cpi0.stage.module = cx->cs0f ? cx->cs0f : cx->cs0;   // 优先 fp16（设备支持时），失败回退 fp32
            if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi0, nullptr, &cx->cs0_pipeline) != VK_SUCCESS) {
                if (cx->cs0f) { VKLOGI("init[8d/9] fp16 cs0 管线失败，回退 fp32"); cpi0.stage.module = cx->cs0; }
                if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi0, nullptr, &cx->cs0_pipeline) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateComputePipelines(cs0) 失败"); return false; }
            }
            if (cx->cs0f && cx->cs0_pipeline) VKLOGI("init[8d/9] cs0 使用 fp16 半精度管线");
            VkComputePipelineCreateInfo cpi1{};
            cpi1.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpi1.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpi1.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi1.stage.pName = "main";
            cpi1.layout = cx->cs1_pl;
            cpi1.stage.module = cx->cs1f ? cx->cs1f : cx->cs1;   // 优先 fp16
            if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi1, nullptr, &cx->cs1_pipeline) != VK_SUCCESS) {
                if (cx->cs1f) { VKLOGI("init[8d/9] fp16 cs1 管线失败，回退 fp32"); cpi1.stage.module = cx->cs1; }
                if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi1, nullptr, &cx->cs1_pipeline) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateComputePipelines(cs1) 失败"); return false; }
            }
            VkDescriptorPoolSize cps0[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
            VkDescriptorPoolCreateInfo cdp0{};
            cdp0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            cdp0.maxSets = 2; cdp0.poolSizeCount = 2; cdp0.pPoolSizes = cps0;
            if (f.CreateDescriptorPool(dc->dev, &cdp0, nullptr, &cx->cs0_ds_pool) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateDescriptorPool(cs0) 失败"); return false; }
            VkDescriptorPoolSize cps1[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
            VkDescriptorPoolCreateInfo cdp1{};
            cdp1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            cdp1.maxSets = 2; cdp1.poolSizeCount = 2; cdp1.pPoolSizes = cps1;
            if (f.CreateDescriptorPool(dc->dev, &cdp1, nullptr, &cx->cs1_ds_pool) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateDescriptorPool(cs1) 失败"); return false; }
            VkDescriptorSetAllocateInfo cda0{};
            cda0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            cda0.descriptorPool = cx->cs0_ds_pool;
            cda0.descriptorSetCount = 2;
            VkDescriptorSetLayout cs0_layouts[2] = {cx->cs0_ds_layout, cx->cs0_ds_layout};
            cda0.pSetLayouts = cs0_layouts;
            if (f.AllocateDescriptorSets(dc->dev, &cda0, cx->cs0_ds) != VK_SUCCESS) { VKLOGE("init[8d/9] AllocateDescriptorSets(cs0) 失败"); return false; }
            VkDescriptorSetAllocateInfo cda1{};
            cda1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            cda1.descriptorPool = cx->cs1_ds_pool;
            cda1.descriptorSetCount = 2;
            VkDescriptorSetLayout cs1_layouts[2] = {cx->cs1_ds_layout, cx->cs1_ds_layout};
            cda1.pSetLayouts = cs1_layouts;
            if (f.AllocateDescriptorSets(dc->dev, &cda1, cx->cs1_ds) != VK_SUCCESS) { VKLOGE("init[8d/9] AllocateDescriptorSets(cs1) 失败"); return false; }
            // GMV 统计管线（轻量，两个小 shader）
            VkComputePipelineCreateInfo cpiG0{};
            cpiG0.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpiG0.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpiG0.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpiG0.stage.pName = "main"; cpiG0.layout = cx->gmv0_pl; cpiG0.stage.module = cx->cs_gmv0;
            if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpiG0, nullptr, &cx->gmv0_pipeline) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateComputePipelines(gmv0) 失败"); return false; }
            VkComputePipelineCreateInfo cpiG1{};
            cpiG1.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpiG1.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpiG1.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpiG1.stage.pName = "main"; cpiG1.layout = cx->gmv1_pl; cpiG1.stage.module = cx->cs_gmv1;
            if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpiG1, nullptr, &cx->gmv1_pipeline) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateComputePipelines(gmv1) 失败"); return false; }
            VkDescriptorPoolSize cpsG0[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
            VkDescriptorPoolCreateInfo cdpG0{};
            cdpG0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            cdpG0.maxSets = 1; cdpG0.poolSizeCount = 2; cdpG0.pPoolSizes = cpsG0;
            if (f.CreateDescriptorPool(dc->dev, &cdpG0, nullptr, &cx->gmv0_ds_pool) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateDescriptorPool(gmv0) 失败"); return false; }
            VkDescriptorPoolSize cpsG1[2]{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
            VkDescriptorPoolCreateInfo cdpG1{};
            cdpG1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            cdpG1.maxSets = 1; cdpG1.poolSizeCount = 2; cdpG1.pPoolSizes = cpsG1;
            if (f.CreateDescriptorPool(dc->dev, &cdpG1, nullptr, &cx->gmv1_ds_pool) != VK_SUCCESS) { VKLOGE("init[8d/9] CreateDescriptorPool(gmv1) 失败"); return false; }
            VkDescriptorSetAllocateInfo cdaG0{};
            cdaG0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            cdaG0.descriptorPool = cx->gmv0_ds_pool;
            cdaG0.descriptorSetCount = 1; cdaG0.pSetLayouts = &cx->gmv0_ds_layout;
            if (f.AllocateDescriptorSets(dc->dev, &cdaG0, &cx->gmv0_ds) != VK_SUCCESS) { VKLOGE("init[8d/9] AllocateDescriptorSets(gmv0) 失败"); return false; }
            VkDescriptorSetAllocateInfo cdaG1{};
            cdaG1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            cdaG1.descriptorPool = cx->gmv1_ds_pool;
            cdaG1.descriptorSetCount = 1; cdaG1.pSetLayouts = &cx->gmv1_ds_layout;
            if (f.AllocateDescriptorSets(dc->dev, &cdaG1, &cx->gmv1_ds) != VK_SUCCESS) { VKLOGE("init[8d/9] AllocateDescriptorSets(gmv1) 失败"); return false; }

            // ---- L2 pipeline + 描述符池/集（失败仅 l2_ok=false）----
            if (cx->l2_ok) {
                VkComputePipelineCreateInfo cpi0l2{};
                cpi0l2.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                cpi0l2.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                cpi0l2.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
                cpi0l2.stage.pName="main"; cpi0l2.layout=cx->cs0_l2_pl;
                cpi0l2.stage.module = cx->cs0_l2f ? cx->cs0_l2f : cx->cs0_l2;
                if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi0l2, nullptr, &cx->cs0_l2_pipeline)!=VK_SUCCESS) {
                    cpi0l2.stage.module = cx->cs0_l2;   // fp16 失败回退 fp32
                    f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi0l2, nullptr, &cx->cs0_l2_pipeline);
                    if (!cx->cs0_l2_pipeline) { VKLOGI("cs0_l2 pipeline 失败，L2 回退单尺度"); cx->l2_ok=false; }
                }
            }
            if (cx->l2_ok) {
                VkComputePipelineCreateInfo cpi1l2{};
                cpi1l2.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                cpi1l2.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                cpi1l2.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
                cpi1l2.stage.pName="main"; cpi1l2.layout=cx->cs1_l2_pl;
                cpi1l2.stage.module = cx->cs1_sub ? cx->cs1_sub
                                    : (cx->cs1_l2f ? cx->cs1_l2f : cx->cs1_l2);
                // subgroup 版 ME1：若设备支持钉 subgroupSize=16（shader 内 4×4 lane 映射），
                // 强制 16 lane 全量参与 + full subgroups，消除大 subgroupSize 下半数 lane 空转。
                VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg_req{};
                if (cx->cs1_sub && g_vk_sgsc && g_vk_sg_min <= 16 && g_vk_sg_max >= 16) {
                    sg_req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
                    sg_req.requiredSubgroupSize = 16;
                    sg_req.pNext = (void *)cpi1l2.stage.pNext;
                    cpi1l2.stage.pNext = &sg_req;
                    cpi1l2.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                }
                if (f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi1l2, nullptr, &cx->cs1_l2_pipeline)!=VK_SUCCESS) {
                    cpi1l2.stage.pNext = nullptr;
                    cpi1l2.stage.flags = 0;
                    cpi1l2.stage.module = cx->cs1_l2;
                    f.CreateComputePipelines(dc->dev, dc->cache, 1, &cpi1l2, nullptr, &cx->cs1_l2_pipeline);
                    if (!cx->cs1_l2_pipeline) { VKLOGI("cs1_l2 pipeline 失败，L2 回退单尺度"); cx->l2_ok=false; }
                }
            }
            if (cx->l2_ok) {
                VkDescriptorPoolSize ps0l2[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,6},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,2}};
                VkDescriptorPoolCreateInfo cdp0l2{};
                cdp0l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                cdp0l2.maxSets=2; cdp0l2.poolSizeCount=2; cdp0l2.pPoolSizes=ps0l2;
                f.CreateDescriptorPool(dc->dev, &cdp0l2, nullptr, &cx->cs0_l2_ds_pool);
                VkDescriptorSetLayout l0[2]={cx->cs0_l2_ds_layout,cx->cs0_l2_ds_layout};
                VkDescriptorSetAllocateInfo cda0l2{};
                cda0l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                cda0l2.descriptorPool=cx->cs0_l2_ds_pool; cda0l2.descriptorSetCount=2; cda0l2.pSetLayouts=l0;
                f.AllocateDescriptorSets(dc->dev, &cda0l2, cx->cs0_l2_ds);

                VkDescriptorPoolSize ps1l2[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,12},{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,2}};
                VkDescriptorPoolCreateInfo cdp1l2{};
                cdp1l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                cdp1l2.maxSets=2; cdp1l2.poolSizeCount=2; cdp1l2.pPoolSizes=ps1l2;
                f.CreateDescriptorPool(dc->dev, &cdp1l2, nullptr, &cx->cs1_l2_ds_pool);
                VkDescriptorSetLayout l1[2]={cx->cs1_l2_ds_layout,cx->cs1_l2_ds_layout};
                VkDescriptorSetAllocateInfo cda1l2{};
                cda1l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                cda1l2.descriptorPool=cx->cs1_l2_ds_pool; cda1l2.descriptorSetCount=2; cda1l2.pSetLayouts=l1;
                f.AllocateDescriptorSets(dc->dev, &cda1l2, cx->cs1_l2_ds);
                VKLOGI("L2 pipeline 就绪: %s", cx->cs1_sub ? "subgroup" : (cx->cs1_l2f?"fp16":"fp32"));
            }
            cx->mci_ok = true;
            VKLOGI("init[8d/9] MCI 资源完成（双 pass + GMV）");
        } else {
            VKLOGI("交换链格式缺少 blit/storage 能力，MCI 降级为普通混合");
        }
    }

    // 占位 mv：MCI 不可用时（能力不足/未启用）也必须有合法视图，否则描述符引用 NULL view，
    // 严格驱动上会在 draw 时校验失败甚至崩溃（用户核心红线：不得因降级而卡死）。
    if (cx->mv_view == VK_NULL_HANDLE) {
        if (!make_image(dc, &cx->mv_img, &cx->mv_mem, &cx->mv_view,
                        VK_FORMAT_R8G8B8A8_UNORM, 1, 1,
                        VK_IMAGE_USAGE_SAMPLED_BIT)) {
            VKLOGE("init[8/9] 占位 mv 图像A 创建失败");
            return false;
        }
        cx->mvW = 1;
        cx->mvH = 1;
    }
    if (cx->mv_viewB == VK_NULL_HANDLE) {
        if (!make_image(dc, &cx->mv_imgB, &cx->mv_memB, &cx->mv_viewB,
                        VK_FORMAT_R8G8B8A8_UNORM, 1, 1,
                        VK_IMAGE_USAGE_SAMPLED_BIT)) {
            VKLOGE("init[8/9] 占位 mv 图像B 创建失败");
            return false;
        }
    }

    // 插帧描述符（prev/cur/mv/gmv 4 采样器 × 2 set）
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8};
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 2; dp.poolSizeCount = 1; dp.pPoolSizes = &ps;
    if (f.CreateDescriptorPool(dc->dev, &dp, nullptr, &cx->ds_pool) != VK_SUCCESS) { VKLOGE("init[9/9] CreateDescriptorPool 失败"); return false; }
    VkDescriptorSetLayout layouts[2] = {cx->ds_layout, cx->ds_layout};
    VkDescriptorSetAllocateInfo da{};
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = cx->ds_pool;
    da.descriptorSetCount = 2; da.pSetLayouts = layouts;
    if (f.AllocateDescriptorSets(dc->dev, &da, cx->ds) != VK_SUCCESS) { VKLOGE("init[9/9] AllocateDescriptorSets 失败"); return false; }

    VkSamplerCreateInfo sa{};
    sa.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sa.magFilter = VK_FILTER_LINEAR; sa.minFilter = VK_FILTER_LINEAR;
    sa.addressModeU = sa.addressModeV = sa.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (f.CreateSampler(dc->dev, &sa, nullptr, &cx->sampler) != VK_SUCCESS) { VKLOGE("init[9/9] CreateSampler 失败"); return false; }

    for (int k = 0; k < 2; k++) {
        VkDescriptorImageInfo imgs[4];
        imgs[0] = {cx->sampler, cx->copy_view[k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        imgs[1] = {cx->sampler, cx->copy_view[1 - k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // 插帧 frag 读 mvB（refine 输出；无 refine 时 pass1 以 sr=0 把粗矢量转写到 B）
        imgs[2] = {cx->sampler, cx->mv_viewB, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // frag 读 GMV（1×1；MCI 未启用时占位 mv_viewB 尺寸不符也不采样——mode 非 MCI 不触达该分支）
        imgs[3] = {cx->sampler, (cx->gmv_view ? cx->gmv_view : cx->mv_viewB), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w[4]{};
        for (int b = 0; b < 4; b++) {
            w[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[b].dstSet = cx->ds[k];
            w[b].dstBinding = (uint32_t)b;
            w[b].descriptorCount = 1;
            w[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[b].pImageInfo = &imgs[b];
        }
        f.UpdateDescriptorSets(dc->dev, 4, w, 0, nullptr);
    }

    // compute 描述符（按槽位：cs*_ds[k] 的 uPrev=ds[1-k]（上一帧）、uCur=ds[k]（当前帧））
    if (cx->mci_ok) {
        for (int k = 0; k < 2; k++) {
            // cs0（3DRS）：uPrev / uCur / uMvPrev(上一帧场=mvB, sampler) / mvA(storage)
            VkDescriptorImageInfo c0imgs[4];
            c0imgs[0] = {cx->sampler, cx->ds_view[1 - k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            c0imgs[1] = {cx->sampler, cx->ds_view[k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            c0imgs[2] = {cx->sampler, cx->mv_viewB, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            c0imgs[3] = {VK_NULL_HANDLE, cx->mv_view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet w0[4]{};
            for (int b = 0; b < 4; b++) {
                w0[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w0[b].dstSet = cx->cs0_ds[k];
                w0[b].dstBinding = (uint32_t)b;
                w0[b].descriptorCount = 1;
                w0[b].descriptorType = (b == 3) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w0[b].pImageInfo = &c0imgs[b];
            }
            f.UpdateDescriptorSets(dc->dev, 4, w0, 0, nullptr);
            // cs1：uPrev / uCur / mvA(sampler) / mvB(storage) / uGMV(sampler)
            VkDescriptorImageInfo c1imgs[5];
            c1imgs[0] = {cx->sampler, cx->ds_view[1 - k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            c1imgs[1] = {cx->sampler, cx->ds_view[k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            c1imgs[2] = {cx->sampler, cx->mv_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            c1imgs[3] = {VK_NULL_HANDLE, cx->mv_viewB, VK_IMAGE_LAYOUT_GENERAL};
            c1imgs[4] = {cx->sampler, (cx->gmv_view ? cx->gmv_view : cx->mv_view), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet w1[5]{};
            for (int b = 0; b < 5; b++) {
                w1[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w1[b].dstSet = cx->cs1_ds[k];
                w1[b].dstBinding = (uint32_t)b;
                w1[b].descriptorCount = 1;
                w1[b].descriptorType = (b == 3) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w1[b].pImageInfo = &c1imgs[b];
            }
            f.UpdateDescriptorSets(dc->dev, 5, w1, 0, nullptr);
        }
        // GMV 描述符（单例，不随槽位变）
        VkDescriptorImageInfo g0img[1] = {{cx->sampler, cx->mv_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        VkDescriptorBufferInfo g0buf{};
        g0buf.buffer = cx->hist_buf; g0buf.offset = 0; g0buf.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet wG0[2]{};
        wG0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wG0[0].dstSet = cx->gmv0_ds; wG0[0].dstBinding = 0; wG0[0].descriptorCount = 1;
        wG0[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wG0[0].pImageInfo = g0img;
        wG0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wG0[1].dstSet = cx->gmv0_ds; wG0[1].dstBinding = 1; wG0[1].descriptorCount = 1;
        wG0[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wG0[1].pBufferInfo = &g0buf;
        f.UpdateDescriptorSets(dc->dev, 2, wG0, 0, nullptr);
        VkDescriptorBufferInfo g1buf = g0buf;
        VkDescriptorImageInfo g1img[1] = {{VK_NULL_HANDLE, cx->gmv_view, VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet wG1[2]{};
        wG1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wG1[0].dstSet = cx->gmv1_ds; wG1[0].dstBinding = 0; wG1[0].descriptorCount = 1;
        wG1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wG1[0].pBufferInfo = &g1buf;
        wG1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wG1[1].dstSet = cx->gmv1_ds; wG1[1].dstBinding = 1; wG1[1].descriptorCount = 1;
        wG1[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wG1[1].pImageInfo = g1img;
        f.UpdateDescriptorSets(dc->dev, 2, wG1, 0, nullptr);

        // L2 描述符（cs0_l2 / cs1_l2）
        if (cx->l2_ok) {
            for (int k = 0; k < 2; k++) {
                VkDescriptorImageInfo a0[4];
                a0[0]={cx->sampler, cx->ds_view_l2[1-k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a0[1]={cx->sampler, cx->ds_view_l2[k],   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a0[2]={cx->sampler, cx->mv_view_l2[1-k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a0[3]={VK_NULL_HANDLE, cx->mv_view_l2[k],   VK_IMAGE_LAYOUT_GENERAL};
                VkWriteDescriptorSet w0[4]{};
                for (int b=0;b<4;b++){
                    w0[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w0[b].dstSet=cx->cs0_l2_ds[k]; w0[b].dstBinding=(uint32_t)b; w0[b].descriptorCount=1;
                    w0[b].descriptorType=(b==3)?VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w0[b].pImageInfo=&a0[b];
                }
                f.UpdateDescriptorSets(dc->dev, 4, w0, 0, nullptr);
                VkDescriptorImageInfo a1[6];
                a1[0]={cx->sampler, cx->ds_view[1-k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[1]={cx->sampler, cx->ds_view[k],    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[2]={cx->sampler, cx->mv_viewB,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[3]={VK_NULL_HANDLE, cx->mv_viewB,   VK_IMAGE_LAYOUT_GENERAL};
                a1[4]={cx->sampler, (cx->gmv_view?cx->gmv_view:cx->mv_view), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[5]={cx->sampler, cx->mv_view_l2[1-k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet w1[6]{};
                for (int b=0;b<6;b++){
                    w1[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w1[b].dstSet=cx->cs1_l2_ds[k]; w1[b].dstBinding=(uint32_t)b; w1[b].descriptorCount=1;
                    w1[b].descriptorType=(b==3)?VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w1[b].pImageInfo=&a1[b];
                }
                f.UpdateDescriptorSets(dc->dev, 6, w1, 0, nullptr);
            }
        }
    }

    // 呈现槽位
    if (dc->cmd_pool == VK_NULL_HANDLE) { VKLOGE("init[9/9] 命令池为空"); return false; }
    VkCommandBufferAllocateInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = dc->cmd_pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 4;
    VkCommandBuffer bufs[4];
    if (f.AllocateCommandBuffers(dc->dev, &ca, bufs) != VK_SUCCESS) { VKLOGE("init[9/9] AllocateCommandBuffers 失败"); return false; }
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < 4; i++) {
        cx->slots[i].cmd = bufs[i];
        if (f.CreateSemaphore(dc->dev, &sci, nullptr, &cx->slots[i].acquire_sem) != VK_SUCCESS) { VKLOGE("init[9/9] CreateSemaphore(acquire %d) 失败", i); return false; }
        if (f.CreateSemaphore(dc->dev, &sci, nullptr, &cx->slots[i].done_sem) != VK_SUCCESS) { VKLOGE("init[9/9] CreateSemaphore(done %d) 失败", i); return false; }
        // T2-2：timeline semaphore 槽（CPU 同步免 ResetFences）；创建失败该槽独立回退 fence
        bool ts_ok = g_vk_ts;
        if (ts_ok) {
            VkSemaphoreTypeCreateInfo tci{};
            tci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            tci.initialValue = 0;
            VkSemaphoreCreateInfo tsci = sci;
            tsci.pNext = &tci;
            if (f.CreateSemaphore(dc->dev, &tsci, nullptr, &cx->slots[i].done_ts) != VK_SUCCESS) {
                VKLOGI("槽 %d timeline semaphore 创建失败，回退 fence", i);
                ts_ok = false;
            }
        }
        if (!ts_ok)
            if (f.CreateFence(dc->dev, &fci, nullptr, &cx->slots[i].fence) != VK_SUCCESS) { VKLOGE("init[9/9] CreateFence(%d) 失败", i); return false; }
    }
    VKLOGI("init[9/9] 描述符/槽位完成");

    cx->resources_ok = true;
    cx->has_prev = false;
    cx->cur_slot = 0;
    cx->observe_left = 30;   // 观察期：只积累 prev/cur，不插帧（隔离插帧动作是否干扰游戏）
    // 管线缓存落盘：首次构建后把编译结果写回 app files，供下次启动复用（失败静默）
    if (dc->cache && dc->f.GetPipelineCacheData) {
        size_t sz = 0;
        if (dc->f.GetPipelineCacheData(dc->dev, dc->cache, &sz, nullptr) == VK_SUCCESS && sz > 0) {
            std::vector<uint8_t> data(sz);
            if (dc->f.GetPipelineCacheData(dc->dev, dc->cache, &sz, data.data()) == VK_SUCCESS) {
                vk_cache_write(data.data(), data.size());
                VKLOGI("管线缓存已写回（%zu 字节）", data.size());
            }
        }
    }
    return true;
}

// ---------------- 包装函数 ----------------

static DeviceCtx *find_device(VkDevice dev) {
    auto it = g_devices.find(dev);
    return it == g_devices.end() ? nullptr : it->second;
}

// 槽位空闲判断：timeline 路径读计数器值（免 ResetFences），fence 路径原样 Wait(0)+Reset
static bool slot_idle(DeviceCtx *dc, PresentSlot *s) {
    const DevFns &f = dc->f;
    if (s->done_ts) {
        uint64_t v = 0;
        if (f.GetSemaphoreCounterValue(dc->dev, s->done_ts, &v) != VK_SUCCESS) return false;
        return v >= s->ts_value;
    }
    if (f.WaitForFences(dc->dev, 1, &s->fence, VK_TRUE, 0) == VK_SUCCESS) {
        f.ResetFences(dc->dev, 1, &s->fence);
        return true;
    }
    return false;
}

static DeviceCtx *find_device_by_queue(VkQueue q) {
    for (auto &kv : g_devices)
        if (kv.second->queue_family.count(q)) return kv.second;
    return nullptr;
}

static VkResult VKAPI_CALL my_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR *pInfo,
                                                 const VkAllocationCallbacks *pAlloc, VkSwapchainKHR *pOut) {
    VkResult r = g_CreateSwapchainKHR(device, pInfo, pAlloc, pOut);
    if (r != VK_SUCCESS || !pOut || !*pOut) return r;
    std::lock_guard<std::mutex> lk(g_mtx);
    DeviceCtx *dc = find_device(device);
    if (!dc) return r;
    auto *cx = new SwapchainCtx();
    cx->dc = dc;
    cx->sc = *pOut;
    cx->format = pInfo->imageFormat;
    cx->extent = pInfo->imageExtent;
    cx->usage = pInfo->imageUsage;
    if (!(cx->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        VKLOGI("交换链缺少 TRANSFER_SRC，Vulkan 捕获禁用");
        cx->insert_disabled = true;
    }
    g_swapchains[*pOut] = cx;
    VKLOGI("捕获交换链创建: %ux%u format=%d", cx->extent.width, cx->extent.height, (int)cx->format);
    return r;
}

static void VKAPI_CALL my_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR sc, const VkAllocationCallbacks *pAlloc) {
    // 稳定性红线：游戏渲染线程绝不在这里做任何 GPU 资源销毁/等待。
    // 只把 cx 从活动表移除、标记 dying 并入 zombie 队列，由 worker 线程在确认不再引用后延迟清理
    // （含原始 vkDestroySwapchainKHR）。这样游戏线程的销毁是 O(1) 非阻塞，
    // 即使 worker 正在初始化（管线编译数百 ms）也不会把游戏主线程卡死 -> 黑屏闪退。
    bool handled = false;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_swapchains.find(sc);
        if (it != g_swapchains.end()) {
            SwapchainCtx *cx = it->second;
            g_swapchains.erase(it);
            cx->dying = true;
            g_zombie.push_back(cx);
            handled = true;
        }
    }
    if (handled) g_cv.notify_all();   // 唤醒 worker 尽快延迟清理（同时避免拖延原 destroy）
}

static void VKAPI_CALL my_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue *pQueue) {
    g_GetDeviceQueue(device, family, index, pQueue);
    if (pQueue && *pQueue) {
        std::lock_guard<std::mutex> lk(g_mtx);
        DeviceCtx *dc = find_device(device);
        if (dc) dc->queue_family[*pQueue] = family;
    }
}

// ---------------- 核心：拦截呈现 ----------------

// 在 worker 线程里处理一帧：copy 当前帧到 copy_img，按需 MCI/混合插帧并额外 present。
// 调用方需持有 g_mtx。
static void process_pending_frame(PendingFrame &pf, const McfiConfig &cfg) {
    SwapchainCtx *cx = pf.cx;
    VkQueue queue = pf.queue;
    uint32_t idx = pf.idx;
    DeviceCtx *dc = cx->dc;
    const DevFns &f = dc->f;

    if (!cx->resources_ok || cx->insert_disabled) return;
    if (idx >= cx->images.size()) return;

    static bool g_vk_proc_logged = false;
    if (!g_vk_proc_logged) {
        g_vk_proc_logged = true;
        VKLOGI("worker 开始处理插帧任务（交换链 %ux%u，图像 %zu 张）",
               cx->extent.width, cx->extent.height, cx->images.size());
    }

    // 视频/游戏分开取间隔：B_VIDEO（=video 后端）用视频插帧间隔，其余用游戏插帧间隔
    int gi = g_backend_vk == McfiConfig::B_VIDEO ? cfg.video_interval : cfg.game_interval;
    if (gi < 1) gi = 1;
    if (cx->perf_level >= 2) gi = 2;                    // 降档2：隔帧插
    // 每帧都拷贝当前帧（prev/cur 严格相邻）：观察期、间隔跳帧帧同样拷贝，
    // 避免跨 30 帧/2 帧大间隔合成导致运动错位、跳变；仅降档 3（熔断停插）完全停止
    if (cx->perf_level >= 3) return;
    bool do_insert = cx->has_prev && (pf.present_count % gi) == 0 && cx->images.size() >= 3
                     && cx->observe_left <= 0;
    if (cx->observe_left > 0) {
        cx->observe_left--;
        if (cx->observe_left == 0) VKLOGI("观察期结束，开始插帧");
    }

    PresentSlot *slot = nullptr;
    for (auto &s : cx->slots) {
        if (slot_idle(dc, &s)) { slot = &s; break; }
    }
    if (!slot) {
        // 全部槽忙 = GPU 队列积压（上一帧插帧还没完成）：丢帧防堆积。
        // 连续丢帧说明插帧已拖累游戏 → 性能自适应升档（降低插帧开销）。
        cx->perf_drops++;
        if (cx->perf_drops > 30 && cx->perf_level < 3) {
            cx->perf_level++;
            VKLOGI("Vulkan 性能自适应: 升档%d（GPU 忙连续丢帧，降低插帧开销）", cx->perf_level);
            cx->perf_drops = 0;
        }
        return;
    }
    cx->perf_drops = 0;

    uint32_t gen_idx = 0;
    if (do_insert) {
        VkResult ar = f.AcquireNextImageKHR(dc->dev, cx->sc, 0ull,
                                            slot->acquire_sem, VK_NULL_HANDLE, &gen_idx);
        if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) do_insert = false;
    }

    int cur = cx->cur_slot;
    VkCommandBuffer cmd = slot->cmd;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (f.BeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return;

    // 1. 真实帧 -> copy_img[cur]（拷贝后必须 barrier 回 PRESENT_SRC_KHR）
    image_barrier(dc, cmd, cx->images[idx],
                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    image_barrier(dc, cmd, cx->copy_img[cur],
                  cx->copy_layout_init[cur] ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {cx->extent.width, cx->extent.height, 1};
    f.CmdCopyImage(cmd, cx->images[idx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   cx->copy_img[cur], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image_barrier(dc, cmd, cx->copy_img[cur],
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    cx->copy_layout_init[cur] = true;
    image_barrier(dc, cmd, cx->images[idx],
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_ACCESS_TRANSFER_READ_BIT, 0,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    // 2. MCI：降采样 + 运动估计（仅当插帧 & 选 MCI 模式 & 能力具备）
    bool do_mci = do_insert && cfg.interp_mode == 0 && cx->mci_ok;
    if (do_mci) {
        // 轻量化：默认半径上限 4（候选 9x9=81）；降档1 收紧到 2（候选 5x5=25）；
        // refine 门槛 80（默认 60 不细化 → pass1 sr=0 近乎免费）
        int range = cfg.me_range();
        if (cx->perf_level >= 1) { if (range > 2) range = 2; }
        else if (range > 4) range = 4;
        bool refine = cfg.me_quality >= 80;
        image_barrier(dc, cmd, cx->copy_img[cur],
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        image_barrier(dc, cmd, cx->ds_img[cur],
                      cx->ds_layout_init[cur] ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit bl{};
        bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bl.srcOffsets[0] = {0, 0, 0};
        bl.srcOffsets[1] = {(int32_t)cx->extent.width, (int32_t)cx->extent.height, 1};
        bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bl.dstOffsets[0] = {0, 0, 0};
        bl.dstOffsets[1] = {cx->dsW, cx->dsH, 1};
        f.CmdBlitImage(cmd, cx->copy_img[cur], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       cx->ds_img[cur], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
        image_barrier(dc, cmd, cx->ds_img[cur],
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        cx->ds_layout_init[cur] = true;
        // ---- L2：1/4 → 1/8 再 blit 一半 ----
        if (cx->l2_ok) {
            image_barrier(dc, cmd, cx->ds_img[cur],
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            image_barrier(dc, cmd, cx->ds_img_l2[cur],
                          cx->l2_layout_init[cur] ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          0, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkImageBlit bl2{};
            bl2.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; bl2.srcSubresource.layerCount = 1;
            bl2.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; bl2.dstSubresource.layerCount = 1;
            bl2.srcOffsets[1] = {cx->dsW, cx->dsH, 1};
            bl2.dstOffsets[1] = {cx->dsW2, cx->dsH2, 1};
            f.CmdBlitImage(cmd, cx->ds_img[cur], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           cx->ds_img_l2[cur], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl2, VK_FILTER_LINEAR);
            image_barrier(dc, cmd, cx->ds_img_l2[cur],
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            image_barrier(dc, cmd, cx->ds_img[cur],
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            cx->l2_layout_init[cur] = true;
        }
        image_barrier(dc, cmd, cx->copy_img[cur],
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        // 上一帧降采样若从未初始化（MCI 刚启用 / 首个插帧帧），补一次 blit，避免 ME 读到 UNDEFINED 数据
        if (!cx->ds_layout_init[1 - cur]) {
            image_barrier(dc, cmd, cx->copy_img[1 - cur],
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            image_barrier(dc, cmd, cx->ds_img[1 - cur],
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          0, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            f.CmdBlitImage(cmd, cx->copy_img[1 - cur], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           cx->ds_img[1 - cur], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
            image_barrier(dc, cmd, cx->ds_img[1 - cur],
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            image_barrier(dc, cmd, cx->copy_img[1 - cur],
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            cx->ds_layout_init[1 - cur] = true;
        }
        // prev 的 ds 已是 SHADER_READ_ONLY（上一帧处理时转过）

        // ---- 运动估计：3DRS 双 pass（pass0 粗搜索→mvA；pass1 细化→mvB），同一张 image 绝不混合读写 ----
        // mvB = 上一帧运动场（cs0 读，SHADER_READ_ONLY）；首帧从 UNDEFINED 初始化（丢弃内容）
        if (!cx->mv_b_layout_init) {
            image_barrier(dc, cmd, cx->mv_imgB,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          0, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            cx->mv_b_layout_init = true;
        }
        // mvA：cs0 写前 → GENERAL（每帧全量写，旧内容可丢弃）
        image_barrier(dc, cmd, cx->mv_img,
                      cx->mv_a_ro ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_GENERAL,
                      0, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        cx->mv_a_ro = false;

        // ---- L2 粗搜索：1/8 层 3DRS 写 mvA_L2（先于 L1 cs0）----
        if (cx->l2_ok) {
            image_barrier(dc, cmd, cx->mv_img_l2[cur],
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                          0, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs0_l2_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs0_l2_pl, 0, 1, &cx->cs0_l2_ds[cur], 0, nullptr);
            int32_t R2 = range / 2;
            int32_t pc0l2[4] = {R2, 8, cx->dsW2, cx->dsH2};
            f.CmdPushConstants(cmd, cx->cs0_l2_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc0l2);
            f.CmdDispatch(cmd, (uint32_t)((cx->mvW2 + 7) / 8), (uint32_t)((cx->mvH2 + 7) / 8), 1);
            image_barrier(dc, cmd, cx->mv_img_l2[cur],
                          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        // pass0：3DRS 粗搜索写 mvA（读 mvB 时间预测 + ds）
        f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs0_pipeline);
        f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs0_pl, 0, 1, &cx->cs0_ds[cur], 0, nullptr);
        int32_t pc0[4] = {range, 8, cx->dsW, cx->dsH};
        f.CmdPushConstants(cmd, cx->cs0_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc0);
        f.CmdDispatch(cmd, (uint32_t)((cx->dsW + 7) / 8), (uint32_t)((cx->dsH + 7) / 8), 1);

        // mvA → SHADER_READ_ONLY（cs1 读粗场 / gmv0 读粗场）
        image_barrier(dc, cmd, cx->mv_img,
                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        cx->mv_a_ro = true;

        // ---- GMV（全局运动向量）：L1 粗场直方图 → 归约找主导运动 → 1×1 纹理 ----
        if (cx->gmv0_pipeline && cx->gmv1_pipeline) {
            int binW = 2 * range + 2;
            // 清直方图（GPU 侧）
            f.CmdFillBuffer(cmd, cx->hist_buf, 0, (1024 + 1) * sizeof(int), 0);
            buffer_barrier(dc, cmd, cx->hist_buf,
                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // gmv0：读 mvA → 直方图
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->gmv0_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->gmv0_pl, 0, 1, &cx->gmv0_ds, 0, nullptr);
            int32_t pg0[4] = {range, cx->mvW, cx->mvH, binW};
            f.CmdPushConstants(cmd, cx->gmv0_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pg0);
            uint32_t g0gw = (uint32_t)((cx->mvW + 7) / 8), g0gh = (uint32_t)((cx->mvH + 7) / 8);
            if (g0gw < 1) g0gw = 1;
            if (g0gh < 1) g0gh = 1;
            f.CmdDispatch(cmd, g0gw, g0gh, 1);
            // 直方图 shader 写 → 归约 shader 读
            buffer_barrier(dc, cmd, cx->hist_buf,
                           VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // gmv image → GENERAL（gmv1 写；首帧 UNDEFINED）
            image_barrier(dc, cmd, cx->gmv_img,
                          cx->gmv_layout_init ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_GENERAL,
                          0, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // gmv1：直方图 → 1×1 GMV 图
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->gmv1_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->gmv1_pl, 0, 1, &cx->gmv1_ds, 0, nullptr);
            int32_t pg1[4] = {binW, range, 0, 0};
            f.CmdPushConstants(cmd, cx->gmv1_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pg1);
            f.CmdDispatch(cmd, 1, 1, 1);
            cx->gmv_layout_init = true;
            // GMV 图 → SHADER_READ_ONLY（cs1/me1 候选与插帧 frag 回退读）
            image_barrier(dc, cmd, cx->gmv_img,
                          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }

        // mvB → GENERAL（cs1 写）
        image_barrier(dc, cmd, cx->mv_imgB,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // pass1：细化写 mvB（L2 模式用 cs1_l2 读 L2 粗场放大；否则原 cs1）
        if (cx->l2_ok) {
            int32_t R2 = range / 2;
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs1_l2_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs1_l2_pl, 0, 1, &cx->cs1_l2_ds[cur], 0, nullptr);
            int32_t pc1l2[9] = {range, 8, cx->dsW, cx->dsH, R2, 0, cx->mvW2, cx->mvH2, refine ? 2 : 0};
            f.CmdPushConstants(cmd, cx->cs1_l2_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 36, pc1l2);
            if (cx->cs1_sub) {
                f.CmdDispatch(cmd, (uint32_t)cx->mvW, (uint32_t)cx->mvH, 1);
            } else {
                f.CmdDispatch(cmd, (uint32_t)((cx->dsW + 7) / 8), (uint32_t)((cx->dsH + 7) / 8), 1);
            }
        } else {
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs1_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs1_pl, 0, 1, &cx->cs1_ds[cur], 0, nullptr);
            int32_t pc1[5] = {range, 8, cx->dsW, cx->dsH, refine ? 2 : 0};
            f.CmdPushConstants(cmd, cx->cs1_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, pc1);
            f.CmdDispatch(cmd, (uint32_t)((cx->dsW + 7) / 8), (uint32_t)((cx->dsH + 7) / 8), 1);
        }

        // mvB -> 插帧 frag 只读 + 下一帧 cs0 时间预测读
        image_barrier(dc, cmd, cx->mv_imgB,
                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    if (do_insert) {
        image_barrier(dc, cmd, cx->images[gen_idx],
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        VkRenderPassBeginInfo rb{};
        rb.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rb.renderPass = cx->render_pass;
        rb.framebuffer = cx->framebuffers[gen_idx];
        rb.renderArea = {{0, 0}, cx->extent};
        f.CmdBeginRenderPass(cmd, &rb, VK_SUBPASS_CONTENTS_INLINE);
        f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cx->pipeline);
        VkDescriptorSet set = cx->ds[1 - cur];
        f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cx->pl, 0, 1, &set, 0, nullptr);
        struct { int mode; float strength; float range; float pad0; float dsTexel[2]; float pad1; float pad2; } pcdata;
        pcdata.mode = (cfg.interp_mode == 0 && !cx->mci_ok) ? 1 : cfg.interp_mode;
        pcdata.strength = cfg.strength / 100.0f;
        pcdata.range = (float)cfg.me_range();
        if (cx->perf_level >= 1) { if (pcdata.range > 2.0f) pcdata.range = 2.0f; }
        else if (pcdata.range > 4.0f) pcdata.range = 4.0f;   // 与 ME 搜索半径一致
        pcdata.pad0 = 0.0f;
        pcdata.dsTexel[0] = 1.0f / (float)cx->dsW;
        pcdata.dsTexel[1] = 1.0f / (float)cx->dsH;
        pcdata.pad1 = (float)cfg.smooth; pcdata.pad2 = 0.0f;   // uSmooth 静止保护
        f.CmdPushConstants(cmd, cx->pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, &pcdata);
        f.CmdDraw(cmd, 3, 1, 0, 0);
        f.CmdEndRenderPass(cmd);
    }
    f.EndCommandBuffer(cmd);

    // 同 queue 上游戏 render 已先 submit，GPU 按序执行，无需 wait 游戏 semaphore。
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    if (do_insert) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &slot->acquire_sem;
        VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.pWaitDstStageMask = &ws;
    }
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    // T2-2：timeline 槽每次 submit 显式 signal 递增值（fence 槽维持原 signal 方式）
    VkSemaphore sig[2];
    uint32_t sig_cnt = 0;
    VkTimelineSemaphoreSubmitInfo tsi{};
    uint64_t ts_val = 0;
    if (slot->done_ts) {
        ts_val = ++slot->ts_value;
        sig[sig_cnt++] = slot->done_ts;
        tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &ts_val;
        si.pNext = &tsi;
    }
    if (do_insert) sig[sig_cnt++] = slot->done_sem;
    si.signalSemaphoreCount = sig_cnt;
    si.pSignalSemaphores = sig;
    VkFence submit_fence = slot->done_ts ? VK_NULL_HANDLE : slot->fence;
    if (f.QueueSubmit(queue, 1, &si, submit_fence) == VK_SUCCESS) {
        if (do_insert) {
            static bool g_first_insert = false;
            if (!g_first_insert) {
                g_first_insert = true;
                char msg[160];
                snprintf(msg, sizeof(msg), "pid=%d: Vulkan 插帧首次成功(gen_idx=%u)", getpid(), gen_idx);
                VKLOGI("%s", msg);
                mcfi_send_event(msg);
            }
            VkPresentInfoKHR gpi{};
            gpi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            gpi.waitSemaphoreCount = 1;
            gpi.pWaitSemaphores = &slot->done_sem;
            gpi.swapchainCount = 1;
            gpi.pSwapchains = &cx->sc;
            gpi.pImageIndices = &gen_idx;
            // 时间戳注入：生成帧 A.5 期望上屏于 +1 vsync（pts_align_vk 开关，默认开）
            mcfi_VkPresentTimesInfoGOOGLE pts{};
            mcfi_VkPresentTimeGOOGLE pt{};
            if (g_vk_pts && cfg.pts_align_vk) {
                pt.presentID = 0;
                pt.desiredPresentTime = mcfi_vk_align_pts(mcfi_vk_now_ns(), g_vk_vsync_ns.load(), 1);
                pts.sType = (VkStructureType)MCFI_PRESENT_TIMES_INFO_GOOGLE;
                pts.swapchainCount = 1;
                pts.pTimes = &pt;
                gpi.pNext = &pts;
                g_vk_gen_seq.fetch_add(1);
            }
            VkResult pr = g_QueuePresentKHR(queue, &gpi);
            if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
                VKLOGI("生成帧呈现失败(%d)，暂停插帧", (int)pr);
                cx->insert_disabled = true;
            }
        }
        cx->cur_slot = 1 - cur;
        cx->has_prev = true;
    } else {
        // QueueSubmit 失败（极罕见）：timeline 槽回退已递增的信号值，避免该槽永久"忙"
        if (slot->done_ts) slot->ts_value--;
    }
}

// worker 延迟清理游戏已销毁的 cx。调用前提：worker 线程、且该 cx 已不在 g_swapchains、
// 不在任何待处理任务中（单线程顺序保证），持 g_init_mtx 与可能仍在跑的 INIT 串行。
static void worker_reap_zombie() {
    std::vector<SwapchainCtx *> z;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (g_zombie.empty()) return;
        z.swap(g_zombie);
    }
    std::lock_guard<std::mutex> ilk(g_init_mtx);
    for (SwapchainCtx *cx : z) {
        DeviceCtx *dc = cx->dc;
        destroy_swapchain_resources(cx);
        // 原始 vkDestroySwapchainKHR：仅当该句柄未被新 cx 占用（Vulkan 句柄可能被驱动复用）
        bool reused = false;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            reused = g_swapchains.count(cx->sc) != 0;
        }
        if (!reused && cx->sc) dc->f.DestroySwapchainKHR(dc->dev, cx->sc, nullptr);
        delete cx;
    }
}

static void worker_main() {
    pthread_setname_np(pthread_self(), "MCFI-vk");
    while (true) {
        std::vector<PendingFrame> jobs;
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cv.wait(lk, [] { return g_worker_stop || !g_pending.empty() || !g_zombie.empty(); });
            if (g_worker_stop) break;
            while (!g_pending.empty()) {
                jobs.push_back(g_pending.front());
                g_pending.pop();
            }
        }
        for (auto &pf : jobs) {
            if (g_worker_stop) break;
            if (pf.init_only) {
                // 初始化不持 g_mtx（管线编译可能耗时数百 ms，绝不能阻塞游戏 present）；
                // 处理前按 sc 重新查表取最新 cx：游戏可能已在 INIT 任务入队后销毁该交换链。
                SwapchainCtx *cur = nullptr;
                {
                    std::lock_guard<std::mutex> lk(g_mtx);
                    auto it = g_swapchains.find(pf.sc);
                    if (it == g_swapchains.end()) continue;   // 已销毁，丢弃任务
                    cur = it->second;
                }
                if (cur->dying) continue;
                std::lock_guard<std::mutex> ilk(g_init_mtx);
                if (cur->dying) continue;
                init_swapchain_resources(cur, pf.queue);
                continue;
            }
            McfiConfig cfg;
            mcfi_get_config(&cfg);
            std::lock_guard<std::mutex> lk(g_mtx);
            // 悬垂防护：游戏可能已销毁该 swapchain（worker 处理时按 sc 查最新 cx）
            auto it = g_swapchains.find(pf.sc);
            if (it == g_swapchains.end()) continue;
            pf.cx = it->second;
            if (pf.cx->dying) continue;
            process_pending_frame(pf, cfg);
        }
        worker_reap_zombie();
    }
}

static VkResult VKAPI_CALL my_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pInfo) {
    McfiConfig cfg;
    mcfi_get_config(&cfg);

    static std::atomic<bool> g_vk_first{false};
    if (!g_vk_first.exchange(true)) {
        char msg[128];
        snprintf(msg, sizeof(msg), "pid=%d: Vulkan 接口 vkQueuePresentKHR 首次被调用", getpid());
        VKLOGI("%s", msg);
        mcfi_send_event(msg);
    }

    if (cfg.enabled && pInfo && pInfo->swapchainCount == 1) {
        std::lock_guard<std::mutex> lk(g_mtx);
        DeviceCtx *dc = find_device_by_queue(queue);
        auto sit = g_swapchains.find(pInfo->pSwapchains[0]);
        static int g_miss_logged = 0;
        if ((!dc || sit == g_swapchains.end()) && g_miss_logged < 3) {
            g_miss_logged++;
            VKLOGE("present 但未跟踪到 device/swapchain: dc=%d sc=%p present_count=%ld",
                   (int)(dc != nullptr), (void*)pInfo->pSwapchains[0],
                   sit != g_swapchains.end() ? sit->second->present_count : 0);
        }
        if (dc && sit != g_swapchains.end()) {
            SwapchainCtx *cx = sit->second;
            cx->present_count++;
            if ((cx->present_count % 120) == 0) {
                mcfi_refresh_config();
                mcfi_get_config(&cfg);
            }
            // 初始化未完成：push INIT 任务到 worker（资源创建/管线编译全部异步，游戏线程零 GPU 重活）
            if (cfg.enabled && !cx->insert_disabled && !cx->resources_ok && !cx->init_pending && !cx->dying) {
                if (g_backend_vk == McfiConfig::B_VIDEO && cx->extent.width > 0 && cx->extent.width < 500) {
                    // 视频模式小窗守卫：只对主画面插帧
                    cx->insert_disabled = true;
                } else {
                    cx->init_pending = true;
                    if (!g_worker_started) {
                        g_worker_started = true;
                        g_worker = std::thread(worker_main);
                    }
                    g_pending.push(PendingFrame{cx, cx->sc, queue, 0, (uint64_t)cx->present_count, true});
                    g_cv.notify_one();
                }
            }

            if (cfg.enabled && cx->resources_ok && !cx->insert_disabled && !cx->dying) {
                uint32_t idx = pInfo->pImageIndices[0];
                if (idx < cx->images.size()) {
                    if (!g_worker_started) {
                        g_worker_started = true;
                        g_worker = std::thread(worker_main);
                    }
                    // push 任务，worker 线程在后台做 GPU 操作，present 回调立即返回。
                    // 队列只保留最新帧：丢弃积压旧任务，避免 worker 追不上导致 prev/cur 间隔过大、
                    // 以及 copy 到已被游戏复用的交换链图像。
                    PendingFrame pf{cx, cx->sc, queue, idx, (uint64_t)cx->present_count, false};
                    std::queue<PendingFrame> empty;
                    g_pending.swap(empty);
                    g_pending.push(pf);
                    g_cv.notify_one();
                }
            }
        }
    }

    // 原始呈现（必须走 trampoline，避免递归）
    // 时间戳注入：若 worker 本周期已 present 生成帧，则真实帧 B 期望上屏于 +2 vsync，
    // 与生成帧 A.5(+1 vsync) 各占一个完整周期；否则原样呈现。
    VkResult r;
    McfiConfig vkcfg; mcfi_get_config(&vkcfg);
    if (g_vk_pts && vkcfg.pts_align_vk) {
        static uint64_t s_last_gen = 0;
        uint64_t gen = g_vk_gen_seq.load();
        if (gen != s_last_gen) {
            s_last_gen = gen;
            mcfi_VkPresentTimesInfoGOOGLE pts{};
            mcfi_VkPresentTimeGOOGLE pt{};
            pt.presentID = 0;
            pt.desiredPresentTime = mcfi_vk_align_pts(mcfi_vk_now_ns(), g_vk_vsync_ns.load(), 2);
            pts.sType = (VkStructureType)MCFI_PRESENT_TIMES_INFO_GOOGLE;
            pts.swapchainCount = pInfo->swapchainCount;
            pts.pTimes = &pt;
            VkPresentInfoKHR pi = *pInfo;
            pts.pNext = pi.pNext;
            pi.pNext = &pts;
            mcfi_vk_vsync_sample(mcfi_vk_now_ns());
            r = g_QueuePresentKHR(queue, &pi);
        } else {
            r = g_QueuePresentKHR(queue, pInfo);
        }
    } else {
        r = g_QueuePresentKHR(queue, pInfo);
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        // 不在游戏线程做任何 GPU 清理：游戏自己会销毁并重建交换链，
        // 我们的 my_DestroySwapchainKHR / my_CreateSwapchainKHR 会自动清理旧资源并注册新 cx，
        // 新交换链首次 present 会重新走 worker 异步初始化。
        static int g_ood_logged = 0;
        if (g_ood_logged < 5) {
            g_ood_logged++;
            VKLOGI("present 返回 OUT_OF_DATE/SUBOPTIMAL(%d)，等待游戏重建交换链", (int)r);
        }
    }
    return r;
}

// ---------------- 入口 Hook ----------------

static PFN_vkVoidFunction VKAPI_CALL my_GetDeviceProcAddr(VkDevice device, const char *pName);
static VkResult VKAPI_CALL my_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *pInfo,
                                           const VkAllocationCallbacks *pAlloc, VkDevice *pDevice);
static PFN_vkVoidFunction VKAPI_CALL my_GetInstanceProcAddr(VkInstance instance, const char *pName);

#define LOAD_FN(name) dc->f.name = (PFN_vk##name)g_OrigGetDeviceProcAddr(dev, "vk" #name)

static VkResult VKAPI_CALL my_CreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *pInfo,
                                           const VkAllocationCallbacks *pAlloc, VkDevice *pDevice) {
    // fp16 计算扩展：设备支持 VK_KHR_shader_float16_int8 时追加启用（失败/不支持则保持 fp32，零风险）
    VkDeviceCreateInfo info = *pInfo;
    std::vector<const char *> exts;
    bool want_f16 = false;
    bool want_disp_timing = false;
    bool want_pcc = false;
    bool want_sync2 = false;
    bool want_sgsc = false;
    bool want_sget = false;
    bool want_ts = false;
    if (g_inst != VK_NULL_HANDLE) {
        PFN_vkEnumerateDeviceExtensionProperties ep =
            (PFN_vkEnumerateDeviceExtensionProperties)g_GetInstanceProcAddr(g_inst, "vkEnumerateDeviceExtensionProperties");
        if (ep) {
            uint32_t n = 0;
            if (ep(pd, nullptr, &n, nullptr) == VK_SUCCESS && n > 0) {
                std::vector<VkExtensionProperties> props(n);
                if (ep(pd, nullptr, &n, props.data()) == VK_SUCCESS) {
                    for (auto &p : props) {
                        if (!strcmp(p.extensionName, "VK_KHR_shader_float16_int8")) want_f16 = true;
                        if (!strcmp(p.extensionName, "VK_GOOGLE_display_timing")) want_disp_timing = true;
                        if (!strcmp(p.extensionName, "VK_EXT_pipeline_creation_cache_control")) want_pcc = true;
                        if (!strcmp(p.extensionName, "VK_KHR_synchronization2")) want_sync2 = true;
                        if (!strcmp(p.extensionName, "VK_EXT_subgroup_size_control")) want_sgsc = true;
                        if (!strcmp(p.extensionName, "VK_KHR_shader_subgroup_extended_types")) want_sget = true;
                        if (!strcmp(p.extensionName, "VK_KHR_timeline_semaphore")) want_ts = true;
                    }
                }
            }
        }
    }
    // subgroup 归约能力：查 core supportedOperations（Vulkan 1.1 core），不依赖
    // VK_KHR_shader_subgroup_arithmetic 这个 KHR 扩展名（Adreno 不声明该扩展名，
    // 但 supportedOperations 通常已置 ARITHMETIC_BIT）。
    if (g_inst != VK_NULL_HANDLE) {
        PFN_vkGetPhysicalDeviceProperties2 pGetProps2 =
            (PFN_vkGetPhysicalDeviceProperties2)g_GetInstanceProcAddr(g_inst, "vkGetPhysicalDeviceProperties2");
        if (pGetProps2) {
            VkPhysicalDeviceSubgroupProperties sg{};
            sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            VkPhysicalDeviceSubgroupSizeControlProperties sgc{};
            sgc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
            // T2-3：驱动标识（driverID/name/version），日志留档 + 未来 per-driver 特判依据
            VkPhysicalDeviceDriverProperties dp{};
            dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
            VkPhysicalDeviceProperties2 p2{};
            p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext = &sg;
            sg.pNext = &sgc;
            sgc.pNext = &dp;
            pGetProps2(pd, &p2);
            g_vk_subgroup = (sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
            g_vk_sg_min = sgc.minSubgroupSize;
            g_vk_sg_max = sgc.maxSubgroupSize;
            VKLOGI("subgroup: arithmetic=%d subgroupSize=%u sizeRange=[%u,%u]",
                   (int)g_vk_subgroup, sg.subgroupSize, g_vk_sg_min, g_vk_sg_max);
            VKLOGI("driver: id=%u name=%s info=%s version=0x%x",
                   (unsigned)dp.driverID, dp.driverName[0] ? dp.driverName : "(n/a)",
                   dp.driverInfo[0] ? dp.driverInfo : "(n/a)", (unsigned)p2.properties.driverVersion);
        }
    }
    // 一次性收集：原始扩展 + 我方追加的扩展（避免重复 push 同一扩展名）
    if (pInfo->ppEnabledExtensionNames)
        for (uint32_t i = 0; i < pInfo->enabledExtensionCount; i++) exts.push_back(pInfo->ppEnabledExtensionNames[i]);
    if (want_f16) {
        exts.push_back("VK_KHR_shader_float16_int8");
        g_vk_f16 = true;
        VKLOGI("设备启用 VK_KHR_shader_float16_int8（fp16 计算）");
    }
    if (want_disp_timing) {
        exts.push_back("VK_GOOGLE_display_timing");
        g_vk_pts = true;
        VKLOGI("设备启用 VK_GOOGLE_display_timing（SurfaceFlinger 时间戳注入）");
    }
    // 管线缓存控制：允许我们为管线编译挂 pNext 反馈/控制（配合 VkPipelineCache 跨启动复用）
    VkPhysicalDevicePipelineCreationCacheControlFeatures pcc{};
    // 同步2：vkCmdPipelineBarrier2 单次携带图/缓冲 barrier，stage/access 更细，减少全管线 flush
    VkPhysicalDeviceSynchronization2Features s2{};
    if (want_pcc) {
        exts.push_back("VK_EXT_pipeline_creation_cache_control");
        pcc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES;
        pcc.pipelineCreationCacheControl = VK_TRUE;
        pcc.pNext = (void *)info.pNext;
        info.pNext = &pcc;
        g_vk_cache_ctrl = true;
        VKLOGI("设备启用 VK_EXT_pipeline_creation_cache_control（管线缓存控制）");
    }
    if (want_sync2) {
        exts.push_back("VK_KHR_synchronization2");
        s2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        s2.synchronization2 = VK_TRUE;
        s2.pNext = (void *)info.pNext;
        info.pNext = &s2;
        VKLOGI("设备启用 VK_KHR_synchronization2（Barrier2 路径）");
    }
    // subgroup size 控制：me1_sub 钉 16 lane，消灭 Adreno 64 lane 下的 48 lane 空转
    VkPhysicalDeviceSubgroupSizeControlFeatures sgsc{};
    // fp16 subgroup 归约：me1_sub 的 subgroupAdd 全程跑 fp16，省一半归约带宽
    VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures sget{};
    if (want_sgsc) {
        exts.push_back("VK_EXT_subgroup_size_control");
        sgsc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        sgsc.subgroupSizeControl = VK_TRUE;
        sgsc.computeFullSubgroups = VK_TRUE;
        sgsc.pNext = (void *)info.pNext;
        info.pNext = &sgsc;
        g_vk_sgsc = true;
        VKLOGI("设备启用 VK_EXT_subgroup_size_control（subgroup size 可钉）");
    }
    if (want_sget) {
        exts.push_back("VK_KHR_shader_subgroup_extended_types");
        sget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES;
        sget.shaderSubgroupExtendedTypes = VK_TRUE;
        sget.pNext = (void *)info.pNext;
        info.pNext = &sget;
        g_vk_sget = true;
        VKLOGI("设备启用 VK_KHR_shader_subgroup_extended_types（fp16 subgroup 归约）");
    }
    // timeline semaphore：CPU 侧槽位空闲判断改计数器值（免 ResetFences），无额外设备开销
    VkPhysicalDeviceTimelineSemaphoreFeatures tsf{};
    if (want_ts) {
        exts.push_back("VK_KHR_timeline_semaphore");
        tsf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        tsf.timelineSemaphore = VK_TRUE;
        tsf.pNext = (void *)info.pNext;
        info.pNext = &tsf;
        g_vk_ts = true;
        VKLOGI("设备启用 VK_KHR_timeline_semaphore（槽位 CPU 同步）");
    }
    info.enabledExtensionCount = (uint32_t)exts.size();
    info.ppEnabledExtensionNames = exts.data();
    VkResult r = g_CreateDevice(pd, &info, pAlloc, pDevice);
    if (r == VK_SUCCESS && pDevice && *pDevice) {
        std::lock_guard<std::mutex> lk(g_mtx);
        VkDevice dev = *pDevice;
        auto *dc = new DeviceCtx();
        dc->dev = dev;
        dc->pd = pd;
        LOAD_FN(CreateCommandPool); LOAD_FN(DestroyCommandPool);
        LOAD_FN(AllocateCommandBuffers); LOAD_FN(BeginCommandBuffer);
        LOAD_FN(EndCommandBuffer); LOAD_FN(ResetCommandBuffer);
        LOAD_FN(CmdPipelineBarrier); LOAD_FN(CmdCopyImage);
        LOAD_FN(CmdPipelineBarrier2);
        LOAD_FN(CmdBlitImage); LOAD_FN(CmdDispatch);
        LOAD_FN(CmdBeginRenderPass); LOAD_FN(CmdEndRenderPass);
        LOAD_FN(CreateBuffer); LOAD_FN(GetBufferMemoryRequirements); LOAD_FN(BindBufferMemory);
        LOAD_FN(DestroyBuffer); LOAD_FN(CmdFillBuffer);
        LOAD_FN(CmdBindPipeline); LOAD_FN(CmdBindDescriptorSets);
        LOAD_FN(CmdPushConstants); LOAD_FN(CmdDraw);
        LOAD_FN(QueueSubmit); LOAD_FN(QueuePresentKHR);
        LOAD_FN(AcquireNextImageKHR);
        LOAD_FN(CreateSemaphore); LOAD_FN(DestroySemaphore);
        LOAD_FN(CreateFence); LOAD_FN(DestroyFence);
        LOAD_FN(WaitForFences); LOAD_FN(ResetFences);
        LOAD_FN(GetSemaphoreCounterValue);
        LOAD_FN(CreateImage); LOAD_FN(DestroyImage);
        LOAD_FN(GetImageMemoryRequirements);
        LOAD_FN(GetImageMemoryRequirements2); LOAD_FN(GetBufferMemoryRequirements2);
        LOAD_FN(AllocateMemory); LOAD_FN(FreeMemory); LOAD_FN(BindImageMemory);
        LOAD_FN(BindImageMemory2); LOAD_FN(BindBufferMemory2);
        LOAD_FN(CreateImageView); LOAD_FN(DestroyImageView);
        LOAD_FN(CreateRenderPass); LOAD_FN(DestroyRenderPass);
        LOAD_FN(CreateFramebuffer); LOAD_FN(DestroyFramebuffer);
        LOAD_FN(CreateShaderModule); LOAD_FN(DestroyShaderModule);
        LOAD_FN(CreatePipelineLayout); LOAD_FN(DestroyPipelineLayout);
        LOAD_FN(CreateGraphicsPipelines); LOAD_FN(CreateComputePipelines); LOAD_FN(DestroyPipeline);
        LOAD_FN(CreatePipelineCache); LOAD_FN(DestroyPipelineCache); LOAD_FN(GetPipelineCacheData);
        LOAD_FN(CreateDescriptorSetLayout); LOAD_FN(DestroyDescriptorSetLayout);
        LOAD_FN(CreateDescriptorPool); LOAD_FN(DestroyDescriptorPool);
        LOAD_FN(AllocateDescriptorSets); LOAD_FN(UpdateDescriptorSets);
        LOAD_FN(CreateSampler); LOAD_FN(DestroySampler);
        LOAD_FN(GetSwapchainImagesKHR);
        LOAD_FN(CreateSwapchainKHR); LOAD_FN(DestroySwapchainKHR);
        LOAD_FN(GetDeviceQueue); LOAD_FN(DestroyDevice);
        dc->sync2 = want_sync2 && dc->f.CmdPipelineBarrier2 != nullptr;
        bool missing = !dc->f.QueuePresentKHR || !dc->f.CreateSwapchainKHR || !dc->f.AcquireNextImageKHR;
        if (missing) {
            VKLOGE("设备函数加载不全，Vulkan 补帧对该设备禁用");
            delete dc;
        } else {
            // 管线缓存：读入上次落盘数据（损坏/缺失则空缓存），进程内后续创建即可复用编译结果
            if (dc->f.CreatePipelineCache) {
                VkPipelineCacheCreateInfo cci{};
                cci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
                auto cached = vk_cache_read();
                if (!cached.empty()) {
                    cci.initialDataSize = cached.size();
                    cci.pInitialData = (void *)cached.data();
                }
                if (dc->f.CreatePipelineCache(dev, &cci, nullptr, &dc->cache) != VK_SUCCESS) {
                    VKLOGE("CreatePipelineCache 失败（用空缓存重试）");
                    dc->cache = VK_NULL_HANDLE;
                    if (!cached.empty()) {
                        cci.initialDataSize = 0;
                        cci.pInitialData = nullptr;
                        dc->f.CreatePipelineCache(dev, &cci, nullptr, &dc->cache);
                    }
                } else {
                    VKLOGI("管线缓存就绪（命中 %zu 字节落盘数据）", cached.size());
                }
            }
            g_devices[dev] = dc;
            VKLOGI("捕获 VkDevice 创建");
        }
    }
    return r;
}

static PFN_vkVoidFunction VKAPI_CALL my_GetDeviceProcAddr(VkDevice device, const char *pName) {
    if (pName) {
        if (!strcmp(pName, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)my_CreateSwapchainKHR;
        if (!strcmp(pName, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)my_DestroySwapchainKHR;
        if (!strcmp(pName, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)my_QueuePresentKHR;
        if (!strcmp(pName, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)my_GetDeviceQueue;
        if (!strcmp(pName, "vkCreateDevice")) return (PFN_vkVoidFunction)my_CreateDevice;
        if (!strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)my_GetDeviceProcAddr;
    }
    return g_OrigGetDeviceProcAddr(device, pName);
}

static PFN_vkVoidFunction VKAPI_CALL my_GetInstanceProcAddr(VkInstance instance, const char *pName) {
    g_inst = instance;
    if (pName) {
        if (!strcmp(pName, "vkCreateDevice")) return (PFN_vkVoidFunction)my_CreateDevice;
        if (!strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)my_GetDeviceProcAddr;
        if (!strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)my_GetInstanceProcAddr;
        // 部分引擎（如 Unity）通过 instance 级查询直接取 device 级函数指针，
        // 不拦的话这些调用完全绕过我们的包装。
        if (!strcmp(pName, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)my_QueuePresentKHR;
        if (!strcmp(pName, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)my_CreateSwapchainKHR;
        if (!strcmp(pName, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)my_DestroySwapchainKHR;
        if (!strcmp(pName, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)my_GetDeviceQueue;
    }
    return g_OrigGetInstanceProcAddr(instance, pName);
}

extern "C" void mcfi_vk_install(int log_level, int backend, const char *pkg) {
    g_log_level = log_level;
    g_backend_vk = backend;
    g_vk_pkg = pkg ? pkg : "";
    g_vk_lib = dlopen("libvulkan.so", RTLD_NOW);
    if (!g_vk_lib) {
        VKLOGI("libvulkan.so 不可用，跳过 Vulkan 后端");
        return;
    }
    g_GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)dlsym(g_vk_lib, "vkGetDeviceProcAddr");
    g_CreateDevice = (PFN_vkCreateDevice)dlsym(g_vk_lib, "vkCreateDevice");
    g_GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(g_vk_lib, "vkGetInstanceProcAddr");
    g_GetPhysDevMemProps = (PFN_vkGetPhysicalDeviceMemoryProperties)dlsym(g_vk_lib, "vkGetPhysicalDeviceMemoryProperties");
    g_GetPhysDevFmtProps = (PFN_vkGetPhysicalDeviceFormatProperties)dlsym(g_vk_lib, "vkGetPhysicalDeviceFormatProperties");
    g_CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)dlsym(g_vk_lib, "vkCreateSwapchainKHR");
    g_DestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)dlsym(g_vk_lib, "vkDestroySwapchainKHR");
    g_QueuePresentKHR = (PFN_vkQueuePresentKHR)dlsym(g_vk_lib, "vkQueuePresentKHR");
    g_GetDeviceQueue = (PFN_vkGetDeviceQueue)dlsym(g_vk_lib, "vkGetDeviceQueue");
    if (!g_GetDeviceProcAddr || !g_CreateDevice || !g_GetInstanceProcAddr || !g_GetPhysDevMemProps ||
        !g_CreateSwapchainKHR || !g_DestroySwapchainKHR || !g_QueuePresentKHR || !g_GetDeviceQueue) {
        VKLOGE("Vulkan 符号解析失败");
        return;
    }
    // 只 hook vkGetInstanceProcAddr 这一个入口。星铁/Unity 通过它取所有 Vulkan 函数指针，
    // 我们在 my_GetInstanceProcAddr 里对 vkCreateDevice/vkGetDeviceProcAddr 返回包装，
    // 其余原样转发。不直接 hook present/swapchain 等导出符号（星铁不走那条路）。
    g_OrigGetDeviceProcAddr = g_GetDeviceProcAddr;
    g_OrigCreateDevice = nullptr;
    int fail = 0;
    fail |= DobbyHook((void *)g_GetInstanceProcAddr, (void *)my_GetInstanceProcAddr, (void **)&g_OrigGetInstanceProcAddr);
    if (fail) {
        VKLOGE("vkGetInstanceProcAddr hook 安装失败(%d)", fail);
        return;
    }
    VKLOGI("Vulkan 后端 hook 安装完成（hook vkGetInstanceProcAddr，MCI 运动补偿已内置）");
}
