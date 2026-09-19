// MCFI 共享配置定义
// 运动补偿插帧模块（Motion-Compensated Frame Interpolation）
#pragma once

#include <string>
#include <cstdlib>
#include <cstring>

#define MCFI_DATA_DIR   "/data/adb/modules/mcfi"
#define MCFI_CONFIG     MCFI_DATA_DIR "/config.conf"
#define MCFI_CONFIG_MIRROR "/data/local/tmp/mcfi_config.conf"
#define MCFI_SOCK       MCFI_DATA_DIR "/daemon.sock"
#define MCFI_PANEL      MCFI_DATA_DIR "/mcfi/panel/index.html"
#define MCFI_MODULE_PROP MCFI_DATA_DIR "/module.prop"
#define MCFI_DAEMON_BIN MCFI_DATA_DIR "/mcfi/bin/mcfid"
#define MCFI_DAEMON_LOG MCFI_DATA_DIR "/daemon.log"
#define MCFI_APPLOG     MCFI_DATA_DIR "/app.log"   // 各应用生效事件（环形，保留最新若干行）
#define MCFI_RUNTIME    MCFI_DATA_DIR "/runtime"   // app 进程直接写事件文件的目录（0777）

struct McfiConfig {
    // 每个应用指定的渲染后端
    enum Backend {
        B_DEFAULT = 0,  // 不写后缀：GLES+Vulkan 都 hook
        B_GLES,         // =gles
        B_VULKAN,       // =vulkan
        B_VIDEO,        // =video（GLES 视频优化路径）
        B_OFF           // =off（完全不 hook）
    };

    bool enabled = false;          // 总开关
    int  target_mode = 1;          // 0=全部应用 1=白名单 2=黑名单
    std::string targets;           // 逗号分隔包名，可带 =gles/=vulkan/=video/=off
    int  interp_mode = 0;          // 0=运动补偿插帧(MCI，推荐) 1=普通混合(兼容) 2=运动自适应混合
    int  strength = 60;            // MCI 遮挡/拖影抑制强度 0~100
    int  smooth = 60;             // 平滑强度 0~100（合成端静止保护 + 运动估计端时域收缩）
    int  gen_interval = 1;         // 每 N 个真实帧插入 1 个生成帧
    int  me_quality = 60;          // 运动估计质量/开销 0~100（决定搜索半径与细化级数）
    int  panel_port = 4400;        // 控制面板端口（被占用自动顺延）
    int  log_level = 1;            // 0=关闭 1=普通
    int  vk_mci = 0;               // Vulkan 运动补偿开关：0=仅普通混合（部分驱动在 MCI 资源创建时崩溃，默认关闭保稳定）1=开启 MCI
    int  pts_enable = 1;           // SurfaceFlinger 时间戳注入：1=让生成帧/真实帧各占一个 vsync（默认开）0=关闭

    static inline std::string trim(const std::string &s) {
        size_t b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    // 解析单个条目 "pkg=backend" -> 返回包名；backend_out 输出后端枚举
    static std::string parse_item(const std::string &item, Backend &be) {
        std::string s = trim(item);
        be = B_DEFAULT;
        size_t eq = s.find('=');
        if (eq == std::string::npos) return s;
        std::string pkg = trim(s.substr(0, eq));
        std::string b = trim(s.substr(eq + 1));
        if (b == "gles")   be = B_GLES;
        else if (b == "vulkan") be = B_VULKAN;
        else if (b == "video")  be = B_VIDEO;
        else if (b == "off")    be = B_OFF;
        return pkg;
    }

    // 在 targets 中查找 pkg，返回其指定的后端；没写后缀返回 B_DEFAULT
    Backend backend_for(const std::string &pkg) const {
        Backend found = B_DEFAULT;
        bool matched = false;
        size_t pos = 0;
        while (pos <= targets.size()) {
            size_t c = targets.find(',', pos);
            std::string item = targets.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
            Backend be;
            std::string p = parse_item(item, be);
            if (p == pkg || p == "*") { found = be; matched = true; }
            if (c == std::string::npos) break;
            pos = c + 1;
        }
        return matched ? found : B_DEFAULT;
    }

    // 该包名是否在 targets 中且指定了 =off
    bool is_off(const std::string &pkg) const {
        return backend_for(pkg) == B_OFF;
    }

    bool package_allowed(const std::string &pkg) const {
        if (pkg.empty()) return false;
        if (pkg == "android" || pkg == "system" || pkg.rfind("com.android.systemui", 0) == 0 ||
            pkg.rfind("com.android.internal", 0) == 0)
            return false;
        bool listed = pkg_in_list(pkg);
        switch (target_mode) {
            case 0: return !is_off(pkg);       // 全部应用，除了显式 =off
            case 2: return !listed;            // 黑名单
            default: return listed && !is_off(pkg); // 白名单，=off 的排除
        }
    }

private:
    bool pkg_in_list(const std::string &pkg) const {
        size_t pos = 0;
        while (pos <= targets.size()) {
            size_t c = targets.find(',', pos);
            Backend be;
            std::string p = parse_item(targets.substr(pos, c == std::string::npos ? std::string::npos : c - pos), be);
            if (!p.empty() && (p == pkg || p == "*")) return true;
            if (c == std::string::npos) break;
            pos = c + 1;
        }
        return false;
    }

public:
    // 运动估计搜索半径（1/4 分辨率下的像素数）：由 me_quality 映射，4 ~ 12
    int me_range() const {
        int q = me_quality < 0 ? 0 : (me_quality > 100 ? 100 : me_quality);
        return 4 + q * 8 / 100;
    }
    // 是否启用细化级（第二遍小范围搜索）
    bool me_refine() const { return me_quality >= 45; }

    static McfiConfig parse(const std::string &text) {
        McfiConfig c;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t nl = text.find('\n', pos);
            std::string line = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
            pos = (nl == std::string::npos) ? text.size() : nl + 1;
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = trim(line.substr(0, eq));
            std::string v = trim(line.substr(eq + 1));
            if (k == "enabled")            c.enabled = (v == "1" || v == "true");
            else if (k == "target_mode")   c.target_mode = atoi(v.c_str());
            else if (k == "targets")       c.targets = v;
            else if (k == "interp_mode")   c.interp_mode = atoi(v.c_str());
            else if (k == "strength")      c.strength = atoi(v.c_str());
            else if (k == "smooth")        c.smooth = atoi(v.c_str());
            else if (k == "gen_interval")  c.gen_interval = atoi(v.c_str());
            else if (k == "me_quality")    c.me_quality = atoi(v.c_str());
            else if (k == "panel_port")    c.panel_port = atoi(v.c_str());
            else if (k == "log_level")     c.log_level = atoi(v.c_str());
            else if (k == "vk_mci")        c.vk_mci = atoi(v.c_str());
            else if (k == "pts_enable")    c.pts_enable = atoi(v.c_str());
        }
        if (c.strength < 0) c.strength = 0;
        if (c.strength > 100) c.strength = 100;
        if (c.smooth < 0) c.smooth = 0;
        if (c.smooth > 100) c.smooth = 100;
        if (c.gen_interval < 1) c.gen_interval = 1;
        if (c.me_quality < 0) c.me_quality = 0;
        if (c.me_quality > 100) c.me_quality = 100;
        if (c.interp_mode < 0 || c.interp_mode > 2) c.interp_mode = 0;
        if (c.panel_port <= 0 || c.panel_port > 65535) c.panel_port = 4400;
        if (c.vk_mci < 0 || c.vk_mci > 1) c.vk_mci = 0;
        if (c.pts_enable < 0 || c.pts_enable > 1) c.pts_enable = 1;
        return c;
    }

    std::string dump() const {
        char buf[4096];
        snprintf(buf, sizeof(buf),
                 "# MCFI 配置文件（由控制面板维护，可手动编辑）\n"
                 "enabled=%d\n"
                 "target_mode=%d\n"
                 "targets=%s\n"
                 "interp_mode=%d\n"
                 "strength=%d\n"
                 "smooth=%d\n"
                 "gen_interval=%d\n"
                 "me_quality=%d\n"
                 "panel_port=%d\n"
                 "log_level=%d\n"
                 "vk_mci=%d\n"
                 "pts_enable=%d\n",
                 enabled ? 1 : 0, target_mode, targets.c_str(), interp_mode,
                 strength, smooth, gen_interval, me_quality, panel_port, log_level, vk_mci, pts_enable);
        return buf;
    }
};

// 从 app_data_dir 提取包名: /data/user/0/com.foo.bar -> com.foo.bar
static inline std::string mcfi_pkg_from_data_dir(const char *dir) {
    if (!dir || !*dir) return "";
    std::string s(dir);
    size_t slash = s.find_last_of('/');
    if (slash == std::string::npos) return "";
    return s.substr(slash + 1);
}
