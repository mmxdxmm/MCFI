// MCFI Zygisk 入口
// - preAppSpecialize: 通过 root companion 通道获取配置，判断目标包名，命中则安装补帧 hook
// - companion: 以 root 权限运行，中继 app 进程与 MCFI 守护进程之间的配置请求

#include <sys/types.h>
#include <zygisk.hpp>
#include <jni.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <android/log.h>
#include <string>
#include <cstdlib>
#include <cstring>

#include "config.h"

#define LOG_TAG "MCFI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using zygisk::Api;
using zygisk::AppSpecializeArgs;

extern "C" void mcfi_payload_init(const char *config_text, int companion_fd);
extern "C" void mcfi_set_pkg(const char *pkg);
extern "C" void mcfi_set_backend(int be);

// ---------------- 工具 ----------------

static std::string read_file(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return "";
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) out.append(buf, (size_t)n);
    close(fd);
    return out;
}

static int connect_daemon() {
    int fd = socket(AF_UNIX, SOCK_STREAM | O_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un un{};
    un.sun_family = AF_UNIX;
    strncpy(un.sun_path, MCFI_SOCK, sizeof(un.sun_path) - 1);
    if (connect(fd, (sockaddr *)&un, sizeof(un)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static bool write_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= (size_t)n;
    }
    return true;
}

// 通过 companion fd 请求配置，@EOF@ 结尾
static bool request_config_via_fd(int fd, std::string &out, int timeout_ms) {
    if (fd < 0) return false;
    pollfd p{fd, POLLOUT, 0};
    if (poll(&p, 1, timeout_ms) <= 0) return false;
    if (!write_all(fd, "CFG\n", 4)) return false;
    char buf[4096];
    for (int i = 0; i < 32; i++) {
        p.fd = fd; p.events = POLLIN; p.revents = 0;
        if (poll(&p, 1, timeout_ms) <= 0) break;
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0) break;
        out.append(buf, (size_t)n);
        size_t e = out.find("@EOF@");
        if (e != std::string::npos) {
            out = out.substr(0, e);
            return true;
        }
    }
    return false;
}

// ---------------- Root companion：中继 app <-> 守护进程 ----------------

static void companion_entry(int client_fd) {
    // 守护进程可能还没起来（开机早期），重试几次
    int daemon_fd = -1;
    for (int i = 0; i < 10 && daemon_fd < 0; i++) {
        daemon_fd = connect_daemon();
        if (daemon_fd < 0) usleep(200 * 1000);
    }
    if (daemon_fd < 0) {
        // 兜底：直接读配置文件应答（companion 拥有 root 权限）
        pollfd p{client_fd, POLLIN, 0};
        while (poll(&p, 1, -1) > 0) {
            char buf[64];
            ssize_t n = read(client_fd, buf, sizeof(buf));
            if (n <= 0) break;
            std::string cfg = read_file(MCFI_CONFIG);
            if (cfg.empty()) cfg = read_file(MCFI_CONFIG_MIRROR);
            if (cfg.empty()) cfg = McfiConfig().dump();
            cfg += "@EOF@";
            if (!write_all(client_fd, cfg.data(), cfg.size())) break;
        }
        close(client_fd);
        return;
    }

    // 双向中继
    pollfd fds[2] = {{client_fd, POLLIN, 0}, {daemon_fd, POLLIN, 0}};
    char buf[8192];
    for (;;) {
        if (poll(fds, 2, -1) <= 0) break;
        bool dead = false;
        for (int i = 0; i < 2; i++) {
            if (fds[i].revents & POLLIN) {
                ssize_t n = read(fds[i].fd, buf, sizeof(buf));
                if (n <= 0) { dead = true; break; }
                if (!write_all(fds[1 - i].fd, buf, (size_t)n)) { dead = true; break; }
            }
            if (fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) { dead = true; }
        }
        if (dead) break;
    }
    close(daemon_fd);
    close(client_fd);
}

REGISTER_ZYGISK_COMPANION(companion_entry)

// ---------------- 模块主体 ----------------

class McfiModule : public zygisk::ModuleBase {
public:
    void onLoad(Api *api, JNIEnv *env) override {
        api_ = api;
        env_ = env;
        // 此函数对所有进程执行，不在此打印（避免无关进程日志噪音）
    }

    void preAppSpecialize(AppSpecializeArgs *args) override {
        if (!args || !args->app_data_dir) return;

        const char *dir = env_->GetStringUTFChars(args->app_data_dir, nullptr);
        std::string pkg = mcfi_pkg_from_data_dir(dir);
        env_->ReleaseStringUTFChars(args->app_data_dir, dir);
        if (pkg.empty()) return;

        int companion_fd = api_->connectCompanion();
        std::string cfg_text;
        bool ok = request_config_via_fd(companion_fd, cfg_text, 1000);
        if (!ok) {
            // 拿不到配置就尝试本地兜底路径（root 阶段的子进程可能可读）
            cfg_text = read_file(MCFI_CONFIG);
            if (cfg_text.empty()) cfg_text = read_file(MCFI_CONFIG_MIRROR);
        }
        if (cfg_text.empty()) {
            LOGI("未取到配置(守护进程未就绪?)，跳过: %s", pkg.c_str());
            if (companion_fd >= 0) close(companion_fd);
            return;
        }

        McfiConfig cfg = McfiConfig::parse(cfg_text);
        if (!cfg.enabled || !cfg.package_allowed(pkg)) {
            // 无关应用静默跳过（不打印，避免日志 I/O 噪音）
            if (companion_fd >= 0) close(companion_fd);
            return;
        }

        McfiConfig::Backend be = cfg.backend_for(pkg);
        const char *be_name = be == McfiConfig::B_GLES ? "gles"
                            : be == McfiConfig::B_VULKAN ? "vulkan"
                            : be == McfiConfig::B_VIDEO ? "video"
                            : be == McfiConfig::B_OFF ? "off" : "auto";
        LOGI("目标进程命中: %s，后端=%s", pkg.c_str(), be_name);
        mcfi_set_pkg(pkg.c_str());
        mcfi_set_backend((int)be);
        mcfi_payload_init(cfg_text.c_str(), companion_fd);
    }

private:
    Api *api_ = nullptr;
    JNIEnv *env_ = nullptr;
};

REGISTER_ZYGISK_MODULE(McfiModule)
