// MCFI 守护进程（root，由 service.sh 启动）
// - 监听 unix socket：应答 app 进程（经 zygisk companion 中继）的配置请求
// - 监听 127.0.0.1:TCP：提供控制面板 HTTP 服务

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cerrno>
#include <cctype>
#include <ctime>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <dirent.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <android/log.h>

#include "config.h"

#define LOG_TAG "MCFID"

// 文件日志：同时写 logcat 和 daemon.log（带时间戳），解决旧版 daemon.log 恒为空的问题
static FILE *g_logf = nullptr;
static int g_daemon_log = 0;   // 调试日志开关（默认关，load_config 后按配置更新；0=logcat 只留 ERROR）

static void mcfi_fprintln(int prio, const char *fmt, va_list ap) {
    // 1) logcat：INFO 受调试日志开关控制（关闭后 logcat 不再刷 MCFID），ERROR 始终打
    if (!(prio == ANDROID_LOG_INFO && g_daemon_log <= 0))
        __android_log_vprint(prio, LOG_TAG, fmt, ap);
    // 2) 文件（daemon.log 始终写，供排障）
    if (!g_logf) return;
    time_t t = time(nullptr);
    struct tm tm;
    localtime_r(&t, &tm);
    flockfile(g_logf);
    fprintf(g_logf, "%02d:%02d:%02d ", tm.tm_hour, tm.tm_min, tm.tm_sec);
    vfprintf(g_logf, fmt, ap);
    fputc('\n', g_logf);
    fflush(g_logf);
    funlockfile(g_logf);
}

// 真正的可变参数函数：va_start 必须在 variadic 函数体内，不能放在宏里直接展开
static void mcfi_log(int prio, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    mcfi_fprintln(prio, fmt, ap);
    va_end(ap);
}

#define LOGI(fmt, ...) mcfi_log(ANDROID_LOG_INFO, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) mcfi_log(ANDROID_LOG_ERROR, fmt, ##__VA_ARGS__)

// ---- 应用生效事件环形日志 ----
// 由 app 进程经 companion 上报，daemon 落盘到 app.log，保留最新 MCFI_APPLOG_KEEP 行
#define MCFI_APPLOG_KEEP 200

static std::string read_file_str(const char *path);

static void append_applog(const std::string &line) {
    // 加时间戳
    time_t t = time(nullptr);
    struct tm tm;
    localtime_r(&t, &tm);
    char ts[32];
    snprintf(ts, sizeof(ts), "%02d:%02d:%02d ", tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::string entry = ts + line + "\n";

    // 读回旧内容，保留尾部 (KEEP-1) 行，再追加新行
    std::string old = read_file_str(MCFI_APPLOG);
    std::string kept;
    if (!old.empty()) {
        // 按行切分，保留尾部 MCFI_APPLOG_KEEP-1 行
        std::vector<std::string> lines;
        size_t pos = 0;
        while (pos < old.size()) {
            size_t nl = old.find('\n', pos);
            std::string l = old.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
            if (!l.empty()) lines.push_back(l);
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        size_t start = lines.size() > (size_t)(MCFI_APPLOG_KEEP - 1) ? lines.size() - (MCFI_APPLOG_KEEP - 1) : 0;
        for (size_t i = start; i < lines.size(); i++) kept += lines[i] + "\n";
    }
    kept += entry;

    FILE *f = fopen(MCFI_APPLOG, "wb");
    if (f) { fwrite(kept.data(), 1, kept.size(), f); fclose(f); chmod(MCFI_APPLOG, 0644); }
    LOGI("[app] %s", line.c_str());
}

static std::string g_config_text;
static int g_real_port = 0;
static bool g_panel_ok = false;

static std::string read_file_str(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return "";
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

static void load_config() {
    std::string s = read_file_str(MCFI_CONFIG);
    if (s.empty()) {
        s = McfiConfig().dump();   // 默认配置（enabled=0）
        FILE *f = fopen(MCFI_CONFIG, "wb");
        if (f) { fputs(s.c_str(), f); fclose(f); chmod(MCFI_CONFIG, 0664); }
    }
    g_config_text = s;
    g_daemon_log = McfiConfig::parse(s).log_level;   // 调试日志开关跟随配置
    // 镜像一份到 app 侧可能可读的位置（兜底通道）
    FILE *m = fopen(MCFI_CONFIG_MIRROR, "wb");
    if (m) { fputs(s.c_str(), m); fclose(m); chmod(MCFI_CONFIG_MIRROR, 0644); }
}

static void update_status(int real_port, bool panel_ok);

static void save_config(const std::string &text) {
    std::string tmp = std::string(MCFI_CONFIG) + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) { LOGE("写配置失败: %s", strerror(errno)); return; }
    fwrite(text.data(), 1, text.size(), f);
    fclose(f);
    chmod(tmp.c_str(), 0664);
    rename(tmp.c_str(), MCFI_CONFIG);
    g_config_text = text;
    g_daemon_log = McfiConfig::parse(text).log_level;   // 面板保存后立即生效
    FILE *m = fopen(MCFI_CONFIG_MIRROR, "wb");
    if (m) { fwrite(text.data(), 1, text.size(), m); fclose(m); chmod(MCFI_CONFIG_MIRROR, 0644); }
    LOGI("配置已保存 (%zu 字节)", text.size());
    update_status(g_real_port, g_panel_ok);
}

// ---------------- 连接会话 ----------------

struct Conn {
    int fd = -1;
    bool http = false;
    std::string rx;
    size_t rx_headers_end = std::string::npos;
    size_t content_length = 0;
    bool headers_done = false;
};

static std::vector<Conn> g_conns;

static bool send_all(int fd, const char *p, size_t len) {
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

static void close_conn(Conn &c) {
    if (c.fd >= 0) close(c.fd);
    c.fd = -1;
}

// ---------------- Unix socket 协议（供 companion 中继） ----------------
// 客户端按行发送命令: CFG -> 返回配置文本 + @EOF@

static void handle_ctrl_data(Conn &c) {
    size_t pos;
    while ((pos = c.rx.find('\n')) != std::string::npos) {
        std::string line = c.rx.substr(0, pos);
        c.rx.erase(0, pos + 1);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        std::string resp;
        if (line == "CFG" || line.empty()) {
            load_config();
            resp = g_config_text + "@EOF@";
        } else if (line == "PING") {
            resp = "PONG@EOF@";
        } else if (line.rfind("EVT ", 0) == 0) {
            // 应用上报生效事件：EVT <文本>
            std::string ev = line.substr(4);
            while (!ev.empty() && (ev.back() == '\r' || ev.back() == ' ')) ev.pop_back();
            append_applog(ev);
            resp = "OK@EOF@";
        } else {
            resp = "ERR@EOF@";
        }
        if (!send_all(c.fd, resp.data(), resp.size())) { close_conn(c); return; }
    }
}

// ---------------- HTTP ----------------

static void http_respond(Conn &c, int code, const char *ctype, const std::string &body, bool keep_alive) {
    char head[512];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                     "Cache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\n"
                     "Connection: %s\r\n\r\n",
                     code, ctype, body.size(), keep_alive ? "keep-alive" : "close");
    if (!send_all(c.fd, head, (size_t)n) || !send_all(c.fd, body.data(), body.size())) {
        close_conn(c);
        return;
    }
    if (!keep_alive) close_conn(c);
}

static void handle_http(Conn &c) {
    if (!c.headers_done) {
        size_t e = c.rx.find("\r\n\r\n");
        if (e == std::string::npos) {
            if (c.rx.size() > 65536) close_conn(c);
            return;
        }
        std::string head = c.rx.substr(0, e);
        c.rx_headers_end = e + 4;
        c.content_length = 0;
        // 解析 Content-Length
        std::string lower = head;
        for (auto &ch : lower) ch = (char)tolower((unsigned char)ch);
        size_t p = lower.find("content-length:");
        if (p != std::string::npos) {
            p += strlen("content-length:");
            c.content_length = (size_t)strtoul(head.c_str() + p, nullptr, 10);
        }
        c.headers_done = true;
    }
    if (c.rx.size() < c.rx_headers_end + c.content_length) return; // 等待 body

    std::string head = c.rx.substr(0, c.rx_headers_end);
    std::string body = c.rx.substr(c.rx_headers_end, c.content_length);
    c.rx.clear();
    c.headers_done = false;
    c.content_length = 0;
    c.rx_headers_end = std::string::npos;

    // 请求行
    std::string method, path;
    {
        size_t s1 = head.find(' ');
        size_t s2 = head.find(' ', s1 + 1);
        if (s1 == std::string::npos || s2 == std::string::npos) { close_conn(c); return; }
        method = head.substr(0, s1);
        path = head.substr(s1 + 1, s2 - s1 - 1);
        size_t q = path.find('?');
        if (q != std::string::npos) path = path.substr(0, q);
    }

    if (method == "OPTIONS") {
        http_respond(c, 204, "text/plain", "", true);
        return;
    }
    if (path == "/" || path == "/index.html") {
        std::string html = read_file_str(MCFI_PANEL);
        if (html.empty()) {
            http_respond(c, 404, "text/plain; charset=utf-8", "panel not found", true);
            return;
        }
        http_respond(c, 200, "text/html; charset=utf-8", html, true);
        return;
    }
    if (path == "/api/config") {
        if (method == "GET") {
            load_config();
            http_respond(c, 200, "text/plain; charset=utf-8", g_config_text, true);
            return;
        }
        if (method == "POST") {
            // 简单校验后保存
            if (body.size() > 16384) {
                http_respond(c, 413, "text/plain", "too large", true);
                return;
            }
            McfiConfig parsed = McfiConfig::parse(body);
            save_config(parsed.dump());
            http_respond(c, 200, "application/json", "{\"ok\":true}", true);
            return;
        }
    }
    if (path == "/api/ping") {
        http_respond(c, 200, "text/plain", "pong", true);
        return;
    }
    if (path == "/api/applog") {
        // 扫描 runtime/<pid>.log，合并所有应用的事件
        std::string s;
        DIR *d = opendir(MCFI_RUNTIME);
        if (d) {
            struct dirent *ent;
            while ((ent = readdir(d)) != nullptr) {
                const char *nm = ent->d_name;
                size_t len = 0;
                while (nm[len] && nm[len] != '.') len++;
                if (len == 0 || strcmp(nm + len, ".log") != 0) continue;
                std::string fp = std::string(MCFI_RUNTIME) + "/" + nm;
                std::string fc = read_file_str(fp.c_str());
                if (!fc.empty()) {
                    s += "── pid=";
                    s.append(nm, len);
                    s += " ──\n";
                    s += fc;
                }
            }
            closedir(d);
        }
        // 同时附上 companion 上报的旧 app.log
        std::string old = read_file_str(MCFI_APPLOG);
        if (!old.empty()) s += old;
        if (s.empty()) s = "（暂无记录：命中白名单的应用启动后会在此显示）\n";
        http_respond(c, 200, "text/plain; charset=utf-8", s, true);
        return;
    }
    if (path == "/api/daemonlog") {
        // 返回 daemon.log 尾部（最近 N 行），含入口/启动/端口等加载日志
        const int TAIL = 60;
        std::string full = read_file_str(MCFI_DAEMON_LOG);
        std::string s;
        if (!full.empty()) {
            size_t start = 0;
            int nl = 0;
            for (size_t i = full.size(); i > 0; i--) {
                if (full[i - 1] == '\n') { nl++; if (nl > TAIL) { start = i; break; } }
            }
            s = full.substr(start);
        }
        if (s.empty()) s = "（暂无日志：守护进程启动后写入 daemon.log）\n";
        http_respond(c, 200, "text/plain; charset=utf-8", s, true);
        return;
    }
    http_respond(c, 404, "text/plain; charset=utf-8", "not found", true);
}

// ---------------- module.prop 状态回写 ----------------

static void set_prop_description(const std::string &desc) {
    std::string text = read_file_str(MCFI_MODULE_PROP);
    if (text.empty()) return;
    // 替换 description= 行
    std::string out;
    size_t pos = 0;
    bool replaced = false;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        std::string line = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? text.size() : nl + 1;
        if (line.rfind("description=", 0) == 0 && !replaced) {
            out += "description=" + desc + "\n";
            replaced = true;
        } else {
            out += line;
            if (nl != std::string::npos) out += "\n";
        }
    }
    if (!replaced) out += "description=" + desc + "\n";
    FILE *f = fopen(MCFI_MODULE_PROP, "wb");
    if (f) { fputs(out.c_str(), f); fclose(f); chmod(MCFI_MODULE_PROP, 0644); }
}

static void update_status(int real_port, bool panel_ok) {
    McfiConfig cfg = McfiConfig::parse(read_file_str(MCFI_CONFIG));
    char buf[512];
    int n_targets = 0;
    if (!cfg.targets.empty()) { n_targets = 1; for (char c : cfg.targets) if (c == ',') n_targets++; }
    const char *mode = cfg.target_mode == 0 ? "全部应用" : (cfg.target_mode == 2 ? "黑名单" : "白名单");
    const char *algo = cfg.interp_mode == 0 ? "MCI运动补偿" : (cfg.interp_mode == 2 ? "运动自适应" : "混合");
    if (!panel_ok)
        snprintf(buf, sizeof(buf), "运行中(面板端口绑定失败) · 补帧%s · %s%d项 · %s",
                 cfg.enabled ? "开" : "关", mode, n_targets, algo);
    else
        snprintf(buf, sizeof(buf), "运行中 · 面板 http://127.0.0.1:%d · 补帧%s · %s%d项 · %s",
                 real_port, cfg.enabled ? "开" : "关", mode, n_targets, algo);
    set_prop_description(buf);
}

// ---------------- main ----------------

int main() {
    mkdir(MCFI_DATA_DIR, 0755);
    mkdir(MCFI_DATA_DIR "/panel", 0755);
    // 打开 daemon.log（追加），此后 LOGI/LOGE 同时写 logcat 和该文件
    g_logf = fopen(MCFI_DAEMON_LOG, "ab");
    if (g_logf) { chmod(MCFI_DAEMON_LOG, 0644); setvbuf(g_logf, nullptr, _IOLBF, 0); }
    LOGI("======== MCFI 守护进程启动 v2.6.13 ========");
    load_config();
    int port = McfiConfig::parse(g_config_text).panel_port;

    unlink(MCFI_SOCK);
    int ufd = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un un{};
    un.sun_family = AF_UNIX;
    strncpy(un.sun_path, MCFI_SOCK, sizeof(un.sun_path) - 1);
    if (bind(ufd, (sockaddr *)&un, sizeof(un)) != 0 || listen(ufd, 16) != 0) {
        LOGE("unix socket 绑定失败: %s", strerror(errno));
        return 1;
    }
    chmod(MCFI_SOCK, 0666);

    int tfd = -1;
    int real_port = 0;
    for (int p = port; p <= port + 10; p++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        int on = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        sockaddr_in in{};
        in.sin_family = AF_INET;
        in.sin_addr.s_addr = inet_addr("127.0.0.1");
        in.sin_port = htons((uint16_t)p);
        if (bind(fd, (sockaddr *)&in, sizeof(in)) == 0 && listen(fd, 8) == 0) {
            tfd = fd;
            real_port = p;
            if (p != port) LOGI("端口 %d 被占用，面板改用 %d", port, p);
            break;
        }
        close(fd);
    }
    if (tfd < 0) LOGE("面板端口 %d~%d 均绑定失败（配置服务继续）", port, port + 10);

    LOGI("守护进程启动: ctrl=%s panel=%d 配置(enabled=%d mode=%d)", MCFI_SOCK, real_port,
         McfiConfig::parse(g_config_text).enabled, McfiConfig::parse(g_config_text).target_mode);
    g_real_port = real_port;
    g_panel_ok = (tfd >= 0);
    update_status(real_port, tfd >= 0);

    for (;;) {
        std::vector<pollfd> pfds;
        pfds.push_back({ufd, POLLIN, 0});
        if (tfd >= 0) pfds.push_back({tfd, POLLIN, 0});
        for (auto &c : g_conns) pfds.push_back({c.fd, POLLIN, 0});

        int r = poll(pfds.data(), (nfds_t)pfds.size(), -1);
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        size_t idx = 0;
        if (pfds[idx].revents & POLLIN) {
            int cfd = accept(ufd, nullptr, nullptr);
            if (cfd >= 0) g_conns.push_back({cfd, false, "", std::string::npos, 0, false});
        }
        idx++;
        if (tfd >= 0) {
            if (pfds[idx].revents & POLLIN) {
                int cfd = accept(tfd, nullptr, nullptr);
                if (cfd >= 0) g_conns.push_back({cfd, true, "", std::string::npos, 0, false});
            }
            idx++;
        }
        for (size_t i = 0; i < g_conns.size() && idx < pfds.size(); i++, idx++) {
            Conn &c = g_conns[i];
            if (c.fd < 0) continue;
            if (pfds[idx].revents & POLLIN) {
                char buf[8192];
                ssize_t n = read(c.fd, buf, sizeof(buf));
                if (n <= 0) {
                    close_conn(c);
                    continue;
                }
                c.rx.append(buf, (size_t)n);
                if (c.http) handle_http(c);
                else handle_ctrl_data(c);
            } else if (pfds[idx].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                close_conn(c);
            }
        }
        // 清理已关闭连接
        for (size_t i = 0; i < g_conns.size();) {
            if (g_conns[i].fd < 0) g_conns.erase(g_conns.begin() + (long)i);
            else i++;
        }
        if (g_conns.size() > 256) { // 防泄漏
            for (size_t i = 0; i < 64; i++) close_conn(g_conns[i]);
            g_conns.erase(g_conns.begin(), g_conns.begin() + 64);
        }
    }
    return 0;
}
