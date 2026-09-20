// MCFI 配置解析单元测试（独立编译，不依赖任何 Android/EGL 环境）
// 验证：game_interval / video_interval 拆分、≥1 钳制、旧键忽略（不兼容）、
//       me_quality 解析回归、dump 输出包含新键。
// 编译运行：g++ -std=c++17 -O2 -I src tests/test_config.cpp -o /tmp/test_config && /tmp/test_config
#include "config.h"
#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); g_fail++; } \
} while (0)

int main() {
    // ---- 1. 正常解析：双键拆分 ----
    {
        McfiConfig c = McfiConfig::parse(
            "enabled=1\n"
            "game_interval=2\n"
            "video_interval=3\n"
            "me_quality=80\n");
        printf("[1] 正常解析双键\n");
        CHECK(c.enabled == true, "enabled=1 解析");
        CHECK(c.game_interval == 2, "game_interval=2 解析");
        CHECK(c.video_interval == 3, "video_interval=3 解析");
        CHECK(c.me_quality == 80, "me_quality=80 解析（回归：面板保存后不再重置 60）");
        CHECK(c.game_interval != c.video_interval, "两键相互独立（2 != 3）");
    }

    // ---- 2. 钳制 ≥1 ----
    {
        McfiConfig c = McfiConfig::parse(
            "game_interval=0\n"
            "video_interval=-5\n");
        printf("[2] 钳制\n");
        CHECK(c.game_interval == 1, "game_interval=0 → 1");
        CHECK(c.video_interval == 1, "video_interval=-5 → 1");
    }
    {
        McfiConfig c = McfiConfig::parse("game_interval=3\nvideo_interval=2\n");
        CHECK(c.game_interval == 3 && c.video_interval == 2, "合法值不被误钳");
    }

    // ---- 3. 旧键忽略（不兼容旧值，直接清理）----
    {
        McfiConfig c = McfiConfig::parse(
            "gen_interval=5\n"
            "video_gen_interval=7\n"
            "game_interval=2\n");
        printf("[3] 旧键 gen_interval/video_gen_interval 应被忽略\n");
        CHECK(c.game_interval == 2, "game_interval 仍按新键解析");
        CHECK(c.video_interval == 1, "旧 video_gen_interval 不影响 video_interval（默认 1）");
    }

    // ---- 4. dump 输出包含新键 ----
    {
        McfiConfig c = McfiConfig::parse("game_interval=2\nvideo_interval=3\n");
        std::string d = c.dump();
        printf("[4] dump\n");
        CHECK(d.find("game_interval=2\n") != std::string::npos, "dump 含 game_interval=2");
        CHECK(d.find("video_interval=3\n") != std::string::npos, "dump 含 video_interval=3");
        CHECK(d.find("gen_interval=") == std::string::npos, "dump 无旧键 gen_interval");
        CHECK(d.find("me_quality=60\n") != std::string::npos, "dump 含 me_quality");
    }

    // ---- 5. 空配置默认值 ----
    {
        McfiConfig c = McfiConfig::parse("");
        printf("[5] 空配置\n");
        CHECK(c.game_interval == 1 && c.video_interval == 1, "双键默认 1");
        CHECK(c.me_quality == 60, "me_quality 默认 60");
    }

    if (g_fail == 0) {
        printf("\n全部用例通过（15 项）\n");
        return 0;
    }
    printf("\n%d 项失败\n", g_fail);
    return 1;
}
