#version 450
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 fragColor;
layout(set = 0, binding = 0) uniform sampler2D uPrev;
layout(set = 0, binding = 1) uniform sampler2D uCur;
layout(set = 0, binding = 2) uniform sampler2D uMV;
layout(set = 0, binding = 3) uniform sampler2D uGMV;   // 1×1 GMV（x,y=归一化 z=峰值占比）
layout(push_constant) uniform PC {
    int uMode;         // 0=MCI 1=普通混合 2=运动自适应
    float uStrength;
    float uRange;
    float uPad0;
    vec2 uDsTexel;
    float uSmooth;    // 平滑 0~100：静止/近静止保护阈值
    float uPad2;
} pc;
void main() {
    vec4 c = texture(uCur, vUV);
    vec4 p = texture(uPrev, vUV);
    vec4 mid = mix(p, c, 0.5);
    if (pc.uMode == 1) { fragColor = mid; return; }
    if (pc.uMode == 2) {
        float d = abs(dot(c.rgb - p.rgb, vec3(0.299, 0.587, 0.114)));
        float w = clamp(d * pc.uStrength * 4.0, 0.0, 1.0);
        fragColor = mix(mid, c, w);
        return;
    }
    // MCI：按运动矢量双向 warp
    vec4 mv = texture(uMV, vUV);
    vec2 v = (mv.xy * 2.0 - 1.0) * pc.uRange * pc.uDsTexel;
    float conf = mv.z;
    // 静止/近静止保护：两帧亮度差低于阈值 → 直接取当前帧（零 warp 零混合）
    float ld = abs(dot(c.rgb - p.rgb, vec3(0.299, 0.587, 0.114)));
    if (ld < pc.uSmooth * 0.0004) { fragColor = c; return; }
    vec4 cp = texture(uPrev, vUV - v * 0.5);
    vec4 cc = texture(uCur,  vUV + v * 0.5);
    vec4 mci = mix(cp, cc, 0.5);
    // 统一中间帧合成（SVP 式）：高置信 = 运动补偿双向 warp（t=0.5 位置采样）；
    // 低置信 = 原始两帧 50% 混合（同样是 t=0.5 时间点）——置信度连续渐变过渡，
    // 无跳帧、无区域硬边界 → 无边缘撕裂。（与 GLES FS_INTERP 对称；ME 端 alpha 恒 1.0）
    float fall = clamp((1.0 - conf) * 2.0, 0.0, 1.0);
    vec4 blend = mix(p, c, 0.5);
    vec4 outC = mix(blend, mci, 1.0 - fall);
    // 遮挡/拖影抑制：warp 后两侧采样亮度差异大（一侧被遮挡）→ 向当前帧轻微渐变（软回退）
    float lp = dot(cp.rgb, vec3(0.299, 0.587, 0.114));
    float lc = dot(cc.rgb, vec3(0.299, 0.587, 0.114));
    float g = clamp(abs(lp - lc) * pc.uStrength * 8.0, 0.0, 1.0);
    outC = mix(outC, c, g * 0.35);
    fragColor = outC;
}
