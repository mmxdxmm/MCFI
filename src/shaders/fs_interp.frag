#version 450
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 fragColor;
layout(set = 0, binding = 0) uniform sampler2D uPrev;
layout(set = 0, binding = 1) uniform sampler2D uCur;
layout(push_constant) uniform PC { int mode; float strength; } pc;
void main() {
    vec4 c = texture(uCur, vUV);
    vec4 p = texture(uPrev, vUV);
    if (pc.mode == 2) { fragColor = c; return; }   // 纯拷贝模式
    if (pc.mode == 0) {
        // 普通混合：偏向当前帧（30% prev + 70% cur），减少运动重影
        fragColor = mix(p, c, 0.70);
    } else {
        // 运动自适应：根据亮度差调整混合比例
        // 运动小：接近 50/50；运动大：偏向当前帧但保留少量 prev，避免跳变
        float d = abs(dot(c.rgb - p.rgb, vec3(0.299, 0.587, 0.114)));
        float w = clamp(d * pc.strength * 4.0, 0.0, 1.0);
        // w=0(静止) → 0.5 混合；w=1(大运动) → 0.85 偏当前
        float t = mix(0.50, 0.85, w);
        fragColor = mix(p, c, t);
    }
}
