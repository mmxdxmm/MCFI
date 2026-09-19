# Vulkan L2 多尺度 + subgroup 加速 —— payload_vk.cpp 完整集成指南

> 目标：在现有单尺度（1/4，me0/me1）基础上，新增 L2 粗层（1/8）+ subgroup 细化。
> 原则：**所有新资源/管线创建失败时静默回退到现有单尺度**，绝不破坏已工作的路径。
> 风格：无分支（step/mix/clamp/select），与现有代码一致。
>
> 新 shader 已写好：
> - `src/shaders/me0_l2.comp` （L2 1/8 粗搜索，push: uP0=(R2,block,dsW2,dsH2)）
> - `src/shaders/me1_l2.comp` （L1 1/4 细化，读 L2 粗场放大×2 作中心，push: uP0=(R,block,dsW,dsH), uP1=(R2,0,gw2,gh2), sr）
> - `src/shaders/me1_sub.comp`（subgroup 加速版 me1_l2，local_size 16×1，dispatch 维度不同）
>
> 编译：`cd src/shaders && ./gen_spv.sh`（需 glslangValidator），产出 `me0_l2_spv.h / me1_l2_spv.h / me1_sub_spv.h`。

---

## 步骤 0：顶部 include（约第 61 行附近）

```cpp
#include "shaders/me0_l2_spv.h"
#include "shaders/me1_l2_spv.h"
#include "shaders/me1_sub_spv.h"
```

## 步骤 1：DeviceCtx 新增字段（约第 230~262 行，现有 ds_img/mv_img 之后）

```cpp
    // ---- L2 多尺度（1/8）新增 ----
    VkImage    ds_img_l2[2]  = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory ds_mem_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView  ds_view_l2[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImage    mv_img_l2 = VK_NULL_HANDLE;   // L2 粗场
    VkDeviceMemory mv_mem_l2 = VK_NULL_HANDLE;
    VkImageView  mv_view_l2 = VK_NULL_HANDLE;
    int dsW2 = 0, dsH2 = 0, mvW2 = 0, mvH2 = 0;   // 1/8 尺寸
    bool l2_ok = false;                       // L2 资源是否就绪
    bool subgroup_ok = false;                 // 设备是否支持 subgroup arithmetic

    VkShaderModule cs0_l2 = VK_NULL_HANDLE, cs1_l2 = VK_NULL_HANDLE, cs1_sub = VK_NULL_HANDLE;
    VkDescriptorSetLayout cs0_l2_ds_layout = VK_NULL_HANDLE, cs1_l2_ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool cs0_l2_ds_pool = VK_NULL_HANDLE, cs1_l2_ds_pool = VK_NULL_HANDLE;
    VkDescriptorSet cs0_l2_ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet cs1_l2_ds[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout cs0_l2_pl = VK_NULL_HANDLE, cs1_l2_pl = VK_NULL_HANDLE;
    VkPipeline cs0_l2_pipeline = VK_NULL_HANDLE, cs1_l2_pipeline = VK_NULL_HANDLE;
```

## 步骤 2：cleanup 函数（约第 414~463 行）新增销毁

```cpp
    if (cs0_l2_pipeline) f.DestroyPipeline(dc->dev, cs0_l2_pipeline, nullptr);
    if (cs1_l2_pipeline) f.DestroyPipeline(dc->dev, cs1_l2_pipeline, nullptr);
    if (cs0_l2_pl) f.DestroyPipelineLayout(dc->dev, cs0_l2_pl, nullptr);
    if (cs1_l2_pl) f.DestroyPipelineLayout(dc->dev, cs1_l2_pl, nullptr);
    if (cs0_l2) f.DestroyShaderModule(dc->dev, cs0_l2, nullptr);
    if (cs1_l2) f.DestroyShaderModule(dc->dev, cs1_l2, nullptr);
    if (cs1_sub) f.DestroyShaderModule(dc->dev, cs1_sub, nullptr);
    if (cs0_l2_ds_pool) f.DestroyDescriptorPool(dc->dev, cs0_l2_ds_pool, nullptr);
    if (cs1_l2_ds_pool) f.DestroyDescriptorPool(dc->dev, cs1_l2_ds_pool, nullptr);
    if (cs0_l2_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cs0_l2_ds_layout, nullptr);
    if (cs1_l2_ds_layout) f.DestroyDescriptorSetLayout(dc->dev, cs1_l2_ds_layout, nullptr);
    for (int i = 0; i < 2; i++) if (ds_img_l2[i]) f.DestroyImage(dc->dev, ds_img_l2[i], nullptr);
    if (mv_img_l2) f.DestroyImage(dc->dev, mv_img_l2, nullptr);
    // 内存随 VkDevice 销毁，无需单独 free（与现有 ds_mem 一致）
```

## 步骤 3：subgroup 能力探测（init 早期，设备创建后）

```cpp
    // 探测 subgroup arithmetic（me1_sub 用）
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    g_GetPhysDevProperties2(dc->pd, &p2);   // 需 LOAD_FN(vkGetPhysicalDeviceProperties2)
    cx->subgroup_ok = (sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    VKLOGI("subgroup arithmetic 支持: %d (size=%u)", (int)cx->subgroup_ok, sg.subgroupSize);
```
> 注意：若 `vkGetPhysicalDeviceProperties2` 未加载，可用 `vkGetPhysicalDeviceProperties` +
> `VkPhysicalDeviceFeatures.shaderSubgroupArithmetic`（VK_KHR_shader_subgroup_arithmetic）等价探测。
> subgroup_ok=false 时用 me1_l2（普通版），不创建 me1_sub 管线。

## 步骤 4：init 8a —— L2 图像（紧接现有 ds_img 创建之后，约第 717 行）

```cpp
            // ---- L2 多尺度：1/8 降采样层 + L2 粗场 ----
            cx->dsW2 = (cx->dsW + 1) / 2;     // 1/4 的一半 = 原图 1/8
            cx->dsH2 = (cx->dsH + 1) / 2;
            cx->mvW2 = (cx->dsW2 + 7) / 8;
            cx->mvH2 = (cx->dsH2 + 7) / 8;
            bool l2_img_ok = true;
            for (int i = 0; i < 2; i++) {
                if (!make_image(dc, &cx->ds_img_l2[i], &cx->ds_mem_l2[i], &cx->ds_view_l2[i],
                                cx->format, cx->dsW2, cx->dsH2,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                    l2_img_ok = false; break;
                }
            }
            if (l2_img_ok && !make_image(dc, &cx->mv_img_l2, &cx->mv_mem_l2, &cx->mv_view_l2,
                                VK_FORMAT_R8G8B8A8_UNORM, cx->mvW2, cx->mvH2,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) {
                l2_img_ok = false;
            }
            cx->l2_ok = l2_img_ok;
            VKLOGI("L2 多尺度资源: %d (1/8 = %dx%d, R2=R/2)", (int)cx->l2_ok, cx->dsW2, cx->dsH2);
```

## 步骤 5：init 8c —— L2 shader module / 布局 / 管线布局（约第 784 行 gmv 之前）

```cpp
            if (cx->l2_ok) {
                sm.codeSize = me0_l2_spv_len; sm.pCode = (const uint32_t *)me0_l2_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs0_l2) != VK_SUCCESS) {
                    VKLOGI("L2 me0_l2 module 失败，回退单尺度"); cx->l2_ok = false;
                }
            }
            if (cx->l2_ok) {
                sm.codeSize = me1_l2_spv_len; sm.pCode = (const uint32_t *)me1_l2_spv;
                if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1_l2) != VK_SUCCESS) {
                    VKLOGI("L2 me1_l2 module 失败，回退单尺度"); cx->l2_ok = false;
                }
                if (cx->subgroup_ok) {
                    sm.codeSize = me1_sub_spv_len; sm.pCode = (const uint32_t *)me1_sub_spv;
                    if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1_sub) != VK_SUCCESS) {
                        VKLOGI("me1_sub module 失败，用普通 me1_l2"); cx->cs1_sub = VK_NULL_HANDLE;
                    }
                }
            }

            if (cx->l2_ok) {
                // cs0_l2：uPrev/uCur/uMvPrev(sampler)/mvA_L2(storage) —— 4 binding
                VkDescriptorSetLayoutBinding cb0l2[4]{};
                cb0l2[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb0l2[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb0l2[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb0l2[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                VkDescriptorSetLayoutCreateInfo cdsl2{};
                cdsl2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                cdsl2.bindingCount=4; cdsl2.pBindings=cb0l2;
                if (f.CreateDescriptorSetLayout(dc->dev, &cdsl2, nullptr, &cx->cs0_l2_ds_layout)!=VK_SUCCESS) {
                    VKLOGI("cs0_l2 layout 失败，回退单尺度"); cx->l2_ok=false;
                }
            }
            if (cx->l2_ok) {
                // cs1_l2：uPrev/uCur/uMVPrev/mvB(storage)/uGMV/uMvL2 —— 6 binding
                VkDescriptorSetLayoutBinding cb1l2[6]{};
                cb1l2[0]={0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[1]={1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[2]={2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[3]={3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[4]={4,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                cb1l2[5]={5,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
                VkDescriptorSetLayoutCreateInfo cdsl1{};
                cdsl1.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                cdsl1.bindingCount=6; cdsl1.pBindings=cb1l2;
                if (f.CreateDescriptorSetLayout(dc->dev, &cdsl1, nullptr, &cx->cs1_l2_ds_layout)!=VK_SUCCESS) {
                    VKLOGI("cs1_l2 layout 失败，回退单尺度"); cx->l2_ok=false;
                }
            }
            if (cx->l2_ok) {
                VkPushConstantRange cpc0l2{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};        // ivec4
                VkPipelineLayoutCreateInfo cpl0l2{};
                cpl0l2.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                cpl0l2.setLayoutCount=1; cpl0l2.pSetLayouts=&cx->cs0_l2_ds_layout;
                cpl0l2.pushConstantRangeCount=1; cpl0l2.pPushConstantRanges=&cpc0l2;
                f.CreatePipelineLayout(dc->dev, &cpl0l2, nullptr, &cx->cs0_l2_pl);

                VkPushConstantRange cpc1l2{VK_SHADER_STAGE_COMPUTE_BIT, 0, 36};        // ivec4+ivec4+int
                VkPipelineLayoutCreateInfo cpl1l2{};
                cpl1l2.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                cpl1l2.setLayoutCount=1; cpl1l2.pSetLayouts=&cx->cs1_l2_ds_layout;
                cpl1l2.pushConstantRangeCount=1; cpl1l2.pPushConstantRanges=&cpc1l2;
                f.CreatePipelineLayout(dc->dev, &cpl1l2, nullptr, &cx->cs1_l2_pl);
            }
```

## 步骤 6：init 8d —— L2 管线 + 描述符池/集（约第 890 行）

```cpp
            if (cx->l2_ok) {
                VkComputePipelineCreateInfo cpi0l2{};
                cpi0l2.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                cpi0l2.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                cpi0l2.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
                cpi0l2.stage.pName="main"; cpi0l2.layout=cx->cs0_l2_pl; cpi0l2.stage.module=cx->cs0_l2;
                if (f.CreateComputePipelines(dc->dev, VK_NULL_HANDLE, 1, &cpi0l2, nullptr, &cx->cs0_l2_pipeline)!=VK_SUCCESS) {
                    VKLOGI("cs0_l2 pipeline 失败，回退单尺度"); cx->l2_ok=false;
                }
            }
            if (cx->l2_ok) {
                // me1 细化管线：优先 subgroup 版（cs1_sub），否则普通 me1_l2（cs1_l2）
                VkComputePipelineCreateInfo cpi1l2{};
                cpi1l2.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
                cpi1l2.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                cpi1l2.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
                cpi1l2.stage.pName="main"; cpi1l2.layout=cx->cs1_l2_pl;
                cpi1l2.stage.module = cx->cs1_sub ? cx->cs1_sub : cx->cs1_l2;
                if (f.CreateComputePipelines(dc->dev, VK_NULL_HANDLE, 1, &cpi1l2, nullptr, &cx->cs1_l2_pipeline)!=VK_SUCCESS) {
                    VKLOGI("cs1_l2 pipeline 失败，回退单尺度"); cx->l2_ok=false;
                }
            }
            // 描述符池 + allocate
            if (cx->l2_ok) {
                VkDescriptorPoolSize ps0l2[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,6},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,2}};
                VkDescriptorPoolCreateInfo cdp0l2{};
                cdp0l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                cdp0l2.maxSets=2; cdp0l2.poolSizeCount=2; cdp0l2.pPoolSizes=ps0l2;
                f.CreateDescriptorPool(dc->dev, &cdp0l2, nullptr, &cx->cs0_l2_ds_pool);
                VkDescriptorSetLayout l0[2]={cx->cs0_l2_ds_layout,cx->cs0_l2_ds_layout};
                VkDescriptorSetAllocateInfo cda0l2{};
                cda0l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                cda0l2.descriptorPool=cx->cs0_l2_ds_pool; cda0l2.descriptorSetCount=2; cda0l2.pSetLayouts=l0;
                f.AllocateDescriptorSets(dc->dev, &cda0l2, cx->cs0_l2_ds);

                VkDescriptorPoolSize ps1l2[2]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,12},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,2}};
                VkDescriptorPoolCreateInfo cdp1l2{};
                cdp1l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                cdp1l2.maxSets=2; cdp1l2.poolSizeCount=2; cdp1l2.pPoolSizes=ps1l2;
                f.CreateDescriptorPool(dc->dev, &cdp1l2, nullptr, &cx->cs1_l2_ds_pool);
                VkDescriptorSetLayout l1[2]={cx->cs1_l2_ds_layout,cx->cs1_l2_ds_layout};
                VkDescriptorSetAllocateInfo cda1l2{};
                cda1l2.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                cda1l2.descriptorPool=cx->cs1_l2_ds_pool; cda1l2.descriptorSetCount=2; cda1l2.pSetLayouts=l1;
                f.AllocateDescriptorSets(dc->dev, &cda1l2, cx->cs1_l2_ds);
            }
```

## 步骤 7：descriptor 更新（约第 981~1015 行，现有 cs0/cs1 更新之后）

```cpp
        if (cx->mci_ok && cx->l2_ok) {
            for (int k = 0; k < 2; k++) {
                // cs0_l2：uPrev=ds_l2[1-k], uCur=ds_l2[k], uMvPrev=mv_view_l2(上一帧L2场), mvA_L2(storage)
                VkDescriptorImageInfo a0[4];
                a0[0]={cx->sampler, cx->ds_view_l2[1-k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a0[1]={cx->sampler, cx->ds_view_l2[k],   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a0[2]={cx->sampler, cx->mv_view_l2,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a0[3]={VK_NULL_HANDLE, cx->mv_view_l2,   VK_IMAGE_LAYOUT_GENERAL};
                VkWriteDescriptorSet w0[4]{};
                for (int b=0;b<4;b++){
                    w0[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w0[b].dstSet=cx->cs0_l2_ds[k]; w0[b].dstBinding=b; w0[b].descriptorCount=1;
                    w0[b].descriptorType=(b==3)?VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w0[b].pImageInfo=&a0[b];
                }
                f.UpdateDescriptorSets(dc->dev, 4, w0, 0, nullptr);

                // cs1_l2：uPrev=ds[1-k], uCur=ds[k], uMVPrev=mv_viewB(上一帧L1场), mvB(storage), uGMV, uMvL2=mv_view_l2
                VkDescriptorImageInfo a1[6];
                a1[0]={cx->sampler, cx->ds_view[1-k], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[1]={cx->sampler, cx->ds_view[k],    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[2]={cx->sampler, cx->mv_viewB,     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[3]={VK_NULL_HANDLE, cx->mv_viewB,   VK_IMAGE_LAYOUT_GENERAL};
                a1[4]={cx->sampler, (cx->gmv_view?cx->gmv_view:cx->mv_view), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                a1[5]={cx->sampler, cx->mv_view_l2,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet w1[6]{};
                for (int b=0;b<6;b++){
                    w1[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w1[b].dstSet=cx->cs1_l2_ds[k]; w1[b].dstBinding=b; w1[b].descriptorCount=1;
                    w1[b].descriptorType=(b==3)?VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w1[b].pImageInfo=&a1[b];
                }
                f.UpdateDescriptorSets(dc->dev, 6, w1, 0, nullptr);
            }
        }
```
> 注意：cs1_l2 的 mvB(storage) 仍写 `mv_viewB`（与现有插帧 frag 读取一致）；
> L2 场 `mv_view_l2` 同时被 cs0_l2 写（storage）和 cs1_l2 读（sampler），靠 barrier 同步。

## 步骤 8：worker dispatch 顺序（约第 1251~1374 行）

在现有「blit copy→ds（1/4）」之后、cs0（L1 me0）之前，插入 L2 流程：

```cpp
        // ===== 新增：L2 1/8 降采样 blit（从 ds[cur] 1/4 再 blit 一半）=====
        if (cx->l2_ok) {
            image_barrier(cmd, cx->ds_img_l2[cur],
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkImageBlit bl2{};
            bl2.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bl2.srcSubresource.layerCount = 1;
            bl2.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bl2.dstSubresource.layerCount = 1;
            bl2.srcOffsets[1] = {cx->dsW, cx->dsH, 1};
            bl2.dstOffsets[1] = {cx->dsW2, cx->dsH2, 1};
            f.CmdBlitImage(cmd, cx->ds_img[cur], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           cx->ds_img_l2[cur], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &bl2, VK_FILTER_LINEAR);
            image_barrier(cmd, cx->ds_img_l2[cur],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        // ===== 新增：L2 粗搜索 cs0_l2 =====
        if (cx->l2_ok) {
            image_barrier(cmd, cx->mv_img_l2,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs0_l2_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs0_l2_pl, 0, 1, &cx->cs0_l2_ds[cur], 0, nullptr);
            int pc0l2[4] = { R2, 8, cx->dsW2, cx->dsH2 };   // R2 = R/2
            f.CmdPushConstants(cmd, cx->cs0_l2_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc0l2);
            f.CmdDispatch(cmd, (cx->mvW2+7)/8, (cx->mvH2+7)/8, 1);
            image_barrier(cmd, cx->mv_img_l2,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
```

然后**把现有 cs1（me1）替换为 L2 版**（cs1_l2_pipeline + cs1_l2_ds[cur] + 36 字节 push constant）：

```cpp
        // cs1 细化：L2 模式用 cs1_l2（含 subgroup 版），否则用原 cs1
        if (cx->l2_ok) {
            image_barrier(cmd, cx->mv_imgB,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            f.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs1_l2_pipeline);
            f.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cx->cs1_l2_pl, 0, 1, &cx->cs1_l2_ds[cur], 0, nullptr);
            // push: uP0=(R,8,dsW,dsH)  uP1=(R2,0,gw2,gh2)  sr
            int pc1l2[9] = { R, 8, cx->dsW, cx->dsH,  R2, 0, cx->mvW2, cx->mvH2,  sr };
            f.CmdPushConstants(cmd, cx->cs1_l2_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 36, pc1l2);
            if (cx->cs1_sub) {
                // subgroup 版：local_size 16×1，每 workgroup 一个 block → (gw, gh, 1)
                f.CmdDispatch(cmd, cx->mvW, cx->mvH, 1);
            } else {
                f.CmdDispatch(cmd, (cx->dsW+7)/8, (cx->dsH+7)/8, 1);
            }
            image_barrier(cmd, cx->mv_imgB,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        } else {
            // ===== 原有单尺度 cs1 路径（保留，作为回退）=====
            ...（原 1370~1377 行代码不变）...
        }
```

> **关键注意**：
> - `R2 = R/2`（R 由现有逻辑算出，L1 层搜索半径；L2 层取一半，因为 L2 分辨率是 L1 的一半，同样的原图位移在 L2 层只要一半像素）。
> - subgroup 版 dispatch 维度是 `(gw, gh, 1)`（每 workgroup 一个 block），普通版是 `(dsW/8, dsH/8, 1)`（每线程一个 block）——**两者不同，切勿混用**。
> - L1 的 cs0（me0，写 mvA）仍保留：它给 cs1_l2 的 `uMVPrev`（binding 2）提供上一帧 L1 场；但 cs1_l2 实际用的是 L2 粗场 `uMvL2`（binding 5）+ 上一帧 L1 场融合。
> - 现有 gmv0/gmv1 统计的是 `mv_view`（mvA，L1 粗场），L2 模式下仍可用（GMV 是全局量，与尺度无关）。

## 步骤 9：开关（可选，panel/config）
- 在 config.h 加 `bool vk_l2 = true;`，init 里 `l2_ok = l2_img_ok && cfg.vk_l2;`。
- 面板加「Vulkan 多尺度 L2」开关，写回 config.conf。

---

## 验证清单（真机 logcat）
1. `L2 多尺度资源: 1 (1/8 = WxH, R2=R/2)`
2. `subgroup arithmetic 支持: 1 (size=16/32/64)`
3. 快速横移/舞蹈视频，L2 模式下大位移块不拖影、不闪。
4. 若 `l2_ok=0`，确认自动回退单尺度（cs1 走 else 分支）。

---

## 补充：L2 fp16 半精度版接线（与旧 me0_f16/me1_f16 同策略）

新 shader 已写两个 fp16 版：`me0_l2_f16.comp` / `me1_l2_f16.comp`
（`me1_sub.comp` 内的采样差也已用 `float16_t` 算、归约时转 float；`subgroupAdd` 本身不支持 half）。

接线方式照抄现有旧版：

1. 顶部 include：
```cpp
#include "shaders/me0_l2_f16_spv.h"
#include "shaders/me1_l2_f16_spv.h"
```
2. DeviceCtx 加字段：
```cpp
VkShaderModule cs0_l2f = VK_NULL_HANDLE, cs1_l2f = VK_NULL_HANDLE;
```
3. 步骤5 创建 module 时（g_vk_f16 已在设备级启用）：
```cpp
if (g_vk_f16) {
    sm.codeSize = me0_l2_f16_spv_len; sm.pCode = (const uint32_t *)me0_l2_f16_spv;
    if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs0_l2f) != VK_SUCCESS) cx->cs0_l2f = VK_NULL_HANDLE;
    sm.codeSize = me1_l2_f16_spv_len; sm.pCode = (const uint32_t *)me1_l2_f16_spv;
    if (f.CreateShaderModule(dc->dev, &sm, nullptr, &cx->cs1_l2f) != VK_SUCCESS) cx->cs1_l2f = VK_NULL_HANDLE;
}
```
4. 步骤6 建管线时选 module（与旧版一致，优先 fp16）：
```cpp
cpi0l2.stage.module = cx->cs0_l2f ? cx->cs0_l2f : cx->cs0_l2;   // cs0_l2
// cs1_l2 细化同理：
cpi1l2.stage.module = cx->cs1_l2f ? cx->cs1_l2f
                    : (cx->cs1_sub ? cx->cs1_sub : cx->cs1_l2);
```
> 注意：`me1_sub`（subgroup 版）本身已用 half 算差，不再单独做 _f16；
> 选优先级：`cs1_sub`（subgroup，已含 half 算差） > `cs1_l2f`（fp16 普通） > `cs1_l2`（fp32）。
> cleanup 里补 `DestroyShaderModule(cs0_l2f/cs1_l2f)`。

