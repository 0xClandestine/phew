import mlx.core as mx


@mx.compile
def ds4_hc_cycle_opt(x0, x1, x2, x3, x4, x5, x6):
    t1 = mx.reshape(x0, [x0.shape[0], x0.shape[1], -1])
    t2 = t1.astype(mx.float32)
    t3 = t2 * t2
    t4 = mx.mean(t3, axis=[2], keepdims=True)
    c5 = mx.array(1e-06)
    t6 = t4 + c5
    t7 = mx.rsqrt(t6)
    t8 = t2 @ x1
    t9 = t8 * t7
    _kernel_17 = mx.fast.metal_kernel(
        name="kernel_17",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source="""
        uint idx = thread_position_in_grid.x;
        constexpr int MIX  = (2 + HC) * HC;
        constexpr int BASE = 2 * HC;

        const device float* mix = (const device float*)mixes + idx * MIX;
        device float* pre_out   = (device float*)pre  + idx * HC;
        device float* post_out  = (device float*)post + idx * HC;
        device float* comb_out  = (device float*)comb + idx * HC * HC;

        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float epsv       = eps[0];

        // Pre-sigmoid
        {
            float4 z = *(const device float4*)mix * pre_scale
                     + *(const device float4*)base;
            *(device float4*)pre_out = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }

        // Post-sigmoid
        {
            float4 z = *(const device float4*)(mix + HC) * post_scale
                     + *(const device float4*)(base + HC);
            *(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));
        }

        // Comb: four float4 loads — all independent, GPU issues in parallel
        float4 v0 = *(const device float4*)(mix  + BASE     ) * comb_scale + *(const device float4*)(base + BASE     );
        float4 v1 = *(const device float4*)(mix  + BASE +  4) * comb_scale + *(const device float4*)(base + BASE +  4);
        float4 v2 = *(const device float4*)(mix  + BASE +  8) * comb_scale + *(const device float4*)(base + BASE +  8);
        float4 v3 = *(const device float4*)(mix  + BASE + 12) * comb_scale + *(const device float4*)(base + BASE + 12);

        // Per-row stable softmax
        float m0 = metal::max(metal::max(v0.x, v0.y), metal::max(v0.z, v0.w));
        float m1 = metal::max(metal::max(v1.x, v1.y), metal::max(v1.z, v1.w));
        float m2 = metal::max(metal::max(v2.x, v2.y), metal::max(v2.z, v2.w));
        float m3 = metal::max(metal::max(v3.x, v3.y), metal::max(v3.z, v3.w));

        float4 e0 = metal::fast::exp(v0 - m0);
        float4 e1 = metal::fast::exp(v1 - m1);
        float4 e2 = metal::fast::exp(v2 - m2);
        float4 e3 = metal::fast::exp(v3 - m3);

        float4 r0 = e0 * 1.0f / (e0.x + e0.y + e0.z + e0.w) + epsv;
        float4 r1 = e1 * 1.0f / (e1.x + e1.y + e1.z + e1.w) + epsv;
        float4 r2 = e2 * 1.0f / (e2.x + e2.y + e2.z + e2.w) + epsv;
        float4 r3 = e3 * 1.0f / (e3.x + e3.y + e3.z + e3.w) + epsv;

        // Initial column normalization
        float4 col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col; r1 *= col; r2 *= col; r3 *= col;

        // Sinkhorn iterations
        for (int iter = 1; iter < ITERS; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col; r1 *= col; r2 *= col; r3 *= col;
        }

        *(device float4*)(comb_out)      = r0;
        *(device float4*)(comb_out +  4) = r1;
        *(device float4*)(comb_out +  8) = r2;
        *(device float4*)(comb_out + 12) = r3;
    """,
    )
    _D0_17 = t9.shape[0]
    _D1_17 = t9.shape[1]
    kernel10_out = _kernel_17(
        inputs=[t9, x2, x3, x4],
        output_shapes=[(_D0_17, _D1_17, 4), (_D0_17, _D1_17, 4), (_D0_17, _D1_17, 4, 4)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(_D0_17 * _D1_17, 1, 1),
        threadgroup=(256, 1, 1),
        template=[("HC", 4), ("ITERS", 20)],
    )
    t11 = kernel10_out[0]
    t12 = kernel10_out[1]
    t13 = kernel10_out[2]
    t14 = mx.expand_dims(t11, 3)
    t15 = x0.astype(mx.float32)
    t16 = t14 * t15
    t17 = mx.sum(t16, axis=[2], keepdims=False)
    t18 = t17.astype(mx.float32)
    t19 = mx.expand_dims(t12, 3)
    _kernel_44 = mx.fast.metal_kernel(
        name="kernel_44",
        input_names=["inp0", "inp1"],
        output_names=["out0"],
        source="""    uint elem = thread_position_in_grid.x;
    
    float v1 = inp0[elem];
    float v2 = inp1[elem];
    
    constexpr float v3 = 10.0f;
    float v4 = metal::min(v1, v3);
    constexpr float v5 = 10.0f;
    constexpr float v6 = -10.0f;
    float v7 = metal::min(v2, v5);
    float v8 = metal::max(v7, v6);
    float v9 = 1.0f / (1.0f + metal::exp(-v4));
    float v10 = v4 * v9;
    float v11 = v10 * v8;
    
    out0[elem] = (float)v11;""",
    )
    kernel20_out = _kernel_44(
        inputs=[x5, x6],
        output_shapes=[x5.shape],
        output_dtypes=[mx.float32],
        grid=(x5.size, 1, 1),
        threadgroup=(256, 1, 1),
        template=[],
    )
    kernel20 = kernel20_out[0]
    t21 = mx.expand_dims(kernel20, 2)
    t22 = t21.astype(mx.float32)
    t23 = t19 * t22
    t24 = mx.transpose(t13, [0, 1, 3, 2])
    t25 = x0.astype(mx.float32)
    t26 = t24 @ t25
    t27 = t23 + t26
    t28 = t27.astype(mx.float32)
    return t28
