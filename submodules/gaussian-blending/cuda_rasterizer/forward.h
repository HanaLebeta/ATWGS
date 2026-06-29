#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    
    void preprocess(
        const dim3 grid,
        const int P, const int D, const int M,
        const int W, const int H,
        const float focal_x, const float focal_y,
        const float tan_fovx, const float tan_fovy,
        const float* means3D,
        const glm::vec3* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* shs,
        const float* cov3D_precomp,
        const float* colors_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* cam_pos,
        int* radii,
        bool* clamped,
        float3* means2D,
        float* depths,
        float* cov3Ds,
        float* rgb,
        float4* cov_opacity,
        float2* lambdas,
        float4* nv1_nv2,
        bool prefiltered
    );

    void render(
        const dim3 grid, const dim3 block,
        const int W, const int H,
        const float* bg_color,
        const float* colors,
        const uint32_t* point_list,
        const float3* means2D,
        const float4* cov_opacity,
        const float2* lambdas,
        const float4* nv1_nv2,
        const uint2* ranges,
        uint32_t* n_contrib,
        float* final_color,
        float* out_color,
        float* out_window,
        const bool gaussian_window,
        const bool hybrid_window,
        const bool footprint_floor,
        const bool depth_coupled,
        const bool exact_window_grad,
        const bool adaptive_window,
        const float aniso_tau,
        const bool point_sample
    );
}

#endif
