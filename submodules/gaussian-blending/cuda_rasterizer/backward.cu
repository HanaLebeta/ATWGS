#include "backward.h"
#include "vec_math.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float3 sat_psd_clamp(float3 M, float fl) {
    const float a = M.x, b = M.y, c = M.z;
    const float tr = a + c;
    const float disc = sqrtf(fmaxf(0.0f, 0.25f * tr * tr - (a * c - b * b)));
    const float l1 = 0.5f * tr + disc;
    const float l2 = 0.5f * tr - disc;
    const float CAP = 1e7f;
    const float c1 = fminf(fmaxf(l1, fl), CAP), c2 = fminf(fmaxf(l2, fl), CAP);
    if (fabsf(b) < 1e-12f)
        return make_float3(fminf(fmaxf(a, fl), CAP), 0.0f, fminf(fmaxf(c, fl), CAP));
    float vx = b, vy = l1 - a;
    const float inv_n = rsqrtf(vx * vx + vy * vy);
    vx *= inv_n; vy *= inv_n;
    const float vxx = vx * vx, vxy = vx * vy, vyy = vy * vy;
    return make_float3(c1 * vxx + c2 * (1.0f - vxx),
                       c1 * vxy - c2 * vxy,
                       c1 * vyy + c2 * (1.0f - vyy));
}

__device__ void computeColorFromSH(
    const int idx,
    const int deg,
    const int max_coeffs,
    const glm::vec3* means,
    const glm::vec3 campos,
    const float* shs,
    const bool* clamped,
    const glm::vec3* dL_dcolor,
    glm::vec3* dL_dmeans,
    glm::vec3* dL_dshs
) {
    
    glm::vec3 pos = means[idx];
    glm::vec3 dir_orig = pos - campos;
    glm::vec3 dir = dir_orig / glm::length(dir_orig);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    glm::vec3 dL_dRGB = dL_dcolor[idx];
    dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
    dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
    dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

    glm::vec3 dRGBdx(0, 0, 0);
    glm::vec3 dRGBdy(0, 0, 0);
    glm::vec3 dRGBdz(0, 0, 0);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

    float dRGBdsh0 = SH_C0;
    dL_dsh[0] = dRGBdsh0 * dL_dRGB;
    if (deg > 0) {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh[1] = dRGBdsh1 * dL_dRGB;
        dL_dsh[2] = dRGBdsh2 * dL_dRGB;
        dL_dsh[3] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh[3];
        dRGBdy = -SH_C1 * sh[1];
        dRGBdz = SH_C1 * sh[2];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh[4] = dRGBdsh4 * dL_dRGB;
            dL_dsh[5] = dRGBdsh5 * dL_dRGB;
            dL_dsh[6] = dRGBdsh6 * dL_dRGB;
            dL_dsh[7] = dRGBdsh7 * dL_dRGB;
            dL_dsh[8] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
            dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
            dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

            if (deg > 2) {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh[9] = dRGBdsh9 * dL_dRGB;
                dL_dsh[10] = dRGBdsh10 * dL_dRGB;
                dL_dsh[11] = dRGBdsh11 * dL_dRGB;
                dL_dsh[12] = dRGBdsh12 * dL_dRGB;
                dL_dsh[13] = dRGBdsh13 * dL_dRGB;
                dL_dsh[14] = dRGBdsh14 * dL_dRGB;
                dL_dsh[15] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (
                    SH_C3[0] * sh[9] * 3.f * 2.f * xy +
                    SH_C3[1] * sh[10] * yz +
                    SH_C3[2] * sh[11] * -2.f * xy +
                    SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                    SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                    SH_C3[5] * sh[14] * 2.f * xz +
                    SH_C3[6] * sh[15] * 3.f * (xx - yy));

                dRGBdy += (
                    SH_C3[0] * sh[9] * 3.f * (xx - yy) +
                    SH_C3[1] * sh[10] * xz +
                    SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
                    SH_C3[3] * sh[12] * -3.f * 2.f * yz +
                    SH_C3[4] * sh[13] * -2.f * xy +
                    SH_C3[5] * sh[14] * -2.f * yz +
                    SH_C3[6] * sh[15] * -3.f * 2.f * xy);

                dRGBdz += (
                    SH_C3[1] * sh[10] * xy +
                    SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                    SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                    SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                    SH_C3[5] * sh[14] * (xx - yy));
            }
        }
    }

    glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

    float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

    dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__global__ void computeCov2DCUDA(
    const int P,
    const float3* means,
    const int* radii,
    const float* cov3Ds,
    const float h_x, float h_y,
    const float tan_fovx, float tan_fovy,
    const float* view_matrix,
    const float* dL_dcovs,
    float3* dL_dmeans,
    float* dL_dcov3D
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    const float* cov3D = cov3Ds + 6 * idx;

    float3 mean = means[idx];
    float3 dL_dcov = { dL_dcovs[4 * idx], dL_dcovs[4 * idx + 1], dL_dcovs[4 * idx + 3] };
    float3 t = transformPoint4x3(mean, view_matrix);
    
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;
    
    const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
    const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

    glm::mat3 J = glm::mat3(
        h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
        0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
        0.0f, 0.0f, 0.0f);

    glm::mat3 W = glm::mat3(
        view_matrix[0], view_matrix[4], view_matrix[8],
        view_matrix[1], view_matrix[5], view_matrix[9],
        view_matrix[2], view_matrix[6], view_matrix[10]);

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 T = W * J;

    glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

    float a = cov2D[0][0];
    float b = cov2D[0][1];
    float c = cov2D[1][1];

    float denom = a * c - b * b;
    float dL_da = 0, dL_db = 0, dL_dc = 0;

    if (denom > 1e-12f) {
        
        dL_da = dL_dcov.x;
        dL_dc = dL_dcov.z;
        dL_db = dL_dcov.y;

        dL_dcov3D[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
        dL_dcov3D[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
        dL_dcov3D[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

        dL_dcov3D[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
        dL_dcov3D[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
        dL_dcov3D[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
    } else {
        for (int i = 0; i < 6; i++)
            dL_dcov3D[6 * idx + i] = 0;
    }

    float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
                        (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
    float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
                        (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
    float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
                         (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
    float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
                        (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
    float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
                        (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
    float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
                        (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

    float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    float tz = 1.f / t.z;
    float tz2 = tz * tz;
    float tz3 = tz2 * tz;

    float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
    float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
    float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

    float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

    dL_dmeans[idx] = dL_dmean;
}

__device__ void computeCov3D(
    int idx,
    const glm::vec3 scale,
    float mod,
    const glm::vec4 rot,
    const float* dL_dcov3Ds,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drots
) {
    
    glm::vec4 q = rot / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 S = glm::mat3(1.0f);

    glm::vec3 s = mod * scale;
    S[0][0] = s.x;
    S[1][1] = s.y;
    S[2][2] = s.z;

    glm::mat3 M = S * R;

    const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

    glm::mat3 dL_dSigma = glm::mat3(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
    );

    glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

    glm::mat3 Rt = glm::transpose(R);
    glm::mat3 dL_dMt = glm::transpose(dL_dM);

    glm::vec3* dL_dscale = dL_dscales + idx;
    dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
    dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
    dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

    dL_dMt[0] *= s.x;
    dL_dMt[1] *= s.y;
    dL_dMt[2] *= s.z;

    glm::vec4 dL_dq;
    dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

    float4* dL_drot = (float4*)(dL_drots + idx);
    *dL_drot = dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

__device__ float3 norm2d_grad(const float2 v, const float norm) {
    float norm3 = norm * norm * norm;
    norm3 = max(norm3, 1e-6f);  
    float3 mat = {
        v.y * v.y / norm3,
        -v.x * v.y / norm3,
        v.x * v.x / norm3
    };
    return mat;
}

template<int C>
__global__ void preprocessCUDA(
    const int P, int D, int M,
    const float3* means,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* proj,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    glm::vec3* dL_dmeans,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    float3 m = means[idx];

    float4 m_hom = transformPoint4x4(m, proj);
    float m_w = 1.0f / (m_hom.w + 0.0000001f);

    glm::vec3 dL_dmean;
    float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
    float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
    dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
    dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    dL_dmeans[idx] += dL_dmean;

    if (shs)
        computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

    if (scales)
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const int W, int H,
    const float* __restrict__ bg_color,
    const float* __restrict__ colors,
    const uint32_t* __restrict__ point_list,
    const float3* __restrict__ points_xy_image,
    const float4* __restrict__ cov_opacity,
    const float2* __restrict__ lambdas,
    const float4* __restrict__ nv1_nv2,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ final_color,
    const float* __restrict__ dL_dpixels,
    float3* __restrict__ dL_dmean2D,
    float4* __restrict__ dL_dcov2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors,
    const bool gaussian_window,
    const bool hybrid_window,
    const bool footprint_floor,
    const bool depth_coupled,
    const bool exact_window_grad,
    const bool adaptive_window,
    const float aniso_tau
) {
    
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool done = !inside;
    int toDo = range.y - range.x;

    __shared__ uint32_t alpha_id[BLOCK_SIZE];
    __shared__ float3 alpha_xy[BLOCK_SIZE];
    __shared__ float4 alpha_cov_o[BLOCK_SIZE];
    __shared__ float2 alpha_lambda[BLOCK_SIZE];
    __shared__ float4 alpha_nv1_nv2[BLOCK_SIZE];

    uint32_t contributor = 0;
    const int last_contributor = inside ? n_contrib[pix_id] : 0;
    float F[C] = { 0.0f };
    float current_color[C] = { 0.0f };
    if (inside)
    {
        for (int i = 0; i < C; i++)
            F[i] = final_color[i * H * W + pix_id];
    }
    float T_val = 1.0f;
    float2 T_xy = pixf;
    float2 T_wh = {1.0f, 1.0f};
    
    float sat_M0 = 1.0f;
    float sat_mux = 0.0f, sat_muy = 0.0f;
    float sat_sxx = 1.0f / 12.0f, sat_sxy = 0.0f, sat_syy = 1.0f / 12.0f;

    float dL_dpixel[C] = { 0.0f };
    if (inside)
    {
        for (int i = 0; i < C; i++)
        {
            dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
        }
    }

    const float ddelx_dx = 0.5 * W;
    const float ddely_dy = 0.5 * H;
    const float gaussian_max_radius = sqrtf(-2.0f * logf(MIN_ALPHA));

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        const int progress = range.x + i * BLOCK_SIZE + block.thread_rank();
        if (progress < range.y) {
            uint32_t coll_id = point_list[progress];
            alpha_id[block.thread_rank()] = coll_id;
            alpha_xy[block.thread_rank()] = points_xy_image[coll_id];
            alpha_cov_o[block.thread_rank()] = cov_opacity[coll_id];
            alpha_lambda[block.thread_rank()] = lambdas[coll_id];
            alpha_nv1_nv2[block.thread_rank()] = nv1_nv2[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor++;
            if(contributor > last_contributor) {
                done = true;
                break;
            }
            const float4 cov_o = alpha_cov_o[j];
            if(cov_o.w < MIN_ALPHA) continue; 
            const float2 xy = {alpha_xy[j].x, alpha_xy[j].y};

            if (gaussian_window || hybrid_window) {
                
                const float2 mu_s = { xy.x - pixf.x, xy.y - pixf.y };
                const float3 Sig_s = make_float3(cov_o.x, cov_o.y, cov_o.z);
                const float3 conic = inverse(Sig_s);
                const float o = cov_o.w;
                const float3 Sig_w = make_float3(sat_sxx, sat_sxy, sat_syy);
                const float2 dmu = { sat_mux - mu_s.x, sat_muy - mu_s.y };
                const float Tb = sat_M0;
                float a, cmask;
                if (hybrid_window) {
                    
                    const float lambda1 = alpha_lambda[j].x, lambda2 = alpha_lambda[j].y;
                    const float sigma1 = sqrtf(lambda1), sigma2 = sqrtf(fabsf(lambda2));
                    const float inv_sigma1 = safe_inverse(sigma1), inv_sigma2 = safe_inverse(sigma2);
                    const float2 v1r = { cov_o.y, lambda1 - cov_o.x }, v2r = { lambda2 - cov_o.z, cov_o.y };
                    const float norm_v1 = length(v1r), norm_v2 = length(v2r);
                    const float2 nv1 = { alpha_nv1_nv2[j].x, alpha_nv1_nv2[j].y };
                    const float2 nv2 = { alpha_nv1_nv2[j].z, alpha_nv1_nv2[j].w };
                    const float u = dmu.x * nv1.x + dmu.y * nv1.y;
                    const float v = dmu.x * nv2.x + dmu.y * nv2.y;
                    const float vw1 = nv1.x*nv1.x*sat_sxx + 2.0f*nv1.x*nv1.y*sat_sxy + nv1.y*nv1.y*sat_syy;
                    const float vw2 = nv2.x*nv2.x*sat_sxx + 2.0f*nv2.x*nv2.y*sat_sxy + nv2.y*nv2.y*sat_syy;
                    const float h1 = sqrtf(3.0f*fmaxf(vw1,1e-12f)), h2 = sqrtf(3.0f*fmaxf(vw2,1e-12f));
                    const float inv_area = 1.0f/fmaxf(4.0f*h1*h2,1e-20f);
                    const float U2=(u+h1)*inv_sigma1, U1=(u-h1)*inv_sigma1, V2=(v+h2)*inv_sigma2, V1=(v-h2)*inv_sigma2;
                    const float erf_U2=erff(U2*(float)M_INV_SQRT_2), erf_U1=erff(U1*(float)M_INV_SQRT_2);
                    const float erf_V2=erff(V2*(float)M_INV_SQRT_2), erf_V1=erff(V1*(float)M_INV_SQRT_2);
                    const float exp_U2=__expf(-0.5f*U2*U2), exp_U1=__expf(-0.5f*U1*U1);
                    const float exp_V2=__expf(-0.5f*V2*V2), exp_V1=__expf(-0.5f*V1*V1);
                    const float intU_0th = sigma1*(float)M_SQRT_PI_2*(erf_U2-erf_U1);
                    const float intV_0th = sigma2*(float)M_SQRT_PI_2*(erf_V2-erf_V1);
                    const float Kbox = intU_0th*intV_0th*inv_area;
                    a = fminf(o*fminf(Kbox,1.0f), 0.99f);
                    cmask = (o*fminf(Kbox,1.0f) < 0.99f) ? 1.0f : 0.0f;
                    const float weight = Tb*a;
                    float dL_da=0.0f;
                    for (int ch=0; ch<C; ch++) {
                        const float cc=colors[alpha_id[j]*C+ch];
                        current_color[ch]+=cc*weight;
                        dL_da += (Tb*cc + (current_color[ch]-F[ch])/(1.0f-a))*dL_dpixel[ch];
                        atomicAdd(&(dL_dcolors[alpha_id[j]*C+ch]), weight*dL_dpixel[ch]);
                    }
                    const float dL_dalpha = dL_da*cmask;
                    const float dL_do = dL_dalpha*intU_0th*intV_0th*inv_area;
                    atomicAdd(&(dL_dopacity[alpha_id[j]]), dL_do);
                    const float dL_dU_0th = dL_dalpha*o*intV_0th*inv_area;
                    const float dL_dV_0th = dL_dalpha*o*intU_0th*inv_area;
                    const float dL_dU2 = dL_dU_0th*sigma1*exp_U2;
                    const float dL_dU1 = -(dL_dU_0th*sigma1*exp_U1);
                    const float dL_dV2 = dL_dV_0th*sigma2*exp_V2;
                    const float dL_dV1 = -(dL_dV_0th*sigma2*exp_V1);
                    const float dL_du = (dL_dU2+dL_dU1)*inv_sigma1;
                    const float dL_dv = (dL_dV2+dL_dV1)*inv_sigma2;
                    const float dL_dsigma1 = dL_dU_0th*intU_0th*inv_sigma1 - dL_dU2*U2*inv_sigma1 - dL_dU1*U1*inv_sigma1;
                    const float dL_dsigma2 = dL_dV_0th*intV_0th*inv_sigma2 - dL_dV2*V2*inv_sigma2 - dL_dV1*V1*inv_sigma2;
                    const float2 dL_dd = dL_du*nv1 + dL_dv*nv2;
                    const float2 dL_dnv1 = dL_du*dmu;
                    const float2 dL_dnv2 = dL_dv*dmu;
                    const float2 dL_dxy = make_float2(-dL_dd.x*ddelx_dx, -dL_dd.y*ddely_dy);
                    atomicAdd(&dL_dmean2D[alpha_id[j]].x, dL_dxy.x);
                    atomicAdd(&dL_dmean2D[alpha_id[j]].y, dL_dxy.y);
                    atomicAdd(&dL_dmean2D[alpha_id[j]].z, length(dL_dxy));
                    float2 dL_dv1, dL_dv2;
                    if (norm_v1 < 1e-6f || norm_v2 < 1e-6f) { dL_dv1=make_float2(0.0f,0.0f); dL_dv2=make_float2(0.0f,0.0f); }
                    else {
                        const float3 dnv1_dv1 = norm2d_grad(v1r, norm_v1);
                        const float3 dnv2_dv2 = norm2d_grad(v2r, norm_v2);
                        dL_dv1 = matmul(dL_dnv1, dnv1_dv1);
                        dL_dv2 = matmul(dL_dnv2, dnv2_dv2);
                    }
                    const float3 dL_dv_dv_dcov = { -dL_dv1.y, dL_dv1.x+dL_dv2.y, -dL_dv2.x };
                    const float dL_dlambda1 = dL_dsigma1*0.5f*inv_sigma1 + dL_dv1.y;
                    const float dL_dlambda2 = dL_dsigma2*0.5f*inv_sigma2*((lambda2>=0.0f)?1.0f:-1.0f) + dL_dv2.x;
                    const float rootsq = alpha_xy[j].z;
                    const float denom = rootsq==0.0f ? 0.0f : 0.5f/rootsq;
                    const float3 dlambda1_dcov = { 0.5f+0.5f*(cov_o.x-cov_o.z)*denom, 2.0f*cov_o.y*denom, 0.5f+0.5f*(cov_o.z-cov_o.x)*denom };
                    const float3 dlambda2_dcov = { 0.5f-0.5f*(cov_o.x-cov_o.z)*denom, -2.0f*cov_o.y*denom, 0.5f-0.5f*(cov_o.z-cov_o.x)*denom };
                    const float3 dL_dcov = dL_dlambda1*dlambda1_dcov + dL_dlambda2*dlambda2_dcov + dL_dv_dv_dcov;
                    atomicAdd(&dL_dcov2D[alpha_id[j]].x, dL_dcov.x);
                    atomicAdd(&dL_dcov2D[alpha_id[j]].y, dL_dcov.y);
                    atomicAdd(&dL_dcov2D[alpha_id[j]].w, dL_dcov.z);
                } else {
                    
                    const float3 S = make_float3(sat_sxx+Sig_s.x, sat_sxy+Sig_s.y, sat_syy+Sig_s.z);
                    const float3 Sinv = inverse(S);
                    const float qf = fmaxf(Sinv.x*dmu.x*dmu.x + 2.0f*Sinv.y*dmu.x*dmu.y + Sinv.z*dmu.y*dmu.y, 0.0f);
                    const float detprod = fmaxf(determinant(conic)*determinant(S), 1e-20f);
                    float K=__expf(-0.5f*qf)*rsqrtf(detprod); K=fminf(K,1.0f);
                    const float oK=o*K; a=fminf(oK,0.99f); cmask=(oK<0.99f)?1.0f:0.0f;
                    const float weight=Tb*a;
                    float dL_da=0.0f;
                    for (int ch=0; ch<C; ch++) {
                        const float cc=colors[alpha_id[j]*C+ch];
                        current_color[ch]+=cc*weight;
                        dL_da += (Tb*cc + (current_color[ch]-F[ch])/(1.0f-a))*dL_dpixel[ch];
                        atomicAdd(&(dL_dcolors[alpha_id[j]*C+ch]), weight*dL_dpixel[ch]);
                    }
                    const float dL_dK = dL_da*o*cmask;
                    atomicAdd(&(dL_dopacity[alpha_id[j]]), dL_da*K*cmask);
                    const float2 g = { Sinv.x*dmu.x+Sinv.y*dmu.y, Sinv.y*dmu.x+Sinv.z*dmu.y };
                    const float2 dL_dxy_ndc = { dL_dK*K*g.x*ddelx_dx, dL_dK*K*g.y*ddely_dy };
                    atomicAdd(&dL_dmean2D[alpha_id[j]].x, dL_dxy_ndc.x);
                    atomicAdd(&dL_dmean2D[alpha_id[j]].y, dL_dxy_ndc.y);
                    atomicAdd(&dL_dmean2D[alpha_id[j]].z, length(dL_dxy_ndc));
                    const float D00=0.5f*(g.x*g.x+conic.x-Sinv.x), D01=0.5f*(g.x*g.y+conic.y-Sinv.y), D11=0.5f*(g.y*g.y+conic.z-Sinv.z);
                    atomicAdd(&dL_dcov2D[alpha_id[j]].x, dL_dK*K*D00);
                    atomicAdd(&dL_dcov2D[alpha_id[j]].y, dL_dK*K*2.0f*D01);
                    atomicAdd(&dL_dcov2D[alpha_id[j]].w, dL_dK*K*D11);
                }
                
                const float3 Siw_inv = inverse(Sig_w);
                const float3 Amat = make_float3(Siw_inv.x+conic.x, Siw_inv.y+conic.y, Siw_inv.z+conic.z);
                const float3 Sig3 = inverse(Amat);
                const float2 tt = { Siw_inv.x*sat_mux+Siw_inv.y*sat_muy+conic.x*mu_s.x+conic.y*mu_s.y,
                                    Siw_inv.y*sat_mux+Siw_inv.z*sat_muy+conic.y*mu_s.x+conic.z*mu_s.y };
                const float2 mu3 = { Sig3.x*tt.x+Sig3.y*tt.y, Sig3.y*tt.x+Sig3.z*tt.y };
                const float M0n = sat_M0*(1.0f-a);
                if (M0n < 0.0001f) { sat_M0 = M0n; done=true; break; }
                const float inv1 = 1.0f/(1.0f-a);
                const float2 mu_wn = { (sat_mux-a*mu3.x)*inv1, (sat_muy-a*mu3.y)*inv1 };
                const float2 da_ = { sat_mux-mu_wn.x, sat_muy-mu_wn.y };
                const float2 db_ = { mu3.x-mu_wn.x, mu3.y-mu_wn.y };
                const float invM0n = 1.0f/M0n;
                float3 Sig_wn = make_float3(
                    (sat_M0*(sat_sxx+da_.x*da_.x) - sat_M0*a*(Sig3.x+db_.x*db_.x))*invM0n,
                    (sat_M0*(sat_sxy+da_.x*da_.y) - sat_M0*a*(Sig3.y+db_.x*db_.y))*invM0n,
                    (sat_M0*(sat_syy+da_.y*da_.y) - sat_M0*a*(Sig3.z+db_.y*db_.y))*invM0n);
                Sig_wn = sat_psd_clamp(Sig_wn, footprint_floor ? (1.0f / 12.0f) : 1e-4f); 
                sat_sxx=Sig_wn.x; sat_sxy=Sig_wn.y; sat_syy=Sig_wn.z;
                sat_mux=mu_wn.x; sat_muy=mu_wn.y; sat_M0=M0n;
                continue;
            }
            const float root_sqrt = alpha_xy[j].z;
            const float2 d = T_xy - xy;
            const float lambda1 = alpha_lambda[j].x, lambda2 = alpha_lambda[j].y;
            const float sigma1 = sqrtf(lambda1), sigma2 = sqrtf(abs(lambda2));
            const float2 v1 = {cov_o.y, lambda1 - cov_o.x}, v2 = {lambda2 - cov_o.z, cov_o.y};
            const float norm_v1 = length(v1), norm_v2 = length(v2);
            const float2 nv1 = {alpha_nv1_nv2[j].x, alpha_nv1_nv2[j].y}, nv2 = {alpha_nv1_nv2[j].z, alpha_nv1_nv2[j].w};

            const float2 uv = {dot(d, nv1), dot(d, nv2)};
            const float2 uv_length = (abs(nv1.x) > abs(nv1.y)) ? T_wh : reverse(T_wh); 
            if (uv.x + 0.5f * uv_length.x < -gaussian_max_radius * sigma1 || uv.x - 0.5f * uv_length.x > gaussian_max_radius * sigma1 || uv.y + 0.5f * uv_length.y < -gaussian_max_radius * sigma2 || uv.y - 0.5f * uv_length.y > gaussian_max_radius * sigma2)
                continue;

            const float inv_sigma1 = safe_inverse(sigma1);
            const float inv_sigma2 = safe_inverse(sigma2);
            const float U2 = (uv.x + 0.5f * uv_length.x) * inv_sigma1, U1 = (uv.x - 0.5f * uv_length.x) * inv_sigma1;
            const float V2 = (uv.y + 0.5f * uv_length.y) * inv_sigma2, V1 = (uv.y - 0.5f * uv_length.y) * inv_sigma2;

            if (U2 - U1 < MIN_UV_DIFF || V2 - V1 < MIN_UV_DIFF || U2 - U1 > MAX_UV_DIFF || V2 - V1 > MAX_UV_DIFF)
            {
                const float3 conic = inverse(make_float3(cov_o));
                const float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
                if(power > 0.0f) continue;
                const float alpha = min(0.99f, cov_o.w * exp(power));
                if(alpha < MIN_ALPHA) continue; 
                const float weight = T_val * alpha;

                float dL_dalpha = 0.0f;
                for(int ch = 0; ch < C; ch++)
                {
                    const float c = colors[alpha_id[j] * C + ch];
                    current_color[ch] += c * weight;
                    dL_dalpha += (T_val * c + (current_color[ch] - F[ch]) / (1.0f - alpha)) * dL_dpixel[ch];
                    atomicAdd(&(dL_dcolors[alpha_id[j] * C + ch]), weight * dL_dpixel[ch]);
                }

                const float dL_do = dL_dalpha * exp(power);
                atomicAdd(&(dL_dopacity[alpha_id[j]]), dL_do);
                const float dL_dpower = dL_dalpha * alpha;
                const float3 dL_dconic = dL_dpower * make_float3(-0.5f * d.x * d.x, -d.x * d.y, -0.5f * d.y * d.y);
                const float2 dL_dd = dL_dpower * make_float2(-conic.x * d.x - conic.y * d.y, -conic.z * d.y - conic.y * d.x);
                const float3 dL_dcov = {
                    -dL_dconic.x * conic.x * conic.x + dL_dconic.y * conic.x * conic.y - dL_dconic.z * conic.y * conic.y,
                    dL_dconic.x * 2.0f * conic.x * conic.y - dL_dconic.y * (conic.x * conic.z + conic.y * conic.y) + dL_dconic.z * 2.0f * conic.y * conic.z,
                    -dL_dconic.x * conic.y * conic.y + dL_dconic.y * conic.y * conic.z - dL_dconic.z * conic.z * conic.z
                };
                atomicAdd(&dL_dcov2D[alpha_id[j]].x, dL_dcov.x);
                atomicAdd(&dL_dcov2D[alpha_id[j]].y, dL_dcov.y);
                atomicAdd(&dL_dcov2D[alpha_id[j]].w, dL_dcov.z);
                const float2 dL_dxy = -dL_dd * make_float2(ddelx_dx, ddely_dy);
                atomicAdd(&dL_dmean2D[alpha_id[j]].x, dL_dxy.x);
                atomicAdd(&dL_dmean2D[alpha_id[j]].y, dL_dxy.y);
                atomicAdd(&dL_dmean2D[alpha_id[j]].z, length(dL_dxy));

                T_val = T_val - weight;
                if (T_val < 0.0001f) {
                    done = true;
                    break;
                }
                continue;
            }

            const float inv_area = 1.0f / (T_wh.x * T_wh.y);

            const float erf_U2 = erff(U2 * (float)M_INV_SQRT_2);
            const float erf_U1 = erff(U1 * (float)M_INV_SQRT_2);
            const float erf_V2 = erff(V2 * (float)M_INV_SQRT_2);
            const float erf_V1 = erff(V1 * (float)M_INV_SQRT_2);
            const float exp_U2 = __expf(-0.5f * U2 * U2);
            const float exp_U1 = __expf(-0.5f * U1 * U1);
            const float exp_V2 = __expf(-0.5f * V2 * V2);
            const float exp_V1 = __expf(-0.5f * V1 * V1);

            const float intU_0th = sigma1 * (float)M_SQRT_PI_2 * (erf_U2 - erf_U1);
            const float intV_0th = sigma2 * (float)M_SQRT_PI_2 * (erf_V2 - erf_V1);
            const float average_alpha = min(0.99f, cov_o.w * intU_0th * intV_0th * inv_area);
            if(average_alpha < MIN_ALPHA) continue; 

            const float weight = T_val * average_alpha;

            float dL_dalpha = 0.0f;
            for(int ch = 0; ch < C; ch++)
            {
                const float c = colors[alpha_id[j] * C + ch];
                current_color[ch] += c * weight;
                dL_dalpha += (T_val * c + (current_color[ch] - F[ch]) / (1.0f - average_alpha)) * dL_dpixel[ch];
                atomicAdd(&(dL_dcolors[alpha_id[j] * C + ch]), weight * dL_dpixel[ch]);
            }

            const float dL_do = dL_dalpha * intU_0th * intV_0th * inv_area;
            atomicAdd(&(dL_dopacity[alpha_id[j]]), dL_do);
            const float dL_dU_0th = dL_dalpha * cov_o.w * intV_0th * inv_area;
            const float dL_dV_0th = dL_dalpha * cov_o.w * intU_0th * inv_area;
            
            const float dL_dU2 = dL_dU_0th * sigma1 * exp_U2;
            const float dL_dU1 = -(dL_dU_0th * sigma1 * exp_U1);
            const float dL_dV2 = dL_dV_0th * sigma2 * exp_V2;
            const float dL_dV1 = -(dL_dV_0th * sigma2 * exp_V1);

            const float dL_du = (dL_dU2 + dL_dU1) * inv_sigma1;
            const float dL_dv = (dL_dV2 + dL_dV1) * inv_sigma2;
            const float dL_dsigma1 = dL_dU_0th * intU_0th * inv_sigma1 - dL_dU2 * U2 * inv_sigma1 - dL_dU1 * U1 * inv_sigma1;
            const float dL_dsigma2 = dL_dV_0th * intV_0th * inv_sigma2 - dL_dV2 * V2 * inv_sigma2 - dL_dV1 * V1 * inv_sigma2;

            const float2 dL_dd = dL_du * nv1 + dL_dv * nv2;
            const float2 dL_dnv1 = dL_du * d;
            const float2 dL_dnv2 = dL_dv * d;

            const float2 dL_dalpha_xy = -dL_dd * make_float2(ddelx_dx, ddely_dy);
            atomicAdd(&dL_dmean2D[alpha_id[j]].x, dL_dalpha_xy.x);
            atomicAdd(&dL_dmean2D[alpha_id[j]].y, dL_dalpha_xy.y);
            atomicAdd(&dL_dmean2D[alpha_id[j]].z, length(dL_dalpha_xy));

            float2 dL_dv1, dL_dv2;
            if (norm_v1 < 1e-6f || norm_v2 < 1e-6f) {
                dL_dv1 = {0.0f, 0.0f};
                dL_dv2 = {0.0f, 0.0f};
            } else {
                const float3 dnv1_dv1 = norm2d_grad(v1, norm_v1);
                const float3 dnv2_dv2 = norm2d_grad(v2, norm_v2);
                dL_dv1 = matmul(dL_dnv1, dnv1_dv1);
                dL_dv2 = matmul(dL_dnv2, dnv2_dv2);
            }
            const float3 dL_dv_dv_dcov = {
                -dL_dv1.y,
                dL_dv1.x + dL_dv2.y,
                -dL_dv2.x,
            };
            const float dL_dlambda1 = dL_dsigma1 * 0.5f * inv_sigma1 + dL_dv1.y;
            const float dL_dlambda2 = dL_dsigma2 * 0.5f * inv_sigma2 * ((lambda2 >= 0.0f) ? 1.0f : -1.0f) + dL_dv2.x;
            
            const float denom = root_sqrt == 0.0f ? 0.0f : 0.5f / root_sqrt; 
            const float3 dlambda1_dcov = {
                0.5f + 0.5f * (cov_o.x - cov_o.z) * denom,
                2.0f * cov_o.y * denom,
                0.5f + 0.5f * (cov_o.z - cov_o.x) * denom
            };
            const float3 dlambda2_dcov = {
                0.5f - 0.5f * (cov_o.x - cov_o.z) * denom,
                -2.0f * cov_o.y * denom,
                0.5f - 0.5f * (cov_o.z - cov_o.x) * denom
            };

            const float3 dL_dcov = dL_dlambda1 * dlambda1_dcov + dL_dlambda2 * dlambda2_dcov + dL_dv_dv_dcov;
            atomicAdd(&dL_dcov2D[alpha_id[j]].x, dL_dcov.x);
            atomicAdd(&dL_dcov2D[alpha_id[j]].y, dL_dcov.y);
            atomicAdd(&dL_dcov2D[alpha_id[j]].w, dL_dcov.z);

            const float intU_1st = sigma1 * sigma1 * (exp_U1 - exp_U2);
            const float intV_1st = sigma2 * sigma2 * (exp_V1 - exp_V2);
            const float intU_2nd = sigma1 * sigma1 * sigma1 * ((float)M_SQRT_PI_2 * (erf_U2 - erf_U1) - exp_U2 * U2 + exp_U1 * U1);
            const float intV_2nd = sigma2 * sigma2 * sigma2 * ((float)M_SQRT_PI_2 * (erf_V2 - erf_V1) - exp_V2 * V2 + exp_V1 * V1);

            const float T_oia = T_val * cov_o.w * inv_area;
            const float m_0 = T_val - weight;
            const float2 m_1 = make_float2(T_val * uv.x - T_oia * intU_1st * intV_0th,
                                          T_val * uv.y - T_oia * intU_0th * intV_1st);
            const float2 m_2 = make_float2(T_val * (uv.x * uv.x + (uv_length.x * uv_length.x) / 12.0f) - T_oia * intU_2nd * intV_0th,
                                          T_val * (uv.y * uv.y + (uv_length.y * uv_length.y) / 12.0f) - T_oia * intU_0th * intV_2nd);
            const float inv_m_0 = safe_inverse(m_0);
            
            T_val = m_0;
            T_xy = xy + m_1.x * inv_m_0 * nv1 + m_1.y * inv_m_0 * nv2;
            const float2 T_var = fmaxf(m_2 * inv_m_0 - m_1 * m_1 * inv_m_0 * inv_m_0, make_float2(0.001f, 0.001f));
            T_wh = (abs(nv1.x) > abs(nv1.y)) ? sqrtf(12.0f * T_var) : sqrtf(12.0f * reverse(T_var));
            if (T_val < 0.0001f) {
                done = true;
                break;
            }
        }
    }
}

void BACKWARD::preprocess(
    const int P, const int D, const int M,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    const float3* means3D,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* campos,
    const float3* dL_dmean2D,
    const float* dL_dcov,
    glm::vec3* dL_dmean3D,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot
) {

    computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
        P,
        means3D,
        radii,
        cov3Ds,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        viewmatrix,
        dL_dcov,
        (float3*)dL_dmean3D,
        dL_dcov3D);

    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P, D, M,
        (float3*)means3D,
        radii,
        shs,
        clamped,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        projmatrix,
        campos,
        (float3*)dL_dmean2D,
        (glm::vec3*)dL_dmean3D,
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        dL_dscale,
        dL_drot);
}

void BACKWARD::render(
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
    const uint32_t* n_contrib,
    const float* final_color,
    const float* dL_dpixels,
    float3* dL_dmean2D,
    float4* dL_dcov2D,
    float* dL_dopacity,
    float* dL_dcolors,
    const bool gaussian_window,
    const bool hybrid_window,
    const bool footprint_floor,
    const bool depth_coupled,
    const bool exact_window_grad,
    const bool adaptive_window,
    const float aniso_tau
) {
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        W, H,
        bg_color,
        colors,
        point_list,
        means2D,
        cov_opacity,
        lambdas,
        nv1_nv2,
        ranges,
        n_contrib,
        final_color,
        dL_dpixels,
        dL_dmean2D,
        dL_dcov2D,
        dL_dopacity,
        dL_dcolors,
        gaussian_window,
        hybrid_window,
        footprint_floor,
        depth_coupled,
        exact_window_grad,
        adaptive_window,
        aniso_tau
    );
}
