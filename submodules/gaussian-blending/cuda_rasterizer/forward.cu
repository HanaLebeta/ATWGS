#include "forward.h"
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

__device__ glm::vec3 computeColorFromSH(
    const int idx,
    const int deg,
    const int max_coeffs,
    const glm::vec3* means,
    const glm::vec3 campos,
    const float* shs,
    bool* clamped
) {

    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2) {
                result = result +
                    SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                    SH_C3[1] * xy * z * sh[10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                    SH_C3[5] * z * (xx - yy) * sh[14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}

__device__ float3 computeCov2D(
    const float3& mean,
    float focal_x,
    float focal_y,
    float tan_fovx,
    float tan_fovy,
    const float* cov3D,
    const float* viewmatrix
) {

    float3 t = transformPoint4x3(mean, viewmatrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.0f, 0.0f, 0.0f);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(
    const glm::vec3 scale,
    float mod,
    const glm::vec4 rot,
    float* cov3D
) {
    
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

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

    glm::mat3 M = S * R;

    glm::mat3 Sigma = glm::transpose(M) * M;

    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

template<int C>
__global__ void preprocessCUDA(
    const dim3 grid,
    const int P, int D, int M,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    const float* __restrict__ orig_points,
    const glm::vec3* __restrict__ scales,
    const float scale_modifier,
    const glm::vec4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float* __restrict__ shs,
    const float* __restrict__ cov3D_precomp,
    const float* __restrict__ colors_precomp,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const glm::vec3* __restrict__ cam_pos,
    int* __restrict__ radii,
    bool* __restrict__ clamped,
    float3* __restrict__ points_xy_image,
    float* __restrict__ depths,
    float* __restrict__ cov3Ds,
    float* __restrict__ rgb,
    float4* __restrict__ cov_opacity,
    float2* __restrict__ lambdas,
    float4* __restrict__ nv1_nv2,
    bool prefiltered
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    radii[idx] = 0;

    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    const float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    const float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    const float p_w = 1.0f / (p_hom.w + 0.0000001f);
    const float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    const float* cov3D;
    if (cov3D_precomp != nullptr) {
        cov3D = cov3D_precomp + idx * 6;
    } else {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    const float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    const float det = determinant(cov);
    if (det < 1e-12f)
        return;

    const float half_trace = 0.5f * (cov.x + cov.z);
    const float root = half_trace * half_trace - det;
    const float root_sqrt = sqrtf(max(0.0f, root));
    const float lambda1 = half_trace + root_sqrt;
    const float lambda2 = half_trace - root_sqrt;
    const float lambda1_ = half_trace + sqrt(max(0.1f, root));
    const float lambda2_ = half_trace - sqrt(max(0.1f, root));
    const float3 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H), root_sqrt};

    uint2 rect_min, rect_max;
    getRectTight(point_image, cov.x, cov.z, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    const int my_radius = ceil(sqrtf(-2.0f * logf(MIN_ALPHA)) * sqrtf(max(0.0f, max(lambda1_, lambda2_))));

    float2 v1 = {cov.y, lambda1 - cov.x};
    float2 v2 = {lambda2 - cov.z, cov.y};
    if (length(v1) < 1e-6f || length(v2) < 1e-6f) {
        v1 = {1.0f, 0.0f};
        v2 = {0.0f, 1.0f};
    } else {
        v1 = normalize(v1);
        v2 = normalize(v2);
    }

    if (colors_precomp == nullptr) {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    depths[idx] = p_view.z;
    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;
    
    cov_opacity[idx] = { cov.x, cov.y, cov.z, opacities[idx] };
    lambdas[idx] = {lambda1, lambda2};
    nv1_nv2[idx] = {v1.x, v1.y, v2.x, v2.y};
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const int W, int H,
    const float* __restrict__ bg_color,
    const float* __restrict__ features,
    const uint32_t* __restrict__ point_list,
    const float3* __restrict__ points_xy_image,
    const float4* __restrict__ cov_opacity,
    const float2* __restrict__ lambdas,
    const float4* __restrict__ nv1_nv2,
    const uint2* __restrict__ ranges,
    uint32_t* __restrict__ n_contrib,
    float* __restrict__ final_color,
    float* __restrict__ out_color,
    float* __restrict__ out_window,
    const bool gaussian_window,
    const bool hybrid_window,
    const bool footprint_floor,
    const bool depth_coupled,
    const bool exact_window_grad,
    const bool adaptive_window,
    const float aniso_tau,
    const bool point_sample
) {
    
    const auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    
    bool done = !inside;

    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ uint32_t alpha_id[BLOCK_SIZE];
    __shared__ float2 alpha_xy[BLOCK_SIZE];
    __shared__ float4 alpha_cov_o[BLOCK_SIZE];
    __shared__ float2 alpha_sigma[BLOCK_SIZE];
    __shared__ float4 alpha_nv1_nv2[BLOCK_SIZE];

    float T_val = 1.0f; 
    float2 T_xy = pixf;
    float2 T_wh = {1.0f, 1.0f};
    
    float sat_M0 = 1.0f;
    float sat_mux = 0.0f, sat_muy = 0.0f;
    float sat_sxx = 1.0f / 12.0f, sat_sxy = 0.0f, sat_syy = 1.0f / 12.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float C[CHANNELS] = { 0.0f };
    const float gaussian_max_radius = sqrtf(-2.0f * logf(MIN_ALPHA));

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        const int progress = range.x + i * BLOCK_SIZE + block.thread_rank();
        if (progress < range.y) {
            uint32_t coll_id = point_list[progress];
            alpha_id[block.thread_rank()] = coll_id;
            float3 alpha_xy_sqrtrt = points_xy_image[coll_id];
            alpha_xy[block.thread_rank()] = {alpha_xy_sqrtrt.x, alpha_xy_sqrtrt.y};
            alpha_cov_o[block.thread_rank()] = cov_opacity[coll_id];
            float2 l1_l2 = lambdas[coll_id];
            alpha_sigma[block.thread_rank()] = {sqrtf(l1_l2.x), sqrtf(abs(l1_l2.y))};
            alpha_nv1_nv2[block.thread_rank()] = nv1_nv2[coll_id];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            contributor++;
            const float4 cov_o = alpha_cov_o[j];
            if(cov_o.w < MIN_ALPHA) continue; 
            const float2 xy = alpha_xy[j];

            if (point_sample) {
                
                const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                const float3 con = inverse(make_float3(cov_o.x, cov_o.y, cov_o.z));
                const float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
                if (power > 0.0f) continue;
                const float alpha = fminf(0.99f, cov_o.w * __expf(power));
                if (alpha < MIN_ALPHA) continue;
                const float weight = T_val * alpha;
                for (int ch = 0; ch < CHANNELS; ch++)
                    C[ch] += features[alpha_id[j] * CHANNELS + ch] * weight;
                last_contributor = contributor;
                T_val = T_val - weight;
                if (T_val < 0.0001f) { done = true; break; }
                continue;
            }

            if (gaussian_window || hybrid_window || adaptive_window) {

                const float2 mu_s = { xy.x - pixf.x, xy.y - pixf.y };
                const float3 Sig_s = make_float3(cov_o.x, cov_o.y, cov_o.z); 
                const float o = cov_o.w;
                const float3 Sig_w = make_float3(sat_sxx, sat_sxy, sat_syy);
                const float2 dmu = { sat_mux - mu_s.x, sat_muy - mu_s.y };

                bool path_hybrid = hybrid_window;
                bool path_iso = false;
                if (adaptive_window) {
                    const float tr = Sig_s.x + Sig_s.z;
                    const float det = Sig_s.x * Sig_s.z - Sig_s.y * Sig_s.y;
                    const float disc = sqrtf(fmaxf(tr * tr - 4.0f * det, 0.0f));
                    const float l1 = 0.5f * (tr + disc), l2 = 0.5f * (tr - disc);
                    const float rho = l1 / fmaxf(l2, 1e-12f);
                    path_hybrid = (rho > aniso_tau);   
                    path_iso = !path_hybrid;           
                }
                float a;
                float3 conic;
                if (path_iso) {
                    
                    const float ss = sqrtf(0.5f * (Sig_s.x + Sig_s.z));
                    const float inv_ss2 = safe_inverse(ss * ss);
                    conic = make_float3(inv_ss2, 0.0f, inv_ss2);
                    const float h1 = sqrtf(3.0f * fmaxf(sat_sxx, 1e-12f));
                    const float h2 = sqrtf(3.0f * fmaxf(sat_syy, 1e-12f));
                    const float is = safe_inverse(ss);
                    const float U2 = (dmu.x + h1) * is, U1 = (dmu.x - h1) * is;
                    const float V2 = (dmu.y + h2) * is, V1 = (dmu.y - h2) * is;
                    const float intU = ss * (float)M_SQRT_PI_2 * (erff(U2 * (float)M_INV_SQRT_2) - erff(U1 * (float)M_INV_SQRT_2));
                    const float intV = ss * (float)M_SQRT_PI_2 * (erff(V2 * (float)M_INV_SQRT_2) - erff(V1 * (float)M_INV_SQRT_2));
                    const float Kbox = intU * intV / fmaxf(4.0f * h1 * h2, 1e-20f);
                    a = fminf(o * fminf(Kbox, 1.0f), 0.99f);
                } else if (path_hybrid) {
                    conic = inverse(Sig_s);

                    const float sigma1 = alpha_sigma[j].x, sigma2 = alpha_sigma[j].y;
                    const float2 e1 = { alpha_nv1_nv2[j].x, alpha_nv1_nv2[j].y };
                    const float2 e2 = { alpha_nv1_nv2[j].z, alpha_nv1_nv2[j].w };
                    const float u = dmu.x * e1.x + dmu.y * e1.y;
                    const float v = dmu.x * e2.x + dmu.y * e2.y;
                    const float vw1 = e1.x * e1.x * sat_sxx + 2.0f * e1.x * e1.y * sat_sxy + e1.y * e1.y * sat_syy;
                    const float vw2 = e2.x * e2.x * sat_sxx + 2.0f * e2.x * e2.y * sat_sxy + e2.y * e2.y * sat_syy;
                    const float h1 = sqrtf(3.0f * fmaxf(vw1, 1e-12f));
                    const float h2 = sqrtf(3.0f * fmaxf(vw2, 1e-12f));
                    const float is1 = safe_inverse(sigma1), is2 = safe_inverse(sigma2);
                    const float U2 = (u + h1) * is1, U1 = (u - h1) * is1;
                    const float V2 = (v + h2) * is2, V1 = (v - h2) * is2;
                    const float intU = sigma1 * (float)M_SQRT_PI_2 * (erff(U2 * (float)M_INV_SQRT_2) - erff(U1 * (float)M_INV_SQRT_2));
                    const float intV = sigma2 * (float)M_SQRT_PI_2 * (erff(V2 * (float)M_INV_SQRT_2) - erff(V1 * (float)M_INV_SQRT_2));
                    const float Kbox = intU * intV / fmaxf(4.0f * h1 * h2, 1e-20f);
                    a = fminf(o * fminf(Kbox, 1.0f), 0.99f);
                } else {
                    conic = inverse(Sig_s);
                    
                    const float3 S = make_float3(sat_sxx + Sig_s.x, sat_sxy + Sig_s.y, sat_syy + Sig_s.z);
                    const float3 Sinv = inverse(S);
                    const float qf = fmaxf(Sinv.x * dmu.x * dmu.x + 2.0f * Sinv.y * dmu.x * dmu.y + Sinv.z * dmu.y * dmu.y, 0.0f);
                    const float detprod = fmaxf(determinant(conic) * determinant(S), 1e-20f);
                    float K = __expf(-0.5f * qf) * rsqrtf(detprod);
                    K = fminf(K, 1.0f);
                    a = fminf(o * K, 0.99f);   
                }
                const float w = sat_M0 * a;

                {
                    const float wtr = sat_sxx + sat_syy;
                    const float wdet = sat_sxx * sat_syy - sat_sxy * sat_sxy;
                    const float wdisc = sqrtf(fmaxf(wtr * wtr - 4.0f * wdet, 0.0f));
                    const float wl1 = 0.5f * (wtr + wdisc), wl2 = 0.5f * (wtr - wdisc);
                    const float aniso = wl1 / fmaxf(wl2, 1e-12f);
                    const uint32_t gid = alpha_id[j];
                    atomicAdd(&out_window[2 * gid + 0], w * aniso);
                    atomicAdd(&out_window[2 * gid + 1], w);
                }
                last_contributor = contributor;
                for (int ch = 0; ch < CHANNELS; ch++)
                    C[ch] += features[alpha_id[j] * CHANNELS + ch] * w;
                const float3 Siw_inv = inverse(Sig_w);
                const float3 A = make_float3(Siw_inv.x + conic.x, Siw_inv.y + conic.y, Siw_inv.z + conic.z);
                const float3 Sig3 = inverse(A);
                const float2 tt = { Siw_inv.x * sat_mux + Siw_inv.y * sat_muy + conic.x * mu_s.x + conic.y * mu_s.y,
                                    Siw_inv.y * sat_mux + Siw_inv.z * sat_muy + conic.y * mu_s.x + conic.z * mu_s.y };
                const float2 mu3 = { Sig3.x * tt.x + Sig3.y * tt.y, Sig3.y * tt.x + Sig3.z * tt.y };
                const float M0n = sat_M0 * (1.0f - a);
                if (M0n < 0.0001f) { sat_M0 = M0n; done = true; break; } 
                const float inv1 = 1.0f / (1.0f - a);
                const float2 mu_wn = { (sat_mux - a * mu3.x) * inv1, (sat_muy - a * mu3.y) * inv1 };
                const float2 da = { sat_mux - mu_wn.x, sat_muy - mu_wn.y };
                const float2 db = { mu3.x - mu_wn.x, mu3.y - mu_wn.y };
                const float invM0n = 1.0f / M0n;
                float3 Sig_wn = make_float3(
                    (sat_M0 * (sat_sxx + da.x * da.x) - sat_M0 * a * (Sig3.x + db.x * db.x)) * invM0n,
                    (sat_M0 * (sat_sxy + da.x * da.y) - sat_M0 * a * (Sig3.y + db.x * db.y)) * invM0n,
                    (sat_M0 * (sat_syy + da.y * da.y) - sat_M0 * a * (Sig3.z + db.y * db.y)) * invM0n);
                Sig_wn = sat_psd_clamp(Sig_wn, footprint_floor ? (1.0f / 12.0f) : 1e-4f); 
                sat_sxx = Sig_wn.x; sat_sxy = Sig_wn.y; sat_syy = Sig_wn.z;
                sat_mux = mu_wn.x; sat_muy = mu_wn.y; sat_M0 = M0n;
                continue;
            }

            const float2 d = T_xy - xy;
            const float2 nv1 = {alpha_nv1_nv2[j].x, alpha_nv1_nv2[j].y}, nv2 = {alpha_nv1_nv2[j].z, alpha_nv1_nv2[j].w};
            const float sigma1 = alpha_sigma[j].x, sigma2 = alpha_sigma[j].y;

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

                for (int ch = 0; ch < CHANNELS; ch++)
                    C[ch] += features[alpha_id[j] * CHANNELS + ch] * weight;
                last_contributor = contributor;
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

            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += features[alpha_id[j] * CHANNELS + ch] * weight;
            last_contributor = contributor;

            const float intU_1st = sigma1 * sigma1 * ((exp_U1 - exp_U2) - 0.0f);  
            const float intV_1st = sigma2 * sigma2 * ((exp_V1 - exp_V2) - 0.0f);

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

    if (inside) {
        n_contrib[pix_id] = last_contributor;
        const float Tbg = (gaussian_window || hybrid_window || adaptive_window) ? sat_M0 : T_val; 
        for (int ch = 0; ch < CHANNELS; ch++)
        {
            final_color[ch * H * W + pix_id] = C[ch] + Tbg * bg_color[ch];
            out_color[ch * H * W + pix_id] = C[ch] + Tbg * bg_color[ch];
        }
    }
}

void FORWARD::render(
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
        out_color,
        out_window,
        gaussian_window,
        hybrid_window,
        footprint_floor,
        depth_coupled,
        exact_window_grad,
        adaptive_window,
        aniso_tau,
        point_sample);
}

void FORWARD::preprocess(
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
) {
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        grid,
        P, D, M,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        means3D,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, 
        projmatrix,
        cam_pos,
        radii,
        clamped,
        means2D,
        depths,
        cov3Ds,
        rgb,
        cov_opacity,
        lambdas,
        nv1_nv2,
        prefiltered
    );
}
