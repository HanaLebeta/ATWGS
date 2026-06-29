<div align="center">

# ATW-GS

### Anisotropic Transmittance Windows for Anti-Aliased 3D Gaussian Splatting

<p>
<a href="#">
<img alt="Paper" src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
</a>
<a href="https://hanalebeta.github.io/ATW_GS/">
<img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4A90D9?style=for-the-badge&logo=githubpages&logoColor=white">
</a>
<a href="#">
<img alt="Code" src="https://img.shields.io/badge/Code-Available-blue?style=for-the-badge&logo=github&logoColor=white">
</a>
<a href="#">
<img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="LICENSE">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</a>
</p>

Hana L. Goshu<sup>1</sup>, Tadesse G. Wakjira<sup>2</sup>, Kin-Man Lam<sup>1</sup>

<sup>1</sup>The Hong Kong Polytechnic University &nbsp;&nbsp; <sup>2</sup>Kennesaw State University

*Preprint, 2026 (Under Review)*

</div>

---

## Method

<div align="center">
<img src="static/images/architecture.png" width="95%">
<br>
<em>ATW-GS propagates an anisotropic Gaussian transmittance window (centre, 2&times;2 covariance, mass) through front-to-back composition, preserving splat orientation within each pixel.</em>
</div>

---

<div align="center">
<img src="static/images/qualitative.png" width="95%">
<br>
<em>Multi-scale comparison on Mip-NeRF 360: ATW-GS suppresses the dilation that 3D Gaussian Splatting exhibits at sampling rates unseen during training, while preserving quality at the training rate.</em>
</div>

---

## Abstract

3D Gaussian Splatting (3DGS) enables real-time novel view synthesis but suffers from aliasing artifacts when rendered at sampling rates that differ from those used during training. Existing anti-aliasing methods mitigate this issue by prefiltering Gaussian primitives or integrating their response over the pixel area; however, they rely on scalar alpha composition, which treats per-pixel transmittance as a single value and therefore discards the spatial structure within each pixel. More recent spatially aware approaches maintain a per-pixel transmittance window, but represent it as an axis-aligned box, preventing the model from capturing the true orientation of projected splats. To address this limitation, we propose **Anisotropic Transmittance Windows for 3D Gaussian Splatting (ATW-GS)**, which models per-pixel transmittance as a full covariance Gaussian that preserves splat orientation throughout the front-to-back composition sequence. Furthermore, ATW-GS computes each splat's contribution weight using the exact error-function integral of the pixel footprint projected onto the window's marginal axes. This ensures that consistency between the anisotropic window state and the per-splat weight computation. Extensive experiments on the NeRF Synthetic (Blender) and Mip-NeRF 360 datasets demonstrate that ATW-GS consistently outperforms state-of-the-art aliasing methods across both single-scale and multi-scale evaluation regimes.

## Installation

Tested with Python 3.10, PyTorch 2.7 (CUDA 12.x), and a recent NVIDIA GPU.

```bash
git clone https://github.com/HanaLebeta/ATW_GS.git
cd ATW_GS

conda create -n atw-gs python=3.10 -y
conda activate atw-gs

pip install torch torchvision torchaudio
pip install -r requirements.txt

export TORCH_CUDA_ARCH_LIST="8.6+PTX"
pip install ./submodules/simple-knn
pip install ./submodules/gaussian-blending
```

Set `TORCH_CUDA_ARCH_LIST` to the compute capability of your GPU (e.g. `8.6`, `8.9`, `12.0`).

## Data

ATW-GS is evaluated on the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset and the NeRF Synthetic (Blender) dataset. Point `-s` at a scene directory in the standard 3DGS layout.

## Training

The window mode is selected by a single flag; all other settings follow the 3DGS defaults. ATW-GS uses the anisotropic window via `--hybrid_window`.

```bash
python train.py -m output/scene_atw -s <path>/mipnerf360/<scene> --eval --filter3d -r 4 --hybrid_window
python train.py -m output/scene_atw -s <path>/nerf_synthetic/<scene> --eval --white_background --hybrid_window
```

## Evaluation

```bash
python render.py -m output/scene_atw -s <path>/mipnerf360/<scene> --eval --lpips --filter3d -r 4 --hybrid_window
```

## Repository Layout

```
arguments/           CLI and configuration dataclasses
gaussian_renderer/   differentiable rasteriser wrapper
scene/               COLMAP loader, dataset readers, Gaussian model
utils/               loss, SH, camera, and image utilities
submodules/          CUDA extensions (gaussian-blending, simple-knn)
train.py             training entry point
render.py            rendering and PSNR / SSIM / LPIPS evaluation
requirements.txt     Python dependencies
```

## Citation

```bibtex
@inproceedings{atwgs2026,
  title     = {ATW-GS: Anisotropic Transmittance Windows for Anti-Aliased 3D Gaussian Splatting},
  author    = {Goshu, Hana L. and Wakjira, Tadesse G. and Lam, Kin-Man},
  year      = {2026},
  note      = {Under Review}
}
```

## License

Released under the [MIT License](LICENSE).
