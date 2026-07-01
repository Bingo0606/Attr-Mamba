# Attr-Mamba

Official implementation of **Attr-Mamba: Attribute-Guided State Space Model for Progressive Medical Referring Image Segmentation**.

Attr-Mamba is designed for Medical Referring Image Segmentation (Medical RIS), where the model segments a lesion or target region according to a clinical-style natural language description. The framework separates anatomical/spatial cues for localization from morphology-related cues for boundary refinement through a progressive state-space decoder.

## Highlights

- Progressive medical referring image segmentation with anatomy-guided localization and morphology-aware boundary refinement.
- Clinical-style referring descriptions for Ref-LITS and Ref-LIDC in `datasets/`.
- Transparent evaluation with a fixed threshold and no connected-component post-processing.

## Requirements

Python 3.10 is recommended. Building the selective-scan extension requires a CUDA toolkit compatible with the installed PyTorch version.

```bash
conda create -n attr-mamba python=3.10
conda activate attr-mamba
pip install -r requirements.txt
cd selective_scan
pip install .
cd ..
```

The code expects the following external pretrained backbones when training from scratch:

- RadBERT: pass with `--bert_path`, or place it at `./checkpoint/RadBERT`
- Swin-T: pass with `--swin-pretrained`, or place it at `./checkpoint/swin_T/swin_tiny_patch4_window7_224.pth`

## Repository Structure

```text
Attr-Mamba/
|-- main.py                 # Training and evaluation entry point
|-- engine.py               # Optimization and metric computation
|-- model/                  # Attr-Mamba architecture and visual encoder
|-- vmamba_model/           # State-space building blocks
|-- selective_scan/         # CUDA selective-scan extension
|-- ref_dataset/            # Medical referring segmentation loader
|-- datasets/               # Clinical-style text descriptions
`-- assets/figures/         # Method and result figures used in this README
```

## Quick Start

Training on Ref-LITS / Ref-LIDC style data:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --distributed \
  --model AttrMamba \
  --data-set ref-lits \
  --data-path ./datasets/Ref-LITS \
  --json-prefix ref_lits \
  --image-size 512 \
  --output_dir ./outputs/ref_lits \
  --batch_size 4 \
  --epochs 200 \
  --lr 1e-4 \
  --weight-decay 0.05
```

Evaluation:

```bash
python main.py \
  --eval \
  --model AttrMamba \
  --data-set ref-lits \
  --data-path ./datasets/Ref-LITS \
  --json-prefix ref_lits \
  --test-split test \
  --resume ./outputs/ref_lits/best_checkpoint.pth
```

Evaluation uses raw sigmoid outputs thresholded at `0.5`. No connected-component selection or hidden post-processing is applied. Metrics include mIoU, mDice, oIoU, and HD95.
The public training loader uses deterministic resizing and normalization only; random data augmentation is not enabled in this release.

Evaluation does not save sample-level logs or visualizations by default. To export qualitative results, pass an explicit visualization directory:

```bash
python main.py \
  --eval \
  --model AttrMamba \
  --data-set ref-lits \
  --data-path ./datasets/Ref-LITS \
  --json-prefix ref_lits \
  --test-split test \
  --resume ./outputs/ref_lits/best_checkpoint.pth \
  --vis-dir ./vis_results/ref_lits
```

Training writes `checkpoint.pth`, `best_checkpoint.pth`, and `log.txt` to `--output_dir`. Set `--seed` to reproduce a run with the same software and hardware configuration.

## Public Text Descriptions

Cleaned referring descriptions are provided for inspection of the clinical-style language component:

| File | Descriptions | Note |
| --- | ---: | --- |
| `datasets/Ref-LITS_descriptions.json` | 14,883 | Liver lesion referring descriptions |
| `datasets/Ref-LIDC_descriptions.json` | 8,721 | Pulmonary nodule referring descriptions |

Example entries follow this format:

```json
{
  "file_id": 1,
  "sentence": "A tiny, round hypodensity with heterogeneous texture and well-circumscribed margins is located in the superior hepatic region."
}
```

## Method Overview

Attr-Mamba follows a cascaded encoder-decoder design:

- **Text encoder**: frozen RadBERT extracts sentence-level and token-level language features.
- **Visual encoder**: Swin Transformer extracts high-level image representations.
- **SCM**: sentence-level anatomical priors calibrate visual states for stable localization.
- **BDM**: token-level morphology cues interact with local visual windows through Mamba/SSM-style state-space modeling for boundary refinement.
- **Objective**: `Ldice + Lfocal + 0.1 * Lbound`, with boundary weight map `W = 1 + 5B`.

<p align="center">
  <img src="assets/figures/conceptual-comparison.png" width="86%" alt="Conceptual comparison">
</p>

<p align="center"><b>Conceptual comparison.</b> Attr-Mamba bridges anatomy-guided localization and morphology-aware boundary refinement for attribute-guided coarse-to-fine Medical RIS.</p>

## Framework

<p align="center">
  <img src="assets/figures/overall-framework.png" width="100%" alt="Overall framework of Attr-Mamba">
</p>

<p align="center"><b>Overall framework.</b> Given a medical image and a referring text, Attr-Mamba extracts visual states, a sentence-level anatomical prior, and token-level textual states. Cascaded decoding performs SCM-based semantic calibration and BDM-based boundary refinement before multi-stage mask prediction.</p>

<p align="center">
  <img src="assets/figures/scm-bdm-modules.png" width="95%" alt="SCM and BDM modules">
</p>

<p align="center"><b>Core mechanisms.</b> SCM uses sentence-level anatomical priors for AdaLN-style visual calibration and gated SS2D residual injection. BDM uses token-level morphology cues and local visual windows for state-space interaction and text-state updating.</p>

## Experimental Results

The following table summarizes selected benchmark results. Metrics are reported as mIoU, mDice, and HD95.

| Dataset | mIoU | mDice | HD95 | Note |
| --- | ---: | ---: | ---: | --- |
| Ref-LITS | 69.62 | 78.41 | 10.20 | +6.08 mIoU over the second-best method |
| Ref-LIDC | 63.44 | 76.07 | 7.04 | +1.96 mIoU over the second-best method |
| QaTa-COV19 | 85.77 | 92.15 | 11.63 | Best mIoU and mDice among compared methods |
| MosMedData+ | 66.01 | 79.61 | 13.19 | Best mIoU and HD95 among compared methods |

<p align="center">
  <img src="assets/figures/qualitative-comparison.jpg" width="100%" alt="Qualitative comparison">
</p>

<p align="center"><b>Qualitative comparison.</b> Yellow, red, and green denote true positives, false negatives, and false positives, respectively.</p>

### Ablation Study

| SCM | BDM | Ref-LITS mIoU | Ref-LIDC mIoU | QaTa-COV19 mIoU | MosMedData+ mIoU |
| --- | --- | ---: | ---: | ---: | ---: |
| - | - | 57.11 | 53.96 | 74.06 | 58.23 |
| Yes | - | 59.73 | 55.93 | 76.60 | 60.01 |
| - | Yes | 62.94 | 57.07 | 78.13 | 61.36 |
| Yes | Yes | 69.62 | 63.44 | 85.77 | 66.01 |

| Progressive stages | Ref-LITS mIoU | Ref-LITS mDice | Ref-LITS HD95 | Params | GFLOPs | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65.51 | 74.72 | 13.89 | 184.16M | 211.70 | 22.15 |
| 2 | 66.72 | 76.03 | 12.94 | 185.85M | 212.21 | 20.91 |
| 3 | 68.03 | 77.46 | 11.72 | 192.12M | 213.89 | 18.54 |
| 4 | 69.62 | 78.41 | 10.20 | 216.20M | 219.89 | 16.85 |

<p align="center">
  <img src="assets/figures/stage-wise-heatmaps.jpg" width="100%" alt="Stage-wise heatmap visualization">
</p>

<p align="center"><b>Stage-wise heatmaps.</b> Spatial responses evolve from broad candidate regions to concentrated activations around referred lesions.</p>

### Efficiency

<p align="center">
  <img src="assets/figures/efficiency-tradeoff-qata-cov19.png" width="62%" alt="Efficiency comparison on QaTa-COV19">
</p>

On QaTa-COV19 at 224 x 224, Attr-Mamba uses 32.97 GFLOPs and reaches 26.88 FPS. Compared with LAVT, DMMI, LViT, and RefSegformer, it reduces GFLOPs by 60.7%, 47.9%, 39.1%, and 68.2%, respectively.

### Text Perturbation and Robustness

<p align="center">
  <img src="assets/figures/prompt-perturbation-visualization.jpg" width="100%" alt="Prompt perturbation visualization">
</p>

<p align="center"><b>Controlled prompt perturbations.</b> Spatial perturbations mainly affect localization, while morphology perturbations mainly affect boundary quality.</p>

| Dataset | Text setting | mIoU | mDice | Center distance | Boundary F-score | HD95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ref-LITS | Original | 76.96 | 86.31 | 2.15 | 78.00 | 6.19 |
| Ref-LITS | Spatial perturbation | 45.78 | 50.90 | 80.34 | 42.00 | 86.54 |
| Ref-LITS | Morphology perturbation | 70.35 | 79.26 | 12.91 | 68.00 | 19.14 |
| Ref-LIDC | Original | 67.59 | 79.26 | 4.18 | 84.00 | 6.56 |
| Ref-LIDC | Spatial perturbation | 31.74 | 45.31 | 92.21 | 49.00 | 92.41 |
| Ref-LIDC | Morphology perturbation | 62.69 | 74.69 | 6.71 | 76.00 | 9.23 |

| Dataset | Text input | mIoU | mDice | HD95 |
| --- | --- | ---: | ---: | ---: |
| Ref-LITS | Original text | 75.15 | 85.37 | 4.91 |
| Ref-LITS | Clinical-style text | 72.37 | 82.46 | 8.29 |
| Ref-LIDC | Original text | 60.89 | 74.78 | 2.89 |
| Ref-LIDC | Clinical-style text | 60.35 | 74.20 | 4.36 |

## Reproducibility Notes

- Dataset splits are read exactly from the specified JSON files; the loader does not create or modify a split.
- Images are resized with bilinear interpolation, while masks use nearest-neighbor interpolation.
- Random data augmentation, small-target filtering, and connected-component target selection are not used.
- Evaluation thresholds raw sigmoid outputs at `0.5` and reports mIoU, mDice, oIoU, and HD95.

## Acknowledgements

This implementation builds on open-source components from Swin Transformer, VMamba/Mamba-style selective scanning, RadBERT, and prior referring image segmentation codebases. We thank the authors and maintainers of these projects.
