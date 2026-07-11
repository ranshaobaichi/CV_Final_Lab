# Core References — Machine Unlearning in Computer Vision

> Curated bibliography for the survey draft. Prioritize top-venue and recent arXiv papers.

---

## Survey & Benchmark Papers (Start Here)

| ID | Paper | Venue | URL |
|----|-------|-------|-----|
| S1 | Machine Unlearning in Computer Vision: Survey of Methods from Discriminative to Generative Foundation Models | CVPRW 2026 | [PDF](https://openaccess.thecvf.com/content/CVPR2026W/MUV/papers/Safavigerdini_Machine_Unlearning_in_Computer_Vision_Survey_of_Methods_from_Discriminative_CVPRW_2026_paper.pdf) |
| S2 | Holistic Unlearning Benchmark (HUB) | ICCV 2025 | [Page](https://openaccess.thecvf.com/content/ICCV2025/html/Moon_Holistic_Unlearning_Benchmark_A_Multi-Faceted_Evaluation_for_Text-to-Image_Diffusion_Model_ICCV_2025_paper.html) |
| S3 | On the Robustness of Machine Unlearning for Vision-Language Models | arXiv 2026 | [HTML](https://arxiv.org/html/2605.26992v1) |
| S4 | Benchmarking Unlearning for Vision Transformers | arXiv 2026 | [HTML](https://arxiv.org/html/2602.20114v1) |

---

## Foundational

```bibtex
@inproceedings{bourtoule2021machine,
  title={Machine Unlearning},
  author={Bourtoule, Lucas and others},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  year={2021}
}

@article{nguyen2022survey,
  title={A Survey of Machine Unlearning},
  author={Nguyen, Quoc Toan and others},
  journal={arXiv preprint arXiv:2209.02299},
  year={2022}
}
```

---

## Discriminative Vision Unlearning

| Method | Paper | Year |
|--------|-------|------|
| SAFE | Bourtoule et al., shard-based exact unlearning | 2021 |
| SCRUB | Kurmanji et al., NeurIPS 2023 | 2023 |
| GLI | Guided Loss-Increasing Unlearning | 2024 |
| PGU | Projected-Gradient Unlearning | 2024 |
| POUR | Provably Optimal Unlearning via Neural Collapse | 2025 |
| LetheViT | Attention-guided contrastive unlearning for ViT | 2025 |
| ERM-KTP | Entanglement-Reduced Mask knowledge-level unlearning | 2024 |
| BU | Boundary Unlearning | 2023 |
| BAMU | Bias-Aware Machine Unlearning | 2024 |

---

## Concept Erasure (Text-to-Image Diffusion)

| Method | Paper | Year |
|--------|-------|------|
| ESD | Erased Stable Diffusion (Gandikota et al.) | 2023 |
| UCE | Unified Concept Editing (WACV 2024) | 2024 |
| MACE | Mass Concept Erasure in Diffusion Models | 2024 |
| RECE | Rapid Erasure with Closed-form Editing | 2024 |
| SNCE | Single Neuron Concept Erasure | 2024 |
| GLoCE | Gated LoRA for Concept Erasure | 2024 |
| SPEED | Null-space editing, 100 concepts in 5s | 2024 |
| DyME | Dynamic Multi-concept Erasure | 2024 |
| DVE | Differential Vector Erasure for Flow Matching | 2025 |
| AdvUnlearn | Adversarial training for robust erasure | 2024 |
| S-GRACE | Semantics-guided adversarial erasure | 2024 |

---

## Vision-Language Model Unlearning

| Method | Paper | Year |
|--------|-------|------|
| MultiDelete | Cross-modal decoupling for CLIP-style VLMs | 2024 |
| HAC | Hyperbolic Alignment Calibration | 2024 |
| HFRU | High-Fidelity Representation Unlearning | 2026 |
| RAZOR | Vision-encoder selective unlearning | 2026 |
| Mmunlearner | Selective parameter finetuning | 2025 |
| SAUCE | Inference-time intervention | 2025 |
| GA / GD / NPO | Adapted from LLM unlearning | 2022–2024 |

---

## Benchmarks & Evaluation

| Benchmark | Paper | Venue |
|-----------|-------|-------|
| MUFAC / MUCAC | Facial attribute unlearning | 2024 |
| Genµ | Generative unlearning benchmark (ERR score) | 2024 |
| EMMA | 206-category concept erasure benchmark | 2025 |
| M-ErasureBench | Multimodal erasure evaluation | 2024 |
| HUB | Holistic 6-dimension evaluation | ICCV 2025 |
| ViT-MU | Vision Transformer unlearning benchmark | arXiv 2026 |
| ErasureBench-H | Hierarchical concept erasure | 2024 |

---

## Reading Order (Recommended)

1. **S1** (CVPRW 2026 survey) — overall map and taxonomy
2. **S2** (HUB, ICCV 2025) — evaluation framework for diffusion erasure
3. Gandikota et al. (ESD, 2023) — foundational concept erasure method
4. Gandikota et al. (UCE, WACV 2024) — closed-form editing
5. **S3** (VLM robustness, 2026) — multimodal unlearning + attacks
6. **S4** (ViT benchmark, 2026) — discriminative ViT unlearning
7. Bourtoule et al. (S&P 2021) — foundational MU formulation

---

## Search Keywords for Further Literature

```
"machine unlearning" + ("computer vision" OR "image classification" OR "diffusion")
"concept erasure" + ("stable diffusion" OR "text-to-image")
"vision-language" + "unlearning"
"forget set" + ("ViT" OR "vision transformer")
membership inference attack + unlearning
Neural Collapse + unlearning
```
