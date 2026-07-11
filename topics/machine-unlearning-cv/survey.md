# Machine Unlearning in Computer Vision: Methods, Benchmarks, and Robustness

> **Course Survey Draft** — Computer Vision Final Lab  
> **Topic Path:** `topics/machine-unlearning-cv/`  
> **Status:** Markdown draft (to be converted to CVPR LaTeX later)  
> **Last Updated:** July 2026

---

## Abstract

Machine unlearning (MU) enables the selective removal of data points, classes, or semantic concepts from trained vision models without full retraining. As computer vision systems are deployed at scale—from face recognition and medical imaging to text-to-image generation and vision-language assistants—the ability to *forget* specific visual knowledge has become a core requirement for privacy compliance, copyright protection, and safety alignment. This survey provides a structured overview of machine unlearning across the full spectrum of vision models: discriminative classifiers (CNNs, Vision Transformers), text-to-image diffusion models, and vision-language models (VLMs). We organize existing work along four axes—unlearning targets, implementation strategies, architectural scope, and evaluation protocols—and highlight a fundamental divide between discriminative forgetting (parameter-space erasure with statistical guarantees) and generative concept erasure (semantic-space suppression in entangled latent representations). We further review emerging benchmarks (HUB, EMMA, Genµ, ViT-MU) and robustness studies revealing that many methods achieve superficial forgetting while remaining vulnerable to adversarial revival. We conclude with open challenges toward verifiable, legally compliant, and robust vision unlearning.

**Keywords:** Machine Unlearning, Computer Vision, Concept Erasure, Diffusion Models, Vision-Language Models, Privacy, Robustness

---

## 1. Introduction

### 1.1 Why Machine Unlearning Is a Computer Vision Problem

Computer vision is not merely an application domain for machine unlearning—it is one of its most technically demanding and socially consequential frontiers. Unlike text-only language models, vision systems operate on high-dimensional perceptual signals where information is distributed across spatial features, cross-modal alignments, and generative latent spaces. The "Right to be Forgotten" under regulations such as GDPR and CCPA requires that when a user requests deletion of their facial images, medical scans, or copyrighted artwork from a training corpus, the deployed model must no longer encode that individual's identity or visual style. Similarly, safety-critical deployments require removing Not-Safe-for-Work (NSFW) content, celebrity likenesses, and proprietary characters from generative models.

These requirements map directly onto core computer vision tasks:

| CV Task | Unlearning Scenario |
|---------|---------------------|
| Image classification | Remove a class or subset of training images |
| Face recognition / Re-ID | Erase a specific identity while retaining others |
| Text-to-image generation | Suppress a visual concept (style, celebrity, IP character) |
| Vision-language models | Decouple image-text associations for sensitive pairs |
| Video generation | Erase temporal concept persistence across frames |

The CVPR 2026 Workshop on Machine Unlearning for Vision (MUV) and ICCV 2025 benchmarks such as HUB demonstrate that the vision community now treats unlearning as a first-class research problem alongside detection, segmentation, and generation.

### 1.2 Problem Formulation

Given a model $f_\theta$ trained on dataset $\mathcal{D}$, a *forget set* $\mathcal{D}_f \subset \mathcal{D}$, and a *retain set* $\mathcal{D}_r = \mathcal{D} \setminus \mathcal{D}_f$, the goal of machine unlearning is to produce updated parameters $\theta'$ such that:

1. **Forgetting efficacy:** $f_{\theta'}$ behaves as if it were never trained on $\mathcal{D}_f$.
2. **Utility retention:** Performance on $\mathcal{D}_r$ is preserved.
3. **Efficiency:** Updating $\theta \to \theta'$ is substantially cheaper than retraining from scratch.

Two formulations dominate the literature:

- **Exact unlearning** requires the distribution of $f_{\theta'}$ to be statistically indistinguishable from a model retrained on $\mathcal{D}_r$ alone (typically measured via KL divergence = 0). Methods like SAFE use shard graphs to make exact unlearning tractable for smaller models.
- **Approximate unlearning** relaxes the indistinguishability guarantee in exchange for computational efficiency, accepting a bounded risk of residual information leakage.

### 1.3 Scope and Organization

This survey covers vision-specific unlearning from 2020 to 2026, drawing primarily from CVPR, ICCV, NeurIPS, and arXiv preprints. We exclude general LLM unlearning unless directly adapted to multimodal vision settings. The paper is organized as follows:

- **Section 2:** Taxonomy and foundational models
- **Section 3:** Discriminative vision unlearning (CNNs, ViTs)
- **Section 4:** Concept erasure in text-to-image diffusion models
- **Section 5:** Vision-language model unlearning
- **Section 6:** Robustness, attacks, and defenses
- **Section 7:** Benchmarks and evaluation metrics
- **Section 8:** Open challenges and future directions
- **Section 9:** Conclusion

---

## 2. Taxonomy and Foundational Models

### 2.1 Four-Axis Taxonomy

We organize machine unlearning for vision along four complementary dimensions, following Safavigerdini et al. (CVPRW 2026):

```
Machine Unlearning for Vision
├── Targets
│   ├── Sample-level (individual images)
│   ├── Class-level (entire categories)
│   └── Concept-level (semantic attributes: style, identity, NSFW)
├── Implementation
│   ├── Data reorganization (sharding, partitioning)
│   ├── Model manipulation (gradient, weight editing, neuron suppression)
│   └── Inference-time intervention (activation steering, null-space projection)
├── Architecture
│   ├── Discriminative (CNN, ViT, Swin)
│   ├── Generative (DDPM, Flow Matching, Stable Diffusion)
│   └── Multimodal (CLIP, LLaVA, GPT-4V-style VLMs)
└── Evaluation
    ├── Forgetting efficacy
    ├── Utility retention
    ├── Privacy guarantees (MIA, LiRA)
    └── Computational efficiency
```

### 2.2 Architectural Backbones

**CNNs vs. Vision Transformers.** ViTs generally exhibit higher utility retention and lower catastrophic forgetting during unlearning due to global contextual modeling, while CNNs pair more effectively with gradient-based methods. Specialized techniques such as LetheViT apply attention-guided contrastive unlearning for ViTs, and parameter-efficient fine-tuning (LoRA, batch normalization tuning) enables scalable unlearning across both architectures.

**Diffusion Models.** Text-to-image models (Stable Diffusion, SDXL) encode concepts primarily in cross-attention layers that map text embeddings to visual features. This architectural insight drives most concept erasure methods to target projection matrices $W_k, W_v$ in cross-attention modules rather than the full U-Net.

**Vision-Language Models.** VLMs decompose into vision encoder $H_{\theta_v}$, text encoder $E_{\theta_t}$, and fusion decoder $G_{\theta_d}$. Unlearning must address cross-modal entanglement: removing a concept from the text encoder may leave residual visual associations recoverable through alternative prompts.

---

## 3. Discriminative Vision Unlearning

Discriminative unlearning removes individual data points or entire classes from classifiers while preserving performance on retained data.

### 3.1 Exact vs. Approximate Methods

| Category | Representative Methods | Key Idea |
|----------|----------------------|----------|
| Exact | SAFE (Shard Graphs) | Bilevel sharding enables tractable exact unlearning |
| Approximate (gradient) | NegGrad, GLI, PGU, LTU | Gradient ascent on forget set with retention regularization |
| Approximate (boundary) | Boundary Unlearning (BU) | Manipulate decision boundaries instead of parameters |
| Approximate (NTK) | Fast-NTK | Neural Tangent Kernel theory for efficient updates |

**Projected-Gradient Unlearning (PGU)** identifies a Core Gradient Space for the retain set and applies orthogonal gradient updates, minimizing interference between forgetting and retention objectives. **Guided Loss-Increasing (GLI)** combines feature distancing with classification guidance to keep updates within the semantic manifold, avoiding the catastrophic utility collapse of naive gradient ascent.

### 3.2 Knowledge and Representation-Level Unlearning

Rather than merely altering output logits, representation-level methods target the internal feature geometry:

- **ERM-KTP** introduces Entanglement-Reduced Masks enforcing sparsity and orthogonality on convolutional filters, decoupling class-specific representations.
- **POUR** leverages Neural Collapse theory, projecting forgotten class features to the origin within the Equiangular Tight Frame structure while preserving optimal geometry for retained classes.
- **L-CODEC** selects a sparse Markov Blanket of influential parameters via conditional independence coefficients.
- **Scissorhands** performs targeted "lobotomy" by reinitializing the most forget-set-sensitive parameters.

The **Representation Unlearning Score (RUS)** uses Centered Kernel Alignment (CKA) to quantify how closely unlearned model representations match those of a gold-standard retrained model.

### 3.3 Specialized Vision Domains

- **Person Re-Identification:** De-ReID uses identity shift mechanisms to move unlearned features into non-retrievable spaces.
- **Facial Attributes:** MUFAC and MUCAC benchmarks require forgetting specific identities while preserving age/gender estimation capability.
- **Reinforcement Learning:** PolicyCleanse neutralizes backdoors by re-optimizing the Bellman equation.

---

## 4. Concept Erasure in Text-to-Image Diffusion Models

Concept erasure suppresses targeted semantic content (celebrities, artistic styles, NSFW categories, copyrighted characters) in generative diffusion models.

### 4.1 Fine-Tuning Based Methods

| Method | Mechanism | Scalability |
|--------|-----------|-------------|
| ESD (Erased Stable Diffusion) | Fine-tune noise prediction toward neutral prompts | Single concept |
| Selective Amnesia (SA) | EWC + Fisher Information Matrix to protect non-target weights | Single concept |
| MACE | Closed-form cross-attention refinement + per-concept LoRA | Up to 100 concepts |
| DyME | Lightweight LoRA with bi-level orthogonality constraints | Multi-concept composition |

**Erased Stable Diffusion (ESD)** aligns the model's noise predictions for target concepts with those of neutral prompts, steering the denoising trajectory away from undesired content. **MACE (Mass Concept Erasure)** scales this to dozens of concepts through a multi-stage pipeline combining closed-form weight editing with individual LoRA adapters.

### 4.2 Closed-Form and Training-Free Methods

- **UCE (Unified Concept Editing):** Formulates cross-attention weight modification as a closed-form optimization problem, steering concept embeddings toward target outputs while preserving non-target behavior.
- **RECE:** Achieves erasure in ~3 seconds via iterative adversarial closed-form updates.
- **GLoCE:** Injects gated low-rank adaptation at inference time, projecting target embeddings onto a null subspace.
- **SPEED:** Null-space editing with Influence-based Prior Filtering; erases 100 concepts in 5 seconds.
- **DVE:** Training-free differential vector erasure applicable to flow matching models beyond DDPM.

These training-free approaches significantly reduce collateral damage to adjacent concepts and maintain higher image fidelity than full fine-tuning.

### 4.3 Neuron-Level and Latent Space Interventions

- **SNCE (Single Neuron Concept Erasure):** Uses Sparse Autoencoders to disentangle text encoder activations into interpretable neurons, suppressing concept-specific neurons via modulated frequency scoring.
- **Prototype-Guided Erasure:** Clusters latent embedding differences into concept prototypes, injecting them as negative conditioning in classifier-free guidance.
- **Concept Corrector:** Performs on-the-fly erasure by detecting target concepts in intermediate generated images and switching to Concept Removal Attention.

### 4.4 Evaluation Challenges for Concept Erasure

Narrow concepts (specific IP characters, individual celebrities) are more easily erased than broad categories (NSFW, violence) due to semantic diversity. Erasing narrow concepts can inadvertently degrade shared supertypes (e.g., general "person" generation quality). Methods must be evaluated under both standard text prompts and adversarial conditions (learned embeddings, inverted latents, paraphrased prompts).

---

## 5. Vision-Language Model Unlearning

VLM unlearning presents unique challenges due to cross-modal entanglement. Lin et al. (arXiv 2026) provide the first systematic robustness study, categorizing methods into four paradigms:

### 5.1 Method Paradigms

| Paradigm | Target Parameters | Representative Methods | Trade-off |
|----------|------------------|----------------------|-----------|
| Full-parameter finetuning | $\theta_v, \theta_t, \theta_d$ | GA, GD, NPO, SimNPO, RMU, FTTP | Strong expressiveness; high cost; risk of over-updating |
| Vision-encoder finetuning | $\theta_v$ only | HFRU, RAZOR, ADU, AUVIC | Effective for visual concepts; may miss text-side leakage |
| Selective parameter finetuning | Subset $\theta_s \subset \theta$ | Mmunlearner, SLUG | Balance of efficiency and precision; depends on selection accuracy |
| Inference-time intervention | None (frozen weights) | SAUCE, MANU, R-MUSE, MIP-Editor | Lightweight; may not guarantee persistent forgetting |

**MultiDelete** achieves cross-modal decoupling by minimizing a decoupling loss while preserving unimodal and multimodal knowledge retention. **Hyperbolic Alignment Calibration (HAC)** leverages hierarchical semantic structures in hyperbolic embedding spaces (MERU) for superior concept removal.

### 5.2 VLM-Specific Challenges

Unlike LLM unlearning, VLM unlearning must satisfy three coupled objectives:

1. **Forgetting:** Suppress memorized image-text associations in $\mathcal{D}_f$.
2. **Retention:** Preserve performance on $\mathcal{D}_r$.
3. **General capability:** Maintain overall vision-language reasoning without degrading cross-modal grounding.

A model may appear to forget under direct queries ("Describe this celebrity") yet retain recoverable associations under paraphrased, contextual, or discriminative prompts ("Is there a [concept] in this image?").

---

## 6. Robustness, Attacks, and Defenses

### 6.1 The "Hide vs. Erase" Problem

A central finding across recent studies is that many unlearning methods achieve *suppression* rather than *erasure*. Forgotten knowledge remains latent in model parameters and can be reactivated through:

| Attack Type | Mechanism | Example |
|-------------|-----------|---------|
| White-box | Gradient-based prompt reconstruction | P4D, Concept Inversion (CI) |
| Black-box | Alternative semantic descriptions | Ring-A-Bell, PEZ |
| Concept revival | Fine-tuning on unrelated data | Post-erasure fine-tuning instability |
| In-context (VLM) | Semantically related contextual cues prepended to prompts | Lin et al. 2026 |
| In-distribution (VLM) | Retraining on forget-distribution samples | Rapid knowledge recovery |
| Out-of-distribution (VLM) | Retraining on semantically related auxiliary data | Distributional transfer reactivation |

M-ErasureBench reveals Concept Reproduction Rates (CRR) exceeding 90% under white-box attacks for many state-of-the-art erasure methods, despite appearing effective under standard text prompts.

### 6.2 Defense Mechanisms

- **AdvUnlearn:** Bi-level optimization incorporating adversarial prompt generation during training, with utility-retaining regularization.
- **S-GRACE:** Semantics-guided adversarial erasure using LLM-constructed comprehensive concept prompt sets.
- **R.A.C.E.:** Adversarial training reducing white-box attack success by 30 percentage points.
- **IRECE:** Inference-time plug-and-play module localizing target concepts via cross-attention maps and perturbing associated latents.
- **AEGIS / FADE:** Adversarial erasure targets with trajectory preservation for retention-data-free robust erasure.

### 6.3 Toward Worst-Case Evaluation

Traditional metrics (forget-set accuracy, average-case MIA) are insufficient and easily gamed. Emerging rigorous metrics include:

- **Dimensional Alignment (DA):** Eigenspace alignment between forget and retain sets
- **Normalized Mutual Information (NMI):** Detects residual semantic leakage in latent space
- **RF-JSD:** Retrain-free Jensen-Shannon Divergence for large-scale distributional comparison
- **LiRA (Likelihood Ratio Attack):** Stronger privacy verification than rudimentary MIAs
- **Worst-case forget sets:** Bi-level optimization identifies subsets significantly harder to unlearn than random selections

---

## 7. Benchmarks and Evaluation Metrics

### 7.1 Discriminative Unlearning Benchmarks

| Benchmark | Focus | Key Metrics |
|-----------|-------|-------------|
| MUFAC / MUCAC | Facial attribute instance unlearning | NoMUS composite score |
| ViT-MU (Zhao et al. 2026) | Vision Transformer unlearning across capacities | Forget quality + retain accuracy + ToW |
| OpenGU | Graph-structured data (37 datasets) | Node/edge/feature-level unlearning |
| Worst-Case Forget Sets | Vulnerability analysis on ImageNet, CelebA | Hardness-stratified evaluation |

### 7.2 Concept Erasure Benchmarks

| Benchmark | Scale | Key Contribution |
|-----------|-------|-----------------|
| **HUB** (ICCV 2025) | 33 concepts, 16K prompts/concept | Six dimensions: faithfulness, alignment, pinpoint-ness, multilingual robustness, attack robustness, efficiency |
| **Genµ** | Stable Diffusion single/multi/continual | ERR (Erasing-Retention Robustness) score |
| **EMMA** | 206 concept categories | 12 metrics including implicit prompts and bias amplification |
| **M-ErasureBench** | Multimodal (text, embedding, latent) | Reveals robustness gaps beyond text prompts |
| **ErasureBench-H** | Hierarchical brand-series-character | Semantic granularity and scalability |

### 7.3 Composite Evaluation Framework

Effective evaluation balances four pillars:

```
┌─────────────────────────────────────────────────────┐
│                  Evaluation Pillars                    │
├──────────────┬──────────────┬────────────┬────────────┤
│  Forgetting  │   Utility    │  Privacy   │ Efficiency │
│  Efficacy    │  Retention   │ Guarantees │            │
├──────────────┼──────────────┼────────────┼────────────┤
│ Forget acc ↓ │ Retain acc ↑ │ MIA ↓      │ Time ↓     │
│ CRR ↓        │ FID ↑        │ LiRA ↓     │ Memory ↓   │
│ DA ↑         │ CLIP score ↑ │ NMI ↓      │ Params Δ ↓ │
└──────────────┴──────────────┴────────────┴────────────┘
```

No single method excels across all HUB evaluation dimensions (ICCV 2025), underscoring the need for multi-faceted assessment protocols.

---

## 8. Open Challenges and Future Directions

### 8.1 Fundamental Theoretical Gaps

1. **No formal guarantees for generative unlearning.** Unlike discriminative settings where Neural Collapse and NTK theory provide principled frameworks, generative concept erasure lacks statistical indistinguishability guarantees.
2. **Concept entanglement.** Visual concepts are not orthogonal in latent space; erasing "Mickey Mouse" may affect general "cartoon" or "mouse" generation.
3. **Multimodal decoupling.** Removing image-text associations in VLMs requires severing connections across three interacting components simultaneously.

### 8.2 Evaluation and Verification

1. **Membership Inference Attacks are inadequate** as sole verification tools; stronger adversarial threats (LiRA, worst-case forget sets) are needed.
2. **Absence of worst-case standards.** Average-case metrics can be manipulated by trivial final-layer fine-tuning.
3. **Retrain-free evaluation at scale.** Gold-standard retrained models are computationally infeasible for billion-parameter vision foundation models.

### 8.3 Practical Deployment

1. **Continual unlearning.** Real systems receive sequential deletion requests; catastrophic interference across multiple unlearning rounds remains unsolved.
2. **Legal compliance.** Mapping technical unlearning guarantees to GDPR "Right to be Forgotten" legal standards requires interdisciplinary frameworks.
3. **Cross-architecture generalization.** Methods developed for CNNs, ViTs, diffusion models, and VLMs do not transfer cleanly; unified frameworks are needed.

### 8.4 Promising Research Directions

- **Neural Collapse and hyperbolic geometry** for principled representation-level forgetting
- **Adversarially robust unlearning** integrated into training (not post-hoc defense)
- **Sparse autoencoder neuron surgery** for interpretable concept removal
- **Flow matching and video model erasure** extending beyond DDPM-based Stable Diffusion
- **Certified unlearning** with provable bounds on residual information leakage

---

## 9. Conclusion

Machine unlearning has evolved from a privacy-motivated niche into a central research area within computer vision. The field now spans discriminative classifiers, generative diffusion models, and multimodal vision-language systems, each presenting distinct technical challenges. Discriminative unlearning benefits from mature gradient-based and representation-level methods with emerging theoretical grounding in Neural Collapse geometry. Generative concept erasure has achieved remarkable practical efficiency through closed-form weight editing and training-free inference interventions, yet remains fundamentally vulnerable to adversarial concept revival. VLM unlearning introduces cross-modal entanglement as a new frontier, where robustness evaluations reveal that current methods often hide rather than erase target knowledge.

For the vision community, the path forward requires moving beyond average-case output metrics toward worst-case, feature-level, and adversarially-informed evaluation protocols. Standardized benchmarks such as HUB, EMMA, and ViT-MU provide essential infrastructure, but no single method yet achieves robust forgetting across all evaluation dimensions. Building verifiable, legally compliant, and adversarially resilient vision unlearning systems remains one of the most pressing open problems at the intersection of computer vision, privacy, and AI safety.

---

## References

> Full BibTeX entries are maintained in `references.md`. Key anchor papers:

1. Safavigerdini et al. "Machine Unlearning in Computer Vision: Survey of Methods from Discriminative to Generative Foundation Models." CVPRW 2026.
2. Moon et al. "Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning." ICCV 2025.
3. Lin et al. "On the Robustness of Machine Unlearning for Vision-Language Models." arXiv:2605.26992, 2026.
4. Zhao et al. "Benchmarking Unlearning for Vision Transformers." arXiv:2602.20114, 2026.
5. Gandikota et al. "Erased Stable Diffusion: Fine-tuning Text-to-Image Diffusion Models to Forget Concepts." 2023.
6. Gandikota et al. "Unified Concept Editing in Pre-trained Diffusion Models." WACV 2024 (UCE).
7. Zhang et al. "Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models." 2023.
8. Kurmanji et al. "Towards Unbounded Machine Unlearning." NeurIPS 2023 (SCRUB).
9. Jia et al. "HFRU: High-Fidelity Representation Unlearning." 2026.
10. Bourtoule et al. "Machine Unlearning." IEEE S&P 2021.

---

## Appendix A: Research Timeline (2020–2026)

```
2020-2021  Foundations
           └── MU formalized (S&P 2021), early exact methods (SISA, SAFE)

2022-2023  Discriminative + Early Generative
           └── ESD, UCE, NegGrad variants, facial attribute benchmarks (MUFAC)

2024       Scaling & Attacks
           └── MACE, AdvUnlearn, concept revival discovered, VLM unlearning begins

2025       Benchmarks & Robustness
           └── HUB (ICCV), EMMA, Genµ, VLM attack studies, Flow Matching erasure (DVE)

2026       Unified Frameworks
           └── CVPRW survey, ViT-MU benchmark, VLM robustness taxonomy, worst-case evaluation
```

## Appendix B: Suggested Paper Structure for CVPR LaTeX Conversion

```
survey.tex
├── \section{Introduction}
├── \section{Background and Taxonomy}
├── \section{Discriminative Vision Unlearning}
├── \section{Concept Erasure in Diffusion Models}
├── \section{Vision-Language Model Unlearning}
├── \section{Robustness and Evaluation}
├── \section{Benchmarks and Metrics}
├── \section{Discussion and Future Work}
├── \section{Conclusion}
└── \bibliography{references}
```
