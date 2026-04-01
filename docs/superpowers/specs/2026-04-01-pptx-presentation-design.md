# Design: PowerPoint Presentation with DIEF Template

**Date:** 2026-04-01
**Goal:** Generate bilingual (IT + EN) PowerPoint presentations for the Perceiver IO project using the official DIEF/UniFi template.

## Approach

Single Python script (`slides_V2/generate_ppt_v3.py`) using `python-pptx` that generates both language versions from the same codebase. Content is sourced from the existing V2 LaTeX Beamer slides.

## Template

- **File:** `slides/DIEF nuovo format presentazione Power point 2023.pptx`
- **Layout 0:** Blue full-background + UniFi logo (title slide)
- **Layout 1:** Blue full-background + UniFi logo (section separator)
- **Layout 2:** White + blue bottom bar + logo (standard content) - placeholder idx 0 = title, idx 13 = body
- **Layout 3/4:** White content variants
- **Layout 5:** Blue full-background (closing slide) - placeholders idx 11-16
- **Layout 6:** Blue section variant

## Slide Structure (39 slides)

### Section 1: Introduction (slides 1-5)
| # | Title | Layout | Content |
|---|-------|--------|---------|
| 1 | Perceiver & Perceiver IO | 0 (cover) | Title, subtitle, author, course |
| 2 | Agenda | 2 (white) | 4 numbered sections |
| 3 | The Problem: Modality-Specific Architectures | 2 | Text + complexity table |
| 4 | The Perceiver Solution | 2 | Key idea + complexity table + contributions |
| 5 | Perceiver IO: Structured Outputs | 2 | Encode/Process/Decode + flexibility |

### Section 2: Architecture (slides 6-15)
| # | Title | Layout | Content |
|---|-------|--------|---------|
| 6 | Architecture (separator) | 1 (blue) | Section title only |
| 7 | Perceiver Architecture Overview | 2 | Image `perceiver_arch.png` + bullets |
| 8 | Cross-Attention: The Key Mechanism | 2 | Formula + key insight |
| 9 | Latent Transformer & Weight Sharing | 2 | Parameter comparison shared/unshared |
| 10 | Fourier Positional Encoding | 2 | Formula + dimension table + finding |
| 11 | Perceiver IO Decoder | 2 | Image `perceiverio_arch.png` + formula |
| 12 | Output Query Design per Task | 2 | Task/queries table |
| 13 | Computational Complexity | 2 | Comparison table + concrete example |
| 14 | GELU & LAMB Optimizer | 2 | Formula + Adam vs LAMB comparison |
| 15 | Code Structure | 2 | Project tree + lines-of-code table |

### Section 3: Experiments & Results (slides 16-30)
| # | Title | Layout | Content |
|---|-------|--------|---------|
| 16 | Experiments & Results (separator) | 1 (blue) | Section title only |
| 17 | CIFAR-10: Setup | 2 | Hyperparameter table + ablation plan |
| 18 | CIFAR-10: Ablation Results | 2 | Image `cifar10_chart.png` + table |
| 19 | CIFAR-10: Impact of PE | 2 | 72% vs 35% comparison |
| 20 | CIFAR-10: Weight Sharing | 2 | Trade-off table |
| 21 | CIFAR-10: Permutation Robustness | 2 | Fourier vs Learned table |
| 22 | CIFAR-10: Perceiver vs Perceiver IO | 2 | 72% vs 78% comparison |
| 23 | ModelNet40: Setup & Results | 2 | Config + results + paper gap |
| 24 | ModelNet40: Augmentation Effects | 2 | Image `modelnet40_chart.png` |
| 25 | WikiText-103: MLM Pre-training | 2 | Byte-level setup + table |
| 26 | GLUE: Fine-tuning 8 Tasks | 2 | Pipeline + task list |
| 27 | GLUE: Per-Task Analysis | 2 | Image `glue_chart.png` + observations |
| 28 | Convergence: All Experiments | 2 | Image `convergence_chart.png` + table |
| 29 | Attention Maps Analysis | 2 | 2 attention map images side by side |
| 30 | Comprehensive Results Summary | 2 | Summary table |

### Section 4: Criticalities & Discussion (slides 31-36)
| # | Title | Layout | Content |
|---|-------|--------|---------|
| 31 | Criticalities & Discussion (separator) | 1 (blue) | Section title only |
| 32 | Gap vs Original Paper | 2 | ModelNet40 vs CIFAR-10 gap |
| 33 | Hardware Limitations | 2 | Google vs our setup table |
| 34 | Possible Improvements | 2 | Architecture + training + data |
| 35 | Lessons Learned | 2 | Pros/cons + main takeaway |
| 36 | Conclusions | 2 | Strengths/weaknesses + final assessment |

### Closing (slides 37-39)
| # | Title | Layout | Content |
|---|-------|--------|---------|
| 37 | References | 2 | 6 bibliographic references |
| 38 | Expected Questions | 2 | 8 Q&A pairs |
| 39 | Thank you! | 5 (closing blue) | Email, code info |

## Technical Details

### Script structure
- Single file: `slides_V2/generate_ppt_v3.py`
- Function `build(lang)` generates one language version
- Bilingual dictionary for all text content: `{"en": "...", "it": "..."}`
- Helper functions: `add_content_slide()`, `add_section_slide()`, `add_title_slide()`, `add_closing_slide()`

### Images (from `Perceiver_Paper/`)
- `figures/perceiver_arch.png` - Perceiver architecture diagram
- `figures/perceiverio_arch.png` - Perceiver IO architecture diagram
- `analysis_results/cifar10_chart.png` - CIFAR-10 ablation chart
- `analysis_results/modelnet40_chart.png` - ModelNet40 results chart
- `analysis_results/glue_chart.png` - GLUE per-task chart
- `analysis_results/convergence_chart.png` - Convergence comparison
- `attention_analysis/exp1_baseline_fourier_evolution.png` - Fourier PE attention
- `attention_analysis/exp3B_rgb_only_evolution.png` - RGB-only attention

### Output files
- `slides_V2/presentation_v3_eng.pptx`
- `slides_V2/presentation_v3_ita.pptx`

### Formulas
Rendered as simplified text (e.g., `O(N*M + M^2*L)`) since python-pptx cannot render LaTeX math natively.

### Placeholder mapping (from template analysis)
- **Layout 0 (title):** idx 0 = title, idx 10 = subtitle, idx 11 = author, idx 12 = course
- **Layout 2 (content):** idx 0 = title, idx 13 = body text
- **Layout 5 (closing):** idx 11 = main text, idx 12 = secondary, idx 13 = name, idx 15 = email, idx 16 = extra
