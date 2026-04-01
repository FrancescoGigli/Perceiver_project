# DIEF PowerPoint Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate bilingual (IT + EN) PowerPoint presentations for the Perceiver IO project using the official DIEF/UniFi template.

**Architecture:** Single Python script with a bilingual content dictionary and helper functions for each DIEF layout type. The script loads the template, iterates through slide definitions, and produces two `.pptx` files.

**Tech Stack:** Python, python-pptx, Pillow (optional for image verification)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `slides_V2/generate_ppt_v3.py` | Main script: content definitions + generation logic |
| `slides_V2/presentation_v3_eng.pptx` | Output: English presentation |
| `slides_V2/presentation_v3_ita.pptx` | Output: Italian presentation |

**Template (read-only):** `slides/DIEF nuovo format presentazione Power point 2023.pptx`

**Images (read-only, from `Perceiver_Paper/`):**
- `figures/perceiver_arch.png`
- `figures/perceiverio_arch.png`
- `analysis_results/cifar10_chart.png`
- `analysis_results/modelnet40_chart.png`
- `analysis_results/glue_chart.png`
- `analysis_results/convergence_chart.png`
- `attention_analysis/exp1_baseline_fourier_evolution.png`
- `attention_analysis/exp3B_rgb_only_evolution.png`

### DIEF Template Layout Reference

```
Layout 0 "Copertina":       idx 0=title, 10=subtitle, 11=author, 12=role
Layout 1 "Copertina_HR":    idx 0=title, 10=subtitle, 11=author, 12=role (same as 0, HR variant)
Layout 2 "Interna_solo testo": idx 0=title, 13=body, 14=footer
Layout 3 "Interna_solo testo_HR": same as 2
Layout 4 "Interna_immagine+testo_HR": idx 10=picture, 11=caption, 12=text, 13=footer
Layout 5 "Controcopertina": idx 0=title, 11=main, 12=sub, 13=name, 14=role, 15=contact, 16=extra
Layout 6 "Controcopertina_HR": same as 5

Slide dimensions: 13.33in x 7.50in (widescreen 16:9)
```

---

### Task 1: Script skeleton with helper functions

**Files:**
- Create: `slides_V2/generate_ppt_v3.py`

- [ ] **Step 1: Create the script with imports, constants, and helper functions**

```python
"""Generate bilingual DIEF PowerPoint presentations for Perceiver IO project."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TEMPLATE = os.path.join(PROJECT_DIR, "slides", "DIEF nuovo format presentazione Power point 2023.pptx")
PAPER_DIR = os.path.join(PROJECT_DIR, "Perceiver_Paper")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
RESULTS_DIR = os.path.join(PAPER_DIR, "analysis_results")
ATTN_DIR = os.path.join(PAPER_DIR, "attention_analysis")


def _set_placeholder(slide, idx, text):
    """Set text of a placeholder by its idx, if it exists."""
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == idx:
            shape.text = text
            return shape
    return None


def _set_placeholder_formatted(slide, idx, lines, font_size=Pt(16), bold_first=False):
    """Set placeholder text with multiple lines formatted as bullet points.

    Each line can be:
    - str: plain paragraph
    - ("bold_prefix", "rest"): first part bold, second plain
    - Lines starting with '- ' become level-0 bullets
    - Lines starting with '  - ' become level-1 bullets
    """
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == idx:
            tf = shape.text_frame
            tf.clear()
            for i, line in enumerate(lines):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()

                if isinstance(line, tuple):
                    bold_part, rest = line
                    run_b = p.add_run()
                    run_b.text = bold_part
                    run_b.font.bold = True
                    run_b.font.size = font_size
                    run_r = p.add_run()
                    run_r.text = rest
                    run_r.font.size = font_size
                    p.level = 0
                elif isinstance(line, str):
                    text = line
                    level = 0
                    if text.startswith("  - "):
                        text = text[4:]
                        level = 1
                    elif text.startswith("- "):
                        text = text[2:]
                        level = 0

                    run = p.add_run()
                    run.text = text
                    run.font.size = font_size
                    if bold_first and i == 0:
                        run.font.bold = True
                    p.level = level
                elif line == "":
                    p.space_after = Pt(6)
            return shape
    return None


def add_title_slide(prs, title, subtitle, author, role):
    """Layout 0: blue full-background cover slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    _set_placeholder(slide, 0, title)
    _set_placeholder(slide, 10, subtitle)
    _set_placeholder(slide, 11, author)
    _set_placeholder(slide, 12, role)
    return slide


def add_section_slide(prs, title):
    """Layout 1: blue section separator."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _set_placeholder(slide, 0, title)
    # Clear other placeholders
    _set_placeholder(slide, 10, "")
    _set_placeholder(slide, 11, "")
    _set_placeholder(slide, 12, "")
    return slide


def add_content_slide(prs, title, lines, font_size=Pt(16)):
    """Layout 2: white content slide with title + body text."""
    slide = prs.slides.add_slide(prs.slide_layouts[2])
    _set_placeholder(slide, 0, title)
    _set_placeholder_formatted(slide, 13, lines, font_size=font_size)
    return slide


def add_image_text_slide(prs, title, img_path, text_lines, caption="", font_size=Pt(14)):
    """Layout 4: image on the left + text on the right."""
    slide = prs.slides.add_slide(prs.slide_layouts[4])
    # Title is not a named placeholder in layout 4, we use the footer area or add a textbox
    # Layout 4 has: idx 10=picture, 11=caption, 12=text, 13=footer
    if os.path.exists(img_path):
        placeholder = None
        for shape in slide.placeholders:
            if shape.placeholder_format.idx == 10:
                placeholder = shape
                break
        if placeholder:
            placeholder.insert_picture(img_path)
    _set_placeholder(slide, 11, caption)
    _set_placeholder_formatted(slide, 12, text_lines, font_size=font_size)
    return slide


def add_content_slide_with_image(prs, title, lines, img_path, font_size=Pt(14)):
    """Layout 2 with a manually placed image on the right side."""
    slide = prs.slides.add_slide(prs.slide_layouts[2])
    _set_placeholder(slide, 0, title)

    # Narrow the body text to the left half
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 13:
            shape.left = Inches(0.85)
            shape.width = Inches(5.5)
            break
    _set_placeholder_formatted(slide, 13, lines, font_size=font_size)

    # Place image on the right
    if os.path.exists(img_path):
        left = Inches(6.8)
        top = Inches(2.0)
        width = Inches(5.8)
        slide.shapes.add_picture(img_path, left, top, width=width)
    return slide


def add_closing_slide(prs, main_text, sub_text, name, role, contact, extra):
    """Layout 5: blue closing slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _set_placeholder(slide, 11, main_text)
    _set_placeholder(slide, 12, sub_text)
    _set_placeholder(slide, 13, name)
    _set_placeholder(slide, 14, role)
    _set_placeholder(slide, 15, contact)
    _set_placeholder(slide, 16, extra)
    return slide
```

- [ ] **Step 2: Verify the script is syntactically valid**

Run: `cd N:/Perceiver_project/slides_V2 && python -c "import generate_ppt_v3; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add slides_V2/generate_ppt_v3.py
git commit -m "feat: add PPTX generation script skeleton with DIEF layout helpers"
```

---

### Task 2: Bilingual content — Section 1 (Introduction, slides 1-5)

**Files:**
- Modify: `slides_V2/generate_ppt_v3.py`

- [ ] **Step 1: Add the `build_section1` function with bilingual content**

Append to `generate_ppt_v3.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# SLIDE CONTENT — bilingual
# ═══════════════════════════════════════════════════════════════════════════

def T(lang, en, it):
    """Return the text for the given language."""
    return en if lang == "en" else it


def build_section1(prs, lang):
    """Slides 1-5: Introduction & Motivation."""

    # Slide 1: Title
    add_title_slide(
        prs,
        title="Perceiver & Perceiver IO",
        subtitle=T(lang,
            "Implementation and Experimental Analysis",
            "Implementazione e Analisi Sperimentale"),
        author="Francesco Gigli",
        role=T(lang, "B031278 - Deep Learning", "B031278 - Deep Learning"),
    )

    # Slide 2: Agenda
    add_content_slide(prs,
        T(lang, "Agenda", "Agenda"),
        [
            T(lang, "1. Introduction & Motivation", "1. Introduzione e Motivazione"),
            T(lang, "2. Architecture", "2. Architettura"),
            T(lang, "3. Experiments & Results", "3. Esperimenti e Risultati"),
            T(lang, "4. Criticalities & Discussion", "4. Criticit\u00e0 e Discussione"),
        ],
        font_size=Pt(20),
    )

    # Slide 3: The Problem
    add_content_slide(prs,
        T(lang,
            "The Problem: Modality-Specific Architectures",
            "Il Problema: Architetture Specifiche per Modalit\u00e0"),
        [
            T(lang, "Traditional State-of-the-Art:", "Stato dell'arte tradizionale:"),
            T(lang, "- Images \u2192 CNN (ResNet, EfficientNet)", "- Immagini \u2192 CNN (ResNet, EfficientNet)"),
            T(lang, "- Text \u2192 Transformer (BERT, GPT)", "- Testo \u2192 Transformer (BERT, GPT)"),
            T(lang, "- Audio \u2192 1D CNN / RNN", "- Audio \u2192 1D CNN / RNN"),
            T(lang, "- Point Clouds \u2192 PointNet, DGCNN", "- Point Cloud \u2192 PointNet, DGCNN"),
            "",
            T(lang,
                "Problem: each modality requires a different architecture with domain-specific assumptions.",
                "Problema: ogni modalit\u00e0 richiede un'architettura diversa con assunzioni specifiche."),
            "",
            T(lang,
                "Question: is a single architecture that works on any data type possible?",
                "Domanda: \u00e8 possibile un'unica architettura che funzioni su qualsiasi tipo di dato?"),
            "",
            T(lang, "Transformer Complexity:", "Complessit\u00e0 del Transformer:"),
            T(lang,
                "- Standard Self-attention: O(N\u00b2 \u00d7 d) where N = input length",
                "- Self-attention standard: O(N\u00b2 \u00d7 d) dove N = lunghezza input"),
            T(lang,
                "- For 224\u00d7224 images: N = 50,176 \u2192 Intractable!",
                "- Per immagini 224\u00d7224: N = 50.176 \u2192 Intrattabile!"),
        ],
    )

    # Slide 4: The Perceiver Solution
    add_content_slide(prs,
        T(lang, "The Perceiver Solution", "La Soluzione: Perceiver"),
        [
            T(lang, "Key idea (Jaegle et al., ICML 2021):", "Idea chiave (Jaegle et al., ICML 2021):"),
            T(lang,
                "- 1. Introduce a compact latent array of size M \u226a N",
                "- 1. Introdurre un latent array compatto di dimensione M \u226a N"),
            T(lang,
                "- 2. Cross-attention: latents (queries) extract info from input (keys/values)",
                "- 2. Cross-attention: i latent (query) estraggono info dall'input (key/value)"),
            T(lang,
                "- 3. Latent self-attention: latents refine each other",
                "- 3. Latent self-attention: i latent si affinano tra loro"),
            T(lang,
                "- 4. Repeat iteratively with weight sharing",
                "- 4. Ripetere iterativamente con weight sharing"),
            "",
            T(lang, "Complexity:", "Complessit\u00e0:"),
            T(lang,
                "- Cross-Attention: O(N \u00d7 M)    |    Latent Self-Attn: O(M\u00b2)",
                "- Cross-Attention: O(N \u00d7 M)    |    Latent Self-Attn: O(M\u00b2)"),
            T(lang,
                "- Total: O(N\u00b7M + M\u00b2\u00b7L)  \u2192  Linear in N since M \u226a N!",
                "- Totale: O(N\u00b7M + M\u00b2\u00b7L)  \u2192  Lineare in N dato che M \u226a N!"),
            "",
            T(lang, "Project contributions:", "Contributi del progetto:"),
            T(lang, "- From-scratch PyTorch implementation (Perceiver + Perceiver IO)",
                    "- Implementazione from-scratch in PyTorch (Perceiver + Perceiver IO)"),
            T(lang, "- 3 modalities: images, point clouds, text",
                    "- 3 modalit\u00e0: immagini, point cloud, testo"),
            T(lang, "- Complete ablation study", "- Ablation study completa"),
        ],
    )

    # Slide 5: Perceiver IO
    add_content_slide(prs,
        T(lang, "Perceiver IO: Structured Outputs", "Perceiver IO: Output Strutturati"),
        [
            T(lang, "Key extension (Jaegle et al., ICLR 2022):", "Estensione chiave (Jaegle et al., ICLR 2022):"),
            T(lang,
                "- Encode: cross-attention input \u2192 latent (identical to Perceiver)",
                "- Encode: cross-attention input \u2192 latent (identico al Perceiver)"),
            T(lang,
                "- Process: latent self-attention (\u00d7 L)",
                "- Process: latent self-attention (\u00d7 L)"),
            T(lang,
                "- Decode: learned output queries attend to latents",
                "- Decode: output query apprese attendono ai latent"),
            "",
            T(lang, "Flexibility:", "Flessibilit\u00e0:"),
            T(lang, "- 1 query \u2192 classification", "- 1 query \u2192 classificazione"),
            T(lang, "- N queries \u2192 structured output (e.g., MLM)", "- N query \u2192 output strutturato (es. MLM)"),
            T(lang, "- H\u00d7W queries \u2192 optical flow", "- H\u00d7W query \u2192 optical flow"),
            T(lang, "- Task-specific queries, shared encoder", "- Query task-specific, encoder condiviso"),
        ],
    )
```

- [ ] **Step 2: Verify syntax**

Run: `cd N:/Perceiver_project/slides_V2 && python -c "import generate_ppt_v3; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add slides_V2/generate_ppt_v3.py
git commit -m "feat: add Section 1 (Introduction) bilingual content"
```

---

### Task 3: Bilingual content — Section 2 (Architecture, slides 6-15)

**Files:**
- Modify: `slides_V2/generate_ppt_v3.py`

- [ ] **Step 1: Add `build_section2` function**

Append to `generate_ppt_v3.py`:

```python
def build_section2(prs, lang):
    """Slides 6-15: Architecture."""

    # Slide 6: Section separator
    add_section_slide(prs, T(lang, "Architecture", "Architettura"))

    # Slide 7: Architecture Overview (with image)
    add_content_slide_with_image(prs,
        T(lang, "Perceiver Architecture Overview", "Panoramica dell'Architettura Perceiver"),
        [
            T(lang, "- Input Array (N \u00d7 C): flat generic sequence (pixels, 3D points, bytes)",
                    "- Input Array (N \u00d7 C): sequenza generica piatta (pixel, punti 3D, byte)"),
            T(lang, "- Latent Array (M \u00d7 D): learned compact bottleneck, M \u226a N",
                    "- Latent Array (M \u00d7 D): bottleneck compatto appreso, M \u226a N"),
            T(lang, "- Weight Sharing: cross-attn + self-attn block reused across iterations",
                    "- Weight Sharing: blocco cross-attn + self-attn riutilizzato tra iterazioni"),
            T(lang, "- Output: mean pooling over latents \u2192 classification head",
                    "- Output: mean pooling sui latent \u2192 classification head"),
        ],
        img_path=os.path.join(FIG_DIR, "perceiver_arch.png"),
    )

    # Slide 8: Cross-Attention
    add_content_slide(prs,
        T(lang, "Cross-Attention: The Key Mechanism", "Cross-Attention: Il Meccanismo Chiave"),
        [
            T(lang, "Formally:", "Formalmente:"),
            T(lang, "- Q = X_latent \u00b7 W_Q  \u2208 R^(M \u00d7 d_k)", "- Q = X_latent \u00b7 W_Q  \u2208 R^(M \u00d7 d_k)"),
            T(lang, "- K = X_input \u00b7 W_K   \u2208 R^(N \u00d7 d_k)", "- K = X_input \u00b7 W_K   \u2208 R^(N \u00d7 d_k)"),
            T(lang, "- V = X_input \u00b7 W_V   \u2208 R^(N \u00d7 d_v)", "- V = X_input \u00b7 W_V   \u2208 R^(N \u00d7 d_v)"),
            "",
            T(lang, "CrossAttn(Q,K,V) = softmax(Q\u00b7K\u1d40 / \u221ad_k) \u00b7 V",
                    "CrossAttn(Q,K,V) = softmax(Q\u00b7K\u1d40 / \u221ad_k) \u00b7 V"),
            "",
            T(lang, "The latents are the queries, the input provides keys and values \u2192 informational bottleneck.",
                    "I latent sono le query, l'input fornisce key e value \u2192 bottleneck informativo."),
            "",
            T(lang, "Key insight:", "Insight chiave:"),
            T(lang, "- Attention matrix: M \u00d7 N (not N \u00d7 N)", "- Matrice di attenzione: M \u00d7 N (non N \u00d7 N)"),
            T(lang, "- Complexity: O(M \u00b7 N \u00b7 d_k)", "- Complessit\u00e0: O(M \u00b7 N \u00b7 d_k)"),
            T(lang, "- Decouples depth from input size", "- Disaccoppia la profondit\u00e0 dalla dimensione dell'input"),
        ],
    )

    # Slide 9: Latent Transformer & Weight Sharing
    add_content_slide(prs,
        T(lang, "Latent Transformer & Weight Sharing", "Latent Transformer e Weight Sharing"),
        [
            T(lang, "Self-Attention among latents:", "Self-Attention tra i latent:"),
            T(lang, "- Q = K = V = X_latent  \u2192  O(M\u00b2)", "- Q = K = V = X_latent  \u2192  O(M\u00b2)"),
            T(lang, "- Pre-norm LayerNorm + Residual connections",
                    "- Pre-norm LayerNorm + Connessioni residue"),
            T(lang, "- Feed-forward with 4\u00d7 expansion", "- Feed-forward con espansione 4\u00d7"),
            T(lang, "- GEGLU activation: GEGLU(x) = (x\u00b7W1) \u2299 GELU(x\u00b7W2)",
                    "- Attivazione GEGLU: GEGLU(x) = (x\u00b7W1) \u2299 GELU(x\u00b7W2)"),
            "",
            T(lang, "Weight sharing mechanism:", "Meccanismo di weight sharing:"),
            T(lang, "- A single block applied L times (like an RNN in latent space)",
                    "- Un singolo blocco applicato L volte (come una RNN nello spazio latente)"),
            "",
            T(lang, "Parameter comparison:", "Confronto parametri:"),
            T(lang, "- Unshared (L=4 blocks): ~11.0M params", "- Non condiviso (L=4 blocchi): ~11.0M param"),
            T(lang, "- Shared (L iterations): ~3.35M params", "- Condiviso (L iterazioni): ~3.35M param"),
            T(lang, "- Reduction: ~3\u00d7", "- Riduzione: ~3\u00d7"),
        ],
    )

    # Slide 10: Fourier PE
    add_content_slide(prs,
        T(lang, "Fourier Positional Encoding", "Fourier Positional Encoding"),
        [
            T(lang, "Why is it needed?", "Perch\u00e9 \u00e8 necessaria?"),
            T(lang, "- Attention is permutation-equivariant",
                    "- L'attenzione \u00e8 permutation-equivariant"),
            T(lang, "- Without PE the model has no notion of 'where' data is",
                    "- Senza PE il modello non sa 'dove' sono i dati"),
            T(lang, "- Experimentally confirmed: 35.41% without PE vs 72.23% with PE",
                    "- Confermato sperimentalmente: 35.41% senza PE vs 72.23% con PE"),
            "",
            T(lang, "Fourier PE formula:", "Formula Fourier PE:"),
            T(lang, "- \u03b3(x) = [sin(2\u03c0 f_k x_i), cos(2\u03c0 f_k x_i)] for each freq k, coord i",
                    "- \u03b3(x) = [sin(2\u03c0 f_k x_i), cos(2\u03c0 f_k x_i)] per ogni freq k, coord i"),
            T(lang, "- Log-spaced frequencies: f_k = f_min \u00b7 (f_max/f_min)^(k/K)",
                    "- Frequenze log-spaced: f_k = f_min \u00b7 (f_max/f_min)^(k/K)"),
            "",
            T(lang, "Resulting dimensions:", "Dimensioni risultanti:"),
            T(lang, "- Images (x,y): 12 + 64 = 76", "- Immagini (x,y): 12 + 64 = 76"),
            T(lang, "- Points (x,y,z): 3 + 64 = 67", "- Punti (x,y,z): 3 + 64 = 67"),
            T(lang, "- Text (pos): 1 + 64 = 65", "- Testo (pos): 1 + 64 = 65"),
            "",
            T(lang, "CRITICAL FINDING: PE is the single most important component \u2014 removing it halves accuracy.",
                    "FINDING CRITICO: la PE \u00e8 il componente pi\u00f9 importante \u2014 rimuoverla dimezza l'accuratezza."),
        ],
    )

    # Slide 11: Perceiver IO Decoder (with image)
    add_content_slide_with_image(prs,
        T(lang, "Perceiver IO Decoder with Output Queries",
                "Decoder Perceiver IO con Output Query"),
        [
            T(lang, "Decoder cross-attention (single layer):", "Decoder cross-attention (singolo layer):"),
            T(lang, "- Q_dec = OutputQuery \u00b7 W_Q", "- Q_dec = OutputQuery \u00b7 W_Q"),
            T(lang, "- K_dec = Latent \u00b7 W_K", "- K_dec = Latent \u00b7 W_K"),
            T(lang, "- V_dec = Latent \u00b7 W_V", "- V_dec = Latent \u00b7 W_V"),
            "",
            T(lang, "Output = softmax(Q_dec \u00b7 K_dec\u1d40 / \u221ad_k) \u00b7 V_dec  \u2208 R^(O \u00d7 E)",
                    "Output = softmax(Q_dec \u00b7 K_dec\u1d40 / \u221ad_k) \u00b7 V_dec  \u2208 R^(O \u00d7 E)"),
            "",
            T(lang, "- Number of output queries O is task-dependent and independent of N and M",
                    "- Il numero di output query O \u00e8 task-dependent e indipendente da N e M"),
            T(lang, "- A single decoder cross-attention layer suffices",
                    "- Un singolo layer di decoder cross-attention \u00e8 sufficiente"),
        ],
        img_path=os.path.join(FIG_DIR, "perceiverio_arch.png"),
    )

    # Slide 12: Output Query Design per Task
    add_content_slide(prs,
        T(lang, "Output Query Design per Task", "Design delle Output Query per Task"),
        [
            T(lang, "Task \u2192 Queries \u2192 Output:", "Task \u2192 Query \u2192 Output:"),
            "",
            T(lang, "- Classification: O=1, single learned vector \u2192 class logits",
                    "- Classificazione: O=1, singolo vettore appreso \u2192 class logit"),
            T(lang, "- Optical Flow: O=H\u00d7W, position encodings \u2192 per-pixel flow",
                    "- Optical Flow: O=H\u00d7W, position encoding \u2192 flusso per-pixel"),
            T(lang, "- Masked LM: O=N_masked, position of masked tokens \u2192 token logits (vocab=256)",
                    "- Masked LM: O=N_masked, posizione dei token mascherati \u2192 token logit (vocab=256)"),
            T(lang, "- Multimodal: O=variable, task-specific embeddings \u2192 mixed outputs",
                    "- Multimodale: O=variabile, embedding task-specific \u2192 output misti"),
            "",
            T(lang, "Our implementations:", "Le nostre implementazioni:"),
            T(lang, "- Classification (CIFAR-10, ModelNet40): single query \u2192 linear \u2192 C classes",
                    "- Classificazione (CIFAR-10, ModelNet40): singola query \u2192 lineare \u2192 C classi"),
            T(lang, "- MLM (WikiText-103): one query per masked position \u2192 vocab size 256",
                    "- MLM (WikiText-103): una query per posizione mascherata \u2192 vocab size 256"),
        ],
    )

    # Slide 13: Computational Complexity
    add_content_slide(prs,
        T(lang, "Computational Complexity Analysis", "Analisi della Complessit\u00e0 Computazionale"),
        [
            T(lang, "Model comparison:", "Confronto modelli:"),
            T(lang, "- Standard Transformer: O(N\u00b2\u00b7d\u00b7L)", "- Transformer standard: O(N\u00b2\u00b7d\u00b7L)"),
            T(lang, "- Perceiver: O((N\u00b7M + M\u00b2\u00b7L)\u00b7d)", "- Perceiver: O((N\u00b7M + M\u00b2\u00b7L)\u00b7d)"),
            T(lang, "- Perceiver IO: O(((N+O)\u00b7M + M\u00b2\u00b7L)\u00b7d)", "- Perceiver IO: O(((N+O)\u00b7M + M\u00b2\u00b7L)\u00b7d)"),
            "",
            T(lang, "Concrete example (CIFAR-10):", "Esempio concreto (CIFAR-10):"),
            T(lang, "- N=1024 (patches), M=96, L=4", "- N=1024 (patch), M=96, L=4"),
            T(lang, "- Standard: 1024\u00b2 = 1,048,576 ops", "- Standard: 1024\u00b2 = 1.048.576 ops"),
            T(lang, "- Perceiver: 1024\u00d796 + 96\u00b2\u00d74 = 135,168 ops", "- Perceiver: 1024\u00d796 + 96\u00b2\u00d74 = 135.168 ops"),
            T(lang, "- ~7.8\u00d7 reduction!", "- ~7.8\u00d7 riduzione!"),
            "",
            T(lang, "Scaling advantage (224\u00d7224 images):", "Vantaggio di scala (immagini 224\u00d7224):"),
            T(lang, "- Transformer: ~2.5 \u00d7 10\u2079 ops", "- Transformer: ~2.5 \u00d7 10\u2079 ops"),
            T(lang, "- Perceiver (M=512): ~26M ops  \u2192  ~100\u00d7 reduction!",
                    "- Perceiver (M=512): ~26M ops  \u2192  ~100\u00d7 riduzione!"),
        ],
    )

    # Slide 14: GELU & LAMB
    add_content_slide(prs,
        T(lang, "GELU Activation & LAMB Optimizer", "Attivazione GELU e Ottimizzatore LAMB"),
        [
            T(lang, "GELU (Gaussian Error Linear Unit):", "GELU (Gaussian Error Linear Unit):"),
            T(lang, "- GELU(x) = x \u00b7 \u03a6(x) \u2248 x \u00b7 \u03c3(1.702x)",
                    "- GELU(x) = x \u00b7 \u03a6(x) \u2248 x \u00b7 \u03c3(1.702x)"),
            T(lang, "- Smooth, non-monotonic; standard in Transformers",
                    "- Liscia, non monotona; standard nei Transformer"),
            "",
            T(lang, "LAMB (Layer-wise Adaptive Moments):", "LAMB (Layer-wise Adaptive Moments):"),
            T(lang, "- Per-layer trust ratio: r = ||w|| / ||w + \u03b7\u00b7\u00fb||",
                    "- Trust ratio per-layer: r = ||w|| / ||w + \u03b7\u00b7\u00fb||"),
            T(lang, "- Scales learning rate independently for each layer",
                    "- Scala il learning rate indipendentemente per ogni layer"),
            "",
            T(lang, "Why LAMB for Perceiver?", "Perch\u00e9 LAMB per il Perceiver?"),
            T(lang, "- Cross-attention bottleneck creates high variance gradients",
                    "- Il bottleneck cross-attention crea gradienti ad alta varianza"),
            T(lang, "- Adam destabilizes with large batches; LAMB stays stable",
                    "- Adam destabilizza con batch grandi; LAMB resta stabile"),
            T(lang, "- All experiments use LAMB with cosine annealing + warmup",
                    "- Tutti gli esperimenti usano LAMB con cosine annealing + warmup"),
        ],
    )

    # Slide 15: Code Structure
    add_content_slide(prs,
        T(lang, "Implementation: Code Structure", "Implementazione: Struttura del Codice"),
        [
            T(lang, "Project organization:", "Organizzazione del progetto:"),
            T(lang, "- perceiver/model.py, perceiver_io.py, attention.py",
                    "- perceiver/model.py, perceiver_io.py, attention.py"),
            T(lang, "- perceiver/positional_encoding.py, lamb_optimizer.py",
                    "- perceiver/positional_encoding.py, lamb_optimizer.py"),
            T(lang, "- data/cifar10, modelnet40, wikitext, glue datamodules",
                    "- data/cifar10, modelnet40, wikitext, glue datamodule"),
            T(lang, "- experiments/run_cifar10, run_modelnet40, run_wikitext, run_glue",
                    "- experiments/run_cifar10, run_modelnet40, run_wikitext, run_glue"),
            "",
            T(lang, "Key numbers:", "Numeri chiave:"),
            T(lang, "- Core model: ~600 lines", "- Modello core: ~600 righe"),
            T(lang, "- Data modules: ~500 lines", "- Moduli dati: ~500 righe"),
            T(lang, "- Experiments: ~800 lines", "- Esperimenti: ~800 righe"),
            T(lang, "- Utilities & analysis: ~600 lines", "- Utilit\u00e0 e analisi: ~600 righe"),
            T(lang, "- Total: ~2500 lines", "- Totale: ~2500 righe"),
            "",
            T(lang, "Design choices:", "Scelte di design:"),
            T(lang, "- PyTorch + PyTorch Lightning, modular architecture",
                    "- PyTorch + PyTorch Lightning, architettura modulare"),
            T(lang, "- All experiments reproducible via reproduce.py",
                    "- Tutti gli esperimenti riproducibili via reproduce.py"),
        ],
        font_size=Pt(14),
    )
```

- [ ] **Step 2: Verify syntax**

Run: `cd N:/Perceiver_project/slides_V2 && python -c "import generate_ppt_v3; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add slides_V2/generate_ppt_v3.py
git commit -m "feat: add Section 2 (Architecture) bilingual content"
```

---

### Task 4: Bilingual content — Section 3 (Experiments, slides 16-30)

**Files:**
- Modify: `slides_V2/generate_ppt_v3.py`

- [ ] **Step 1: Add `build_section3` function**

Append to `generate_ppt_v3.py`:

```python
def build_section3(prs, lang):
    """Slides 16-30: Experiments & Results."""

    # Slide 16: Section separator
    add_section_slide(prs, T(lang, "Experiments & Results", "Esperimenti e Risultati"))

    # Slide 17: CIFAR-10 Setup
    add_content_slide(prs,
        T(lang, "CIFAR-10: Experimental Setup", "CIFAR-10: Setup Sperimentale"),
        [
            T(lang, "Goal: validate the Perceiver on raw pixels without any CNN assumptions.",
                    "Obiettivo: validare il Perceiver su pixel grezzi senza assunzioni da CNN."),
            "",
            T(lang, "Perceiver Configuration:", "Configurazione Perceiver:"),
            T(lang, "- Latent dim: 96 \u00d7 384  |  Cross-attn stages: 4  |  Transformer blocks: 4 (shared)",
                    "- Dim. latent: 96 \u00d7 384  |  Stadi cross-attn: 4  |  Blocchi transformer: 4 (condivisi)"),
            T(lang, "- Attention heads: 3  |  Patch size: 2\u00d72  |  Dropout: 0.2",
                    "- Teste di attenzione: 3  |  Patch size: 2\u00d72  |  Dropout: 0.2"),
            T(lang, "- Optimizer: LAMB  |  LR: 0.004  |  Batch size: 64  |  Epochs: 120",
                    "- Ottimizzatore: LAMB  |  LR: 0.004  |  Batch size: 64  |  Epoche: 120"),
            T(lang, "- Total parameters: ~3.35M", "- Parametri totali: ~3.35M"),
            "",
            T(lang, "Ablation study plan:", "Piano dell'ablation study:"),
            T(lang, "- 1. Impact of positional encoding", "- 1. Impatto della positional encoding"),
            T(lang, "- 2. Weight sharing vs no sharing", "- 2. Weight sharing vs no sharing"),
            T(lang, "- 3. Robustness to pixel permutation", "- 3. Robustezza alla permutazione dei pixel"),
            T(lang, "- 4. Perceiver vs Perceiver IO", "- 4. Perceiver vs Perceiver IO"),
        ],
    )

    # Slide 18: CIFAR-10 Ablation Results (with chart)
    add_content_slide_with_image(prs,
        T(lang, "CIFAR-10: Ablation Results", "CIFAR-10: Risultati dell'Ablation"),
        [
            T(lang, "Perceiver IO: 78.20% (best)", "Perceiver IO: 78.20% (migliore)"),
            T(lang, "- No Weight Sharing: 73.61%", "- No Weight Sharing: 73.61%"),
            T(lang, "- Fourier Control: 72.23%", "- Fourier Control: 72.23%"),
            T(lang, "- Baseline Fourier: 69.69%", "- Baseline Fourier: 69.69%"),
            T(lang, "- WS Control: 68.70%", "- WS Control: 68.70%"),
            T(lang, "- Fourier Permuted: 62.31%", "- Fourier Permuted: 62.31%"),
            T(lang, "- Learned PE Permuted: 55.12%", "- Learned PE Permuted: 55.12%"),
            T(lang, "- RGB Only (no PE): 35.41% (worst)", "- RGB Only (no PE): 35.41% (peggiore)"),
            "",
            T(lang, "Key observations:", "Osservazioni chiave:"),
            T(lang, "- PE is essential: -36.82pp without it", "- PE \u00e8 essenziale: -36.82pp senza"),
            T(lang, "- Perceiver IO > Perceiver base", "- Perceiver IO > Perceiver base"),
            T(lang, "- Weight sharing: competitive with 3\u00d7 fewer params",
                    "- Weight sharing: competitivo con 3\u00d7 meno parametri"),
        ],
        img_path=os.path.join(RESULTS_DIR, "cifar10_chart.png"),
        font_size=Pt(13),
    )

    # Slide 19: PE Impact
    add_content_slide(prs,
        T(lang, "CIFAR-10: Impact of Positional Encoding",
                "CIFAR-10: Impatto della Positional Encoding"),
        [
            T(lang, "With Fourier PE: 72.23%  vs  Without PE (RGB only): 35.41%",
                    "Con Fourier PE: 72.23%  vs  Senza PE (solo RGB): 35.41%"),
            T(lang, "- Delta: -36.82 percentage points", "- Delta: -36.82 punti percentuali"),
            "",
            T(lang, "Why does this happen?", "Perch\u00e9 succede questo?"),
            T(lang, "- Without PE, attention is permutation-equivariant",
                    "- Senza PE, l'attenzione \u00e8 permutation-equivariant"),
            T(lang, "- The model cannot distinguish spatial positions",
                    "- Il modello non pu\u00f2 distinguere le posizioni spaziali"),
            T(lang, "- Effectively treats image as a 'bag of patches'",
                    "- Tratta l'immagine come un 'bag of patches'"),
            T(lang, "- Loses all 2D spatial structure", "- Perde tutta la struttura spaziale 2D"),
            "",
            T(lang, "This confirms the original paper's hypothesis that PE is the most critical component.",
                    "Questo conferma l'ipotesi del paper originale che la PE \u00e8 il componente pi\u00f9 critico."),
        ],
    )

    # Slide 20: Weight Sharing
    add_content_slide(prs,
        T(lang, "CIFAR-10: Weight Sharing Analysis", "CIFAR-10: Analisi del Weight Sharing"),
        [
            T(lang, "Comparison:", "Confronto:"),
            T(lang, "- Shared (baseline): 3.35M params \u2192 69.69%",
                    "- Condiviso (baseline): 3.35M param \u2192 69.69%"),
            T(lang, "- WS Control: 3.35M params \u2192 68.70%",
                    "- WS Control: 3.35M param \u2192 68.70%"),
            T(lang, "- No Sharing: 11.0M params \u2192 73.61%",
                    "- Non condiviso: 11.0M param \u2192 73.61%"),
            "",
            T(lang, "Trade-off:", "Trade-off:"),
            T(lang, "- No sharing: +3.92pp accuracy but 3.3\u00d7 more parameters",
                    "- No sharing: +3.92pp accuracy ma 3.3\u00d7 pi\u00f9 parametri"),
            T(lang, "- Efficiency: +1.19pp per 1M extra params",
                    "- Efficienza: +1.19pp per 1M parametri extra"),
            "",
            T(lang, "Why weight sharing works:", "Perch\u00e9 il weight sharing funziona:"),
            T(lang, "- Acts like a recurrent processor in latent space",
                    "- Funziona come un processore ricorrente nello spazio latente"),
            T(lang, "- Iterative refinement + implicit regularization",
                    "- Affinamento iterativo + regolarizzazione implicita"),
            T(lang, "- Achieves 93.5% of no-sharing accuracy with 30% of the parameters",
                    "- Raggiunge il 93.5% dell'accuratezza no-sharing con il 30% dei parametri"),
        ],
    )

    # Slide 21: Permutation Robustness
    add_content_slide(prs,
        T(lang, "CIFAR-10: Permutation Robustness", "CIFAR-10: Robustezza alla Permutazione"),
        [
            T(lang, "Experiment: apply a fixed random permutation to all pixel positions.",
                    "Esperimento: applicare una permutazione casuale fissa a tutte le posizioni dei pixel."),
            "",
            T(lang, "Results:", "Risultati:"),
            T(lang, "- Fourier PE (normal): 72.23%", "- Fourier PE (normale): 72.23%"),
            T(lang, "- Fourier PE (permuted): 62.31%  (\u221209.92pp)",
                    "- Fourier PE (permutato): 62.31%  (\u221209.92pp)"),
            T(lang, "- Learned PE (permuted): 55.12%  (\u221217.11pp)",
                    "- Learned PE (permutato): 55.12%  (\u221217.11pp)"),
            "",
            T(lang, "Analysis:", "Analisi:"),
            T(lang, "- Fourier PE is more robust to permutation than learned PE",
                    "- La Fourier PE \u00e8 pi\u00f9 robusta alla permutazione della learned PE"),
            T(lang, "- Fourier frequencies encode relative distances, partially surviving permutation",
                    "- Le frequenze Fourier codificano distanze relative, sopravvivendo parzialmente"),
            T(lang, "- Learned PE memorizes absolute positions \u2192 breaks on shuffled data",
                    "- La learned PE memorizza posizioni assolute \u2192 si rompe su dati mescolati"),
            "",
            T(lang, "Fourier PE retains 86% of accuracy under permutation vs 76% for learned PE.",
                    "La Fourier PE mantiene l'86% dell'accuracy sotto permutazione vs 76% per la learned PE."),
        ],
    )

    # Slide 22: Perceiver vs Perceiver IO
    add_content_slide(prs,
        T(lang, "CIFAR-10: Perceiver vs Perceiver IO", "CIFAR-10: Perceiver vs Perceiver IO"),
        [
            T(lang, "Perceiver (Fourier): 72.23%", "Perceiver (Fourier): 72.23%"),
            T(lang, "Perceiver IO: 78.20%  (+5.97pp)", "Perceiver IO: 78.20%  (+5.97pp)"),
            "",
            T(lang, "Why Perceiver IO is better:", "Perch\u00e9 Perceiver IO \u00e8 migliore:"),
            T(lang, "- Output query acts as a learned aggregation mechanism",
                    "- La output query funziona come un meccanismo di aggregazione appreso"),
            T(lang, "- More flexible than simple mean pooling",
                    "- Pi\u00f9 flessibile del semplice mean pooling"),
            T(lang, "- Decoder cross-attention can selectively attend to relevant latents",
                    "- La decoder cross-attention pu\u00f2 attendere selettivamente ai latent rilevanti"),
            T(lang, "- Additional parameters in decoder head provide extra capacity",
                    "- Parametri aggiuntivi nel decoder head forniscono capacit\u00e0 extra"),
        ],
    )

    # Slide 23: ModelNet40 Setup & Results
    add_content_slide(prs,
        T(lang, "ModelNet40: Setup & Results", "ModelNet40: Setup e Risultati"),
        [
            T(lang, "Configuration:", "Configurazione:"),
            T(lang, "- Latent dim: 128 \u00d7 512  |  Cross-attn stages: 2  |  Transformer blocks: 6 (shared)",
                    "- Dim. latent: 128 \u00d7 512  |  Stadi cross-attn: 2  |  Blocchi transformer: 6 (condivisi)"),
            T(lang, "- Points per object: 2048  |  Optimizer: LAMB  |  Epochs: 200  |  Batch size: 128",
                    "- Punti per oggetto: 2048  |  Ottimizzatore: LAMB  |  Epoche: 200  |  Batch size: 128"),
            "",
            T(lang, "Results:", "Risultati:"),
            T(lang, "- Baseline (scale): 84.16%", "- Baseline (scala): 84.16%"),
            T(lang, "- + Rotation: 83.06%", "- + Rotazione: 83.06%"),
            T(lang, "- + Translation: 82.90%", "- + Traslazione: 82.90%"),
            T(lang, "- Original Paper: 85.7%", "- Paper originale: 85.7%"),
            "",
            T(lang, "Excellent reproduction: gap of only 1.54pp vs original paper, despite batch size 128 vs 512 (GPU constraints).",
                    "Ottima riproduzione: gap di solo 1.54pp vs paper originale, nonostante batch size 128 vs 512 (limiti GPU)."),
        ],
    )

    # Slide 24: ModelNet40 Augmentation (with chart)
    add_content_slide_with_image(prs,
        T(lang, "ModelNet40: Augmentation Effects", "ModelNet40: Effetti dell'Augmentation"),
        [
            T(lang, "Scale only (baseline): 84.16% (best)", "Solo scala (baseline): 84.16% (migliore)"),
            T(lang, "- + Rotation: -1.10pp", "- + Rotazione: -1.10pp"),
            T(lang, "- + Translation: -1.26pp", "- + Traslazione: -1.26pp"),
            "",
            T(lang, "Why more augmentation hurts:", "Perch\u00e9 pi\u00f9 augmentation peggiora:"),
            T(lang, "- Limited model capacity (128 latents)",
                    "- Capacit\u00e0 limitata del modello (128 latent)"),
            T(lang, "- Aggressive augmentation requires more epochs",
                    "- Augmentation aggressiva richiede pi\u00f9 epoche"),
            T(lang, "- Fourier PE encodes absolute coords \u2192 rotation/translation shifts the encoding",
                    "- La Fourier PE codifica coordinate assolute \u2192 rotazione/traslazione spostano la codifica"),
        ],
        img_path=os.path.join(RESULTS_DIR, "modelnet40_chart.png"),
    )

    # Slide 25: WikiText-103 MLM
    add_content_slide(prs,
        T(lang, "WikiText-103: MLM Pre-training", "WikiText-103: Pre-training MLM"),
        [
            T(lang, "Goal: pre-train a general language encoder without specialized tokenizers.",
                    "Obiettivo: pre-addestrare un encoder linguistico generale senza tokenizer specializzati."),
            "",
            T(lang, "Configuration:", "Configurazione:"),
            T(lang, "- Tokenization: Byte-level (UTF-8)  |  Vocabulary: 256",
                    "- Tokenizzazione: Byte-level (UTF-8)  |  Vocabolario: 256"),
            T(lang, "- Sequence length: 1024  |  Mask probability: 15%",
                    "- Lunghezza sequenza: 1024  |  Probabilit\u00e0 di mascheramento: 15%"),
            T(lang, "- Model: Perceiver IO  |  Parameters: ~10M  |  Epochs: 50",
                    "- Modello: Perceiver IO  |  Parametri: ~10M  |  Epoche: 50"),
            "",
            T(lang, "Why byte-level?", "Perch\u00e9 byte-level?"),
            T(lang, "- No BPE/WordPiece tokenizer needed", "- Nessun tokenizer BPE/WordPiece necessario"),
            T(lang, "- Truly language-agnostic input", "- Input veramente agnostico alla lingua"),
            T(lang, "- Fixed vocabulary of size 256", "- Vocabolario fisso di dimensione 256"),
            T(lang, "- Trade-off: longer sequences for same text",
                    "- Trade-off: sequenze pi\u00f9 lunghe per lo stesso testo"),
        ],
    )

    # Slide 26: GLUE Fine-tuning
    add_content_slide(prs,
        T(lang, "GLUE: Fine-tuning on 8 Tasks", "GLUE: Fine-tuning su 8 Task"),
        [
            T(lang, "Transfer learning pipeline:", "Pipeline di transfer learning:"),
            T(lang, "- 1. Pre-train MLM encoder on WikiText-103",
                    "- 1. Pre-addestramento encoder MLM su WikiText-103"),
            T(lang, "- 2. Transfer encoder weights to classification model",
                    "- 2. Trasferimento pesi encoder al modello di classificazione"),
            T(lang, "- 3. Replace output queries with classification query",
                    "- 3. Sostituzione output query con classification query"),
            "",
            T(lang, "Fine-tuning setup:", "Setup fine-tuning:"),
            T(lang, "- 30 epochs  |  LR = 5\u00d710\u207b\u2074  |  Byte-level input",
                    "- 30 epoche  |  LR = 5\u00d710\u207b\u2074  |  Input byte-level"),
            "",
            T(lang, "8 GLUE benchmark tasks:", "8 task del benchmark GLUE:"),
            T(lang, "- Single sentence: CoLA, SST-2", "- Frase singola: CoLA, SST-2"),
            T(lang, "- Similarity: MRPC, STS-B, QQP", "- Similarit\u00e0: MRPC, STS-B, QQP"),
            T(lang, "- Inference: MNLI, QNLI, RTE", "- Inferenza: MNLI, QNLI, RTE"),
        ],
    )

    # Slide 27: GLUE Per-Task (with chart)
    add_content_slide_with_image(prs,
        T(lang, "GLUE: Per-Task Analysis", "GLUE: Analisi per Task"),
        [
            T(lang, "Key observations:", "Osservazioni chiave:"),
            T(lang, "- Results above majority baseline on most tasks",
                    "- Risultati sopra la baseline di maggioranza sulla maggior parte dei task"),
            T(lang, "- SST-2 (sentiment): strong performance",
                    "- SST-2 (sentiment): performance forte"),
            T(lang, "- CoLA: hardest \u2014 requires fine-grained grammaticality judgment",
                    "- CoLA: pi\u00f9 difficile \u2014 richiede giudizio grammaticale fine"),
            T(lang, "- MNLI: 3-class NLI, challenging for byte-level model",
                    "- MNLI: NLI a 3 classi, sfidante per modello byte-level"),
            "",
            T(lang, "Limitations:", "Limitazioni:"),
            T(lang, "- Byte-level tokenization \u2192 long sequences",
                    "- Tokenizzazione byte-level \u2192 sequenze lunghe"),
            T(lang, "- Smaller model (~10M) vs BERT (110M)",
                    "- Modello pi\u00f9 piccolo (~10M) vs BERT (110M)"),
            T(lang, "- No subword semantics available",
                    "- Nessuna semantica a livello di subword disponibile"),
        ],
        img_path=os.path.join(RESULTS_DIR, "glue_chart.png"),
    )

    # Slide 28: Convergence (with chart)
    add_content_slide_with_image(prs,
        T(lang, "Convergence: All Experiments", "Convergenza: Tutti gli Esperimenti"),
        [
            T(lang, "Convergence speed by modality:", "Velocit\u00e0 di convergenza per modalit\u00e0:"),
            T(lang, "- Point Clouds: ~50 epochs (fast)", "- Point Cloud: ~50 epoche (veloce)"),
            T(lang, "- Text (GLUE FT): ~15-30 epochs", "- Testo (GLUE FT): ~15-30 epoche"),
            T(lang, "- Images (CIFAR-10): ~120 epochs (slow)", "- Immagini (CIFAR-10): ~120 epoche (lento)"),
            "",
            T(lang, "Observations:", "Osservazioni:"),
            T(lang, "- Point clouds: low-dimensional input, fast convergence",
                    "- Point cloud: input bassa dimensionalit\u00e0, convergenza rapida"),
            T(lang, "- GLUE: transfer learning accelerates training",
                    "- GLUE: il transfer learning accelera il training"),
            T(lang, "- CIFAR-10: pixel-level input requires many iterations",
                    "- CIFAR-10: input a livello pixel richiede molte iterazioni"),
        ],
        img_path=os.path.join(RESULTS_DIR, "convergence_chart.png"),
    )

    # Slide 29: Attention Maps (two images side by side)
    # Use layout 2 with two manually placed images
    slide = prs.slides.add_slide(prs.slide_layouts[2])
    _set_placeholder(slide, 0,
        T(lang, "Attention Maps Analysis", "Analisi delle Attention Map"))
    # Narrow body text to bottom
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 13:
            shape.top = Inches(5.2)
            shape.height = Inches(1.5)
            break
    _set_placeholder_formatted(slide, 13, [
        T(lang,
            "With Fourier PE, cross-attention learns spatially coherent patterns. Without PE, attention entropy remains high and patterns lack structure.",
            "Con Fourier PE, la cross-attention apprende pattern spazialmente coerenti. Senza PE, l'entropia dell'attenzione resta alta e i pattern mancano di struttura."),
    ], font_size=Pt(14))
    # Left image: Fourier PE
    img1 = os.path.join(ATTN_DIR, "exp1_baseline_fourier_evolution.png")
    img2 = os.path.join(ATTN_DIR, "exp3B_rgb_only_evolution.png")
    if os.path.exists(img1):
        slide.shapes.add_picture(img1, Inches(0.5), Inches(1.8), width=Inches(5.8))
    if os.path.exists(img2):
        slide.shapes.add_picture(img2, Inches(6.8), Inches(1.8), width=Inches(5.8))

    # Slide 30: Comprehensive Results Summary
    add_content_slide(prs,
        T(lang, "Comprehensive Results Summary", "Riepilogo Completo dei Risultati"),
        [
            T(lang, "Images (CIFAR-10):", "Immagini (CIFAR-10):"),
            T(lang, "- Perceiver: 72.23%  |  Perceiver IO: 78.20% (best)  |  Paper: >90%  |  7 ablations",
                    "- Perceiver: 72.23%  |  Perceiver IO: 78.20% (migliore)  |  Paper: >90%  |  7 ablation"),
            "",
            T(lang, "Point Clouds (ModelNet40):", "Point Cloud (ModelNet40):"),
            T(lang, "- Perceiver: 84.16%  |  Paper: 85.7%  |  3 augmentation experiments",
                    "- Perceiver: 84.16%  |  Paper: 85.7%  |  3 esperimenti di augmentation"),
            "",
            T(lang, "Text (WikiText-103 + GLUE):", "Testo (WikiText-103 + GLUE):"),
            T(lang, "- MLM pre-training + 8 GLUE fine-tuning tasks",
                    "- Pre-training MLM + 8 task di fine-tuning GLUE"),
            "",
            T(lang, "Overall numbers:", "Numeri complessivi:"),
            T(lang, "- ~2500 lines of PyTorch code (from-scratch)",
                    "- ~2500 righe di codice PyTorch (from-scratch)"),
            T(lang, "- 20+ distinct training runs", "- 20+ esperimenti di training distinti"),
            T(lang, "- 3 modalities: 2D images, 3D points, text",
                    "- 3 modalit\u00e0: immagini 2D, punti 3D, testo"),
            T(lang, "- 100% reproducible via reproduce.py",
                    "- 100% riproducibile via reproduce.py"),
        ],
    )
```

- [ ] **Step 2: Verify syntax**

Run: `cd N:/Perceiver_project/slides_V2 && python -c "import generate_ppt_v3; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add slides_V2/generate_ppt_v3.py
git commit -m "feat: add Section 3 (Experiments & Results) bilingual content"
```

---

### Task 5: Bilingual content — Section 4 (Discussion) + Closing (slides 31-39)

**Files:**
- Modify: `slides_V2/generate_ppt_v3.py`

- [ ] **Step 1: Add `build_section4` and `build_closing` functions**

Append to `generate_ppt_v3.py`:

```python
def build_section4(prs, lang):
    """Slides 31-36: Criticalities & Discussion."""

    # Slide 31: Section separator
    add_section_slide(prs, T(lang, "Criticalities & Discussion", "Criticit\u00e0 e Discussione"))

    # Slide 32: Gap vs Original Paper
    add_content_slide(prs,
        T(lang, "Gap vs Original Paper", "Gap vs Paper Originale"),
        [
            T(lang, "ModelNet40 (Point Clouds):", "ModelNet40 (Point Cloud):"),
            T(lang, "- Original Paper: 85.7%  |  Our implementation: 84.16%  |  Gap: 1.54pp",
                    "- Paper originale: 85.7%  |  Nostra implementazione: 84.16%  |  Gap: 1.54pp"),
            T(lang, "- Excellent reproduction. Small gap due to batch size (128 vs 512).",
                    "- Ottima riproduzione. Gap piccolo dovuto al batch size (128 vs 512)."),
            "",
            T(lang, "CIFAR-10 (Images):", "CIFAR-10 (Immagini):"),
            T(lang, "- Original Paper: >90%  |  Our Perceiver IO: 78.20%  |  Gap: ~12pp",
                    "- Paper originale: >90%  |  Nostro Perceiver IO: 78.20%  |  Gap: ~12pp"),
            T(lang, "- Significant gap. Caused by hardware constraints: 512 TPU cores vs single consumer GPU.",
                    "- Gap significativo. Causato da limiti hardware: 512 core TPU vs singola GPU consumer."),
            "",
            T(lang, "Root cause: batch size (64 vs 1024), latent count (96 vs 256), and compute time limitations.",
                    "Causa principale: batch size (64 vs 1024), numero di latent (96 vs 256) e limiti di tempo di calcolo."),
        ],
    )

    # Slide 33: Hardware Limitations
    add_content_slide(prs,
        T(lang, "Hardware Limitations & Design Choices",
                "Limitazioni Hardware e Scelte di Design"),
        [
            T(lang, "Comparison with original paper (Google):", "Confronto con paper originale (Google):"),
            T(lang, "- Hardware: 512 TPU v3 cores vs 1 NVIDIA consumer GPU",
                    "- Hardware: 512 core TPU v3 vs 1 GPU NVIDIA consumer"),
            T(lang, "- Batch size: up to 1024 vs max 64-128",
                    "- Batch size: fino a 1024 vs max 64-128"),
            T(lang, "- Latents: 256-512 vs 96-128", "- Latent: 256-512 vs 96-128"),
            T(lang, "- Model size: 30-100M+ vs 3.35-10M params",
                    "- Dim. modello: 30-100M+ vs 3.35-10M param"),
            "",
            T(lang, "Consequences:", "Conseguenze:"),
            T(lang, "- LAMB optimizer less effective with small batches",
                    "- Ottimizzatore LAMB meno efficace con batch piccoli"),
            T(lang, "- Fewer latents limit representational capacity",
                    "- Meno latent limitano la capacit\u00e0 rappresentativa"),
            "",
            T(lang, "Mitigation strategies:", "Strategie di mitigazione:"),
            T(lang, "- Weight sharing to reduce memory", "- Weight sharing per ridurre la memoria"),
            T(lang, "- Gradient accumulation where possible",
                    "- Gradient accumulation dove possibile"),
            T(lang, "- Careful cosine annealing schedule",
                    "- Cosine annealing schedule calibrato"),
        ],
    )

    # Slide 34: Possible Improvements
    add_content_slide(prs,
        T(lang, "Possible Improvements", "Possibili Miglioramenti"),
        [
            T(lang, "Architecture:", "Architettura:"),
            T(lang, "- Increase latent count (M=256+) with gradient checkpointing",
                    "- Aumentare i latent (M=256+) con gradient checkpointing"),
            T(lang, "- Multi-scale cross-attention (coarse \u2192 fine)",
                    "- Cross-attention multi-scala (grossolano \u2192 fine)"),
            T(lang, "- Explore relative PE (RoPE, ALiBi)", "- Esplorare PE relative (RoPE, ALiBi)"),
            "",
            T(lang, "Training:", "Training:"),
            T(lang, "- Larger batches with gradient accumulation",
                    "- Batch pi\u00f9 grandi con gradient accumulation"),
            T(lang, "- Mixed precision (FP16/BF16)", "- Mixed precision (FP16/BF16)"),
            T(lang, "- Advanced augmentation (CutMix, MixUp)",
                    "- Augmentation avanzata (CutMix, MixUp)"),
            "",
            T(lang, "Data & Analysis:", "Dati e Analisi:"),
            T(lang, "- Subword tokenization for NLP", "- Tokenizzazione subword per NLP"),
            T(lang, "- Multi-modal joint training", "- Training congiunto multi-modale"),
            T(lang, "- Comparison with efficient Transformers (Linformer, Performer)",
                    "- Confronto con Transformer efficienti (Linformer, Performer)"),
        ],
    )

    # Slide 35: Lessons Learned
    add_content_slide(prs,
        T(lang, "Lessons Learned", "Lezioni Apprese"),
        [
            T(lang, "What works well:", "Cosa funziona bene:"),
            T(lang, "- Genuinely modality-agnostic architecture",
                    "- Architettura genuinamente agnostica alla modalit\u00e0"),
            T(lang, "- Weight sharing: 3\u00d7 fewer params, comparable performance",
                    "- Weight sharing: 3\u00d7 meno parametri, performance comparabile"),
            T(lang, "- Perceiver IO outperforms base Perceiver on CIFAR-10",
                    "- Perceiver IO supera il Perceiver base su CIFAR-10"),
            T(lang, "- Transfer learning MLM \u2192 GLUE is effective",
                    "- Transfer learning MLM \u2192 GLUE \u00e8 efficace"),
            "",
            T(lang, "Observed limitations:", "Limitazioni osservate:"),
            T(lang, "- Performance gap vs paper on images (~12pp)",
                    "- Gap di performance vs paper sulle immagini (~12pp)"),
            T(lang, "- LAMB optimizer sensitive to batch size",
                    "- Ottimizzatore LAMB sensibile al batch size"),
            T(lang, "- Byte-level tokenization \u2192 very long sequences",
                    "- Tokenizzazione byte-level \u2192 sequenze molto lunghe"),
            "",
            T(lang, "MAIN TAKEAWAY: Positional encoding is the single most critical component \u2014 without it, accuracy drops from 72% to 35% on CIFAR-10.",
                    "CONCLUSIONE PRINCIPALE: la positional encoding \u00e8 il componente pi\u00f9 critico \u2014 senza, l'accuracy cala dal 72% al 35% su CIFAR-10."),
        ],
    )

    # Slide 36: Conclusions
    add_content_slide(prs,
        T(lang, "Conclusions", "Conclusioni"),
        [
            T(lang, "Strengths of the Perceiver:", "Punti di forza del Perceiver:"),
            T(lang, "- Single architecture for images, point clouds, and text",
                    "- Singola architettura per immagini, point cloud e testo"),
            T(lang, "- Linear complexity in input size",
                    "- Complessit\u00e0 lineare nella dimensione dell'input"),
            T(lang, "- 60% memory savings via weight sharing",
                    "- 60% risparmio di memoria via weight sharing"),
            T(lang, "- Zero CNN/RNN domain-specific assumptions",
                    "- Zero assunzioni specifiche di dominio CNN/RNN"),
            "",
            T(lang, "Weaknesses:", "Punti deboli:"),
            T(lang, "- Requires large compute to match specialized models",
                    "- Richiede molto calcolo per eguagliare modelli specializzati"),
            T(lang, "- Heavily depends on positional encoding",
                    "- Dipende fortemente dalla positional encoding"),
            T(lang, "- Latent bottleneck limits fine-grained spatial reasoning",
                    "- Il bottleneck latente limita il ragionamento spaziale fine"),
            "",
            T(lang, "The Perceiver demonstrates that a single general-purpose architecture can handle diverse modalities. A compelling step toward modality-agnostic perception.",
                    "Il Perceiver dimostra che un'architettura general-purpose pu\u00f2 gestire modalit\u00e0 diverse. Un passo promettente verso la percezione agnostica alla modalit\u00e0."),
        ],
    )


def build_closing(prs, lang):
    """Slides 37-39: References, Expected Questions, Thank You."""

    # Slide 37: References
    add_content_slide(prs,
        T(lang, "References", "Riferimenti"),
        [
            "- Jaegle et al., Perceiver: General Perception with Iterative Attention, ICML 2021",
            "- Jaegle et al., Perceiver IO: A General Architecture for Structured Inputs & Outputs, ICLR 2022",
            "- Vaswani et al., Attention Is All You Need, NeurIPS 2017",
            "- You et al., Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes, ICLR 2020",
            "- Tancik et al., Fourier Features Let Networks Learn High Frequency Functions, NeurIPS 2020",
            "- Shazeer, GLU Variants Improve Transformer, arXiv 2020",
        ],
        font_size=Pt(14),
    )

    # Slide 38: Expected Questions
    add_content_slide(prs,
        T(lang, "Appendix: Expected Questions", "Appendice: Domande Attese"),
        [
            T(lang, "Architecture:", "Architettura:"),
            T(lang, "- Why cross-attention instead of full self-attention?  \u2192 Reduces O(N\u00b2) to O(NM)",
                    "- Perch\u00e9 cross-attention invece di full self-attention?  \u2192 Riduce O(N\u00b2) a O(NM)"),
            T(lang, "- How does weight sharing affect performance?  \u2192 -3.92pp for 3\u00d7 fewer params",
                    "- Come influisce il weight sharing?  \u2192 -3.92pp per 3\u00d7 meno parametri"),
            T(lang, "- Why Fourier PE over learned PE?  \u2192 More robust to permutation",
                    "- Perch\u00e9 Fourier PE rispetto a learned PE?  \u2192 Pi\u00f9 robusta alla permutazione"),
            T(lang, "- What is the role of GEGLU?  \u2192 Higher non-linear capacity, stable gradients",
                    "- Qual \u00e8 il ruolo di GEGLU?  \u2192 Maggiore capacit\u00e0 non-lineare, gradienti stabili"),
            "",
            T(lang, "Experiments:", "Esperimenti:"),
            T(lang, "- Why is the CIFAR-10 gap so large?  \u2192 Hardware: batch 64 vs 1024, 96 vs 256 latents",
                    "- Perch\u00e9 il gap CIFAR-10 \u00e8 cos\u00ec grande?  \u2192 Hardware: batch 64 vs 1024, 96 vs 256 latent"),
            T(lang, "- Why byte-level tokenization?  \u2192 Truly modality-agnostic, no external tokenizer",
                    "- Perch\u00e9 tokenizzazione byte-level?  \u2192 Veramente agnostico, nessun tokenizer esterno"),
            T(lang, "- How does Perceiver IO improve on Perceiver?  \u2192 Learned output queries vs mean pooling",
                    "- Come migliora Perceiver IO rispetto a Perceiver?  \u2192 Output query apprese vs mean pooling"),
            T(lang, "- Could this replace ViT/BERT?  \u2192 Not yet at limited scale; promising at Google scale",
                    "- Pu\u00f2 sostituire ViT/BERT?  \u2192 Non ancora a scala limitata; promettente a scala Google"),
        ],
        font_size=Pt(13),
    )

    # Slide 39: Thank You
    add_closing_slide(
        prs,
        main_text=T(lang, "Thank you!", "Grazie!"),
        sub_text=T(lang, "Questions?", "Domande?"),
        name="Francesco Gigli",
        role="",
        contact="francesco.gigli@stud.unifi.it",
        extra=T(lang, "Code available upon request", "Codice disponibile su richiesta"),
    )
```

- [ ] **Step 2: Verify syntax**

Run: `cd N:/Perceiver_project/slides_V2 && python -c "import generate_ppt_v3; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add slides_V2/generate_ppt_v3.py
git commit -m "feat: add Section 4 (Discussion) + Closing bilingual content"
```

---

### Task 6: Main function and generation

**Files:**
- Modify: `slides_V2/generate_ppt_v3.py`

- [ ] **Step 1: Add `build` and `main` functions**

Append to `generate_ppt_v3.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def build(lang):
    """Build the full presentation for the given language ('en' or 'it')."""
    prs = Presentation(TEMPLATE)
    build_section1(prs, lang)
    build_section2(prs, lang)
    build_section3(prs, lang)
    build_section4(prs, lang)
    build_closing(prs, lang)

    suffix = "eng" if lang == "en" else "ita"
    output_path = os.path.join(SCRIPT_DIR, f"presentation_v3_{suffix}.pptx")
    prs.save(output_path)
    print(f"Generated: {output_path}")
    return output_path


def main():
    build("en")
    build("it")
    print("Done! Both presentations generated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and generate both presentations**

Run: `cd N:/Perceiver_project/slides_V2 && python generate_ppt_v3.py`
Expected:
```
Generated: .../slides_V2/presentation_v3_eng.pptx
Generated: .../slides_V2/presentation_v3_ita.pptx
Done! Both presentations generated.
```

- [ ] **Step 3: Verify the output files exist and have reasonable size**

Run: `ls -lh N:/Perceiver_project/slides_V2/presentation_v3_*.pptx`
Expected: two files, each > 500KB

- [ ] **Step 4: Commit**

```bash
git add slides_V2/generate_ppt_v3.py
git commit -m "feat: add main entry point, generate bilingual DIEF presentations"
```

---

### Task 7: Smoke test and fix issues

**Files:**
- Modify: `slides_V2/generate_ppt_v3.py` (if needed)

- [ ] **Step 1: Verify slide count**

Run:
```bash
cd N:/Perceiver_project/slides_V2 && python -c "
from pptx import Presentation
for name in ['presentation_v3_eng.pptx', 'presentation_v3_ita.pptx']:
    prs = Presentation(name)
    print(f'{name}: {len(prs.slides)} slides')
    for i, slide in enumerate(prs.slides):
        title = ''
        for shape in slide.placeholders:
            if shape.placeholder_format.idx == 0:
                title = shape.text
                break
        print(f'  [{i+1}] {title}')
"
```
Expected: 39 slides for each file, with correct titles in the respective language.

- [ ] **Step 2: Verify images are embedded**

Run:
```bash
cd N:/Perceiver_project/slides_V2 && python -c "
from pptx import Presentation
prs = Presentation('presentation_v3_eng.pptx')
img_count = 0
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.shape_type == 13:  # Picture
            img_count += 1
print(f'Total images embedded: {img_count}')
"
```
Expected: at least 8 images (perceiver_arch, perceiverio_arch, cifar10_chart, modelnet40_chart, glue_chart, convergence_chart, 2 attention maps).

- [ ] **Step 3: Fix any issues found and re-generate**

If any issues are found in steps 1-2, fix the script and re-run `python generate_ppt_v3.py`.

- [ ] **Step 4: Commit final version**

```bash
git add slides_V2/generate_ppt_v3.py slides_V2/presentation_v3_eng.pptx slides_V2/presentation_v3_ita.pptx
git commit -m "feat: finalize bilingual DIEF PowerPoint presentations (39 slides each)"
```
