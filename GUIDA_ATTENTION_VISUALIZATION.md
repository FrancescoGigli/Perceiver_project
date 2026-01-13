# 📊 GUIDA: Visualizzazione Attention Maps

**Script principale:** `visualize_attention.py`  
**Funzionalità:** Visualizzazione completa e analisi delle attention maps salvate durante il training

---

## 🚀 **COMANDI PRINCIPALI**

### **1. Visualizzazione Base (Single Epoch)**
```bash
# Visualizza tutti gli epochs di tutti gli esperimenti
python visualize_attention.py --logs_dir logs

# Visualizza con colormap specifico
python visualize_attention.py --logs_dir logs --colormap viridis

# Visualizza epochs specifici
python visualize_attention.py --logs_dir logs --epochs 1 10 50 120
```

### **2. Analisi Evoluzione Temporale**
```bash
# Crea visualizzazione evoluzione attention nel tempo
python visualize_attention.py --logs_dir logs --create_evolution

# Evoluzione per epochs specifici con colormap plasma
python visualize_attention.py --logs_dir logs --create_evolution --epochs 1 21 41 61 81 101 120 --colormap plasma
```

### **3. Controlli Visualizzazione**
```bash
# Alpha blending (trasparenza overlay)
python visualize_attention.py --logs_dir logs --alpha 0.8

# Colormap disponibili: jet, viridis, plasma, inferno, hot, cool, seismic
python visualize_attention.py --logs_dir logs --colormap inferno --alpha 0.5
```

---

## 📁 **STRUTTURA OUTPUT**

Lo script genera per ogni experimento:

### **File Per Epoch:**
- `comprehensive_epoch_X_analysis.png` - Analisi dettagliata per epoch X

### **File Evoluzione:**
- `attention_evolution_analysis.png` - Evoluzione attention nel tempo

---

## 🎯 **ESEMPI PRATICI**

### **Esempio 1: Analisi Completa Esperimento Specifico**
```bash
# Vai nella directory del progetto
cd /path/to/Perceiver_project

# Analizza exp6_fourier_permuted (il migliore!)
python visualize_attention.py --logs_dir logs --create_evolution --colormap jet
```

### **Esempio 2: Confronto RGB-only vs Fourier PE**
```bash
# Prima analizza exp3B (RGB-only)
python visualize_attention.py --logs_dir logs --epochs 10 50 120 --colormap viridis

# I risultati saranno in:
# logs/exp3B_rgb_only/attention_maps/comprehensive_epoch_X_analysis.png
# logs/exp3A_fourier_control/attention_maps/comprehensive_epoch_X_analysis.png
```

### **Esempio 3: Evoluzione Best Performers**
```bash
# Analizza evoluzione dei 2 migliori esperimenti
python visualize_attention.py --logs_dir logs --create_evolution --epochs 1 11 21 31 41 51 61 71 81 91 101 111 120 --colormap plasma
```

---

## 📊 **TIPI DI VISUALIZZAZIONI GENERATE**

### **1. Comprehensive Analysis (per ogni epoch):**
- **Original Image**: Immagine originale CIFAR-10
- **Attention Heatmap**: Mappa attention pura 
- **Attention Overlay**: Overlay attention su immagine
- **Statistics Panel**: Statistiche dettagliate
- **Distribution Plot**: Distribuzione valori attention
- **Attention Peaks**: Regioni attention più forte
- **Spatial Analysis**: Analisi spaziale row/column
- **Entropy Analysis**: Entropia attention per row
- **Colormap Comparison**: Confronto diversi colormaps

### **2. Evolution Analysis:**
- **Attention Maps Progression**: Come evolve l'attention
- **Overlay Progression**: Come cambia focus nel tempo  
- **Statistics Per Epoch**: Metriche attention per epoch

---

## 🔧 **PARAMETRI AVANZATI**

### **Colormap Options:**
- `jet` - Classico blu→rosso (default)
- `viridis` - Verde→viola (perceptually uniform)
- `plasma` - Viola→giallo (high contrast)
- `inferno` - Nero→giallo (heat style)
- `hot` - Nero→bianco (thermal)
- `cool` - Ciano→magenta (cold style)
- `seismic` - Blu→bianco→rosso (diverging)

### **Alpha Blending:**
- `0.0` - Solo immagine originale
- `0.3` - Overlay leggero
- `0.6` - Bilanciato (default)
- `0.8` - Attention dominante
- `1.0` - Solo attention map

---

## 📈 **INTERPRETAZIONE RISULTATI**

### **Cosa Cercare Nelle Visualizzazioni:**

#### **1. Attention Distribution:**
- **Peaked**: Attention concentrata (buon focus)
- **Uniform**: Attention dispersa (poco focus)
- **Bimodal**: Due zone principali attention

#### **2. Spatial Patterns:**
- **Center-focused**: Attention al centro immagine
- **Edge-focused**: Attention sui bordi
- **Object-aligned**: Attention sugli oggetti

#### **3. Evolution Trends:**
- **Early epochs**: Attention sparse, rumorosa
- **Mid training**: Graduale strutturazione
- **Late epochs**: Pattern raffinati, task-specific

#### **4. Statistical Metrics:**
- **High entropy**: Attention distribuita
- **Low entropy**: Attention concentrata  
- **High sparsity**: Molte regioni ignorate
- **High peak ratio**: Focus su poche regioni

---

## 🎯 **ANALISI SUGGERITE PER I TUOI ESPERIMENTI**

### **1. Confronto PE Types:**
```bash
# Confronta Fourier vs Learned PE su dati permutati
python visualize_attention.py --logs_dir logs --create_evolution --epochs 1 21 41 61 81 101 120 --colormap viridis
```
**Focus:** 
- `exp2_learned_pe_permuted` vs `exp6_fourier_permuted`
- Come differiscono i pattern attention?

### **2. Impatto Positional Encoding:**
```bash
# Confronta con vs senza PE
python visualize_attention.py --logs_dir logs --epochs 20 60 120 --colormap jet
```
**Focus:**
- `exp3A_fourier_control` vs `exp3B_rgb_only`
- PE crea pattern più strutturati?

### **3. Weight Sharing Analysis:**
```bash
# Confronta con vs senza weight sharing
python visualize_attention.py --logs_dir logs --create_evolution --colormap plasma
```
**Focus:**
- `exp4A_weight_sharing_control` vs `exp4B_no_weight_sharing`
- No weight sharing → pattern più complessi?

### **4. Permutation Effect:**
```bash
# Analizza effetto permutazioni
python visualize_attention.py --logs_dir logs --create_evolution --epochs 1 10 21 31 41 51 61 71 81 91 101 111 120 --colormap inferno
```
**Focus:**
- `exp1_baseline_fourier` vs `exp6_fourier_permuted`
- Permutazioni cambiano strategia attention?

---

## 📁 **DOVE TROVARE I RISULTATI**

Dopo aver eseguito lo script, trovi le visualizzazioni in:
```
logs/
├── exp1_baseline_fourier/attention_maps/
│   ├── comprehensive_epoch_1_analysis.png
│   ├── comprehensive_epoch_10_analysis.png
│   ├── ...
│   └── attention_evolution_analysis.png
├── exp2_learned_pe_permuted/attention_maps/
│   ├── comprehensive_epoch_1_analysis.png
│   └── ...
├── exp3A_fourier_control/attention_maps/
├── exp3B_rgb_only/attention_maps/
├── exp4A_weight_sharing_control/attention_maps/
├── exp4B_no_weight_sharing/attention_maps/
└── exp6_fourier_permuted/attention_maps/
```

---

## 🚀 **COMANDO COMPLETO SUGGERITO**

**Per analisi comprensiva di tutti i tuoi esperimenti:**
```bash
python visualize_attention.py \
    --logs_dir logs \
    --create_evolution \
    --epochs 1 11 21 31 41 51 61 71 81 91 101 111 120 \
    --colormap viridis \
    --alpha 0.6
```

Questo comando:
- ✅ Processa tutti gli esperimenti
- ✅ Crea analisi evoluzione temporale  
- ✅ Copre tutto il training (ogni 10 epochs)
- ✅ Usa colormap perceptually uniform
- ✅ Alpha bilanciato per overlay chiari

**Tempo stimato:** ~5-10 minuti per processare tutti gli esperimenti
**Output:** ~70 file di visualizzazione ad alta risoluzione
