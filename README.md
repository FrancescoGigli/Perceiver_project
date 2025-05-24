# 🧠 Perceiver Experiments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Un progetto di ricerca completo per sperimentare con l'architettura **Perceiver** su diversi dataset, con strumenti avanzati di visualizzazione e analisi. Il Perceiver è un'architettura neurale rivoluzionaria che utilizza meccanismi di attenzione per processare input di struttura e dimensione arbitraria, rendendolo ideale per modalità diverse come immagini, audio, video e nuvole di punti 3D.

## 📋 Indice

- [🚀 Quick Start](#-quick-start)
- [🏗️ Architettura e Funzionamento del Perceiver](#️-architettura-e-funzionamento-del-perceiver)
- [⚙️ Setup e Installazione](#️-setup-e-installazione)
- [🧪 Esperimenti](#-esperimenti)
- [📊 Strumenti di Visualizzazione Avanzati](#-strumenti-di-visualizzazione-avanzati)
- [📈 Risultati Attesi](#-risultati-attesi)
- [🔧 Parametri Chiave](#-parametri-chiave)
- [💡 Utilizzo Avanzato](#-utilizzo-avanzato)
- [📦 Dipendenze](#-dipendenze)
- [💡 Tips e Best Practices](#-tips-e-best-practices)
- [❓ FAQ](#-faq)
- [🤝 Contribuire](#-contribuire)
- [📄 Licenza](#-licenza)

## 🚀 Quick Start

Vuoi iniziare subito? Segui questi passi per un setup rapido:

```bash
# 1. Clona il repository
git clone https://github.com/yourusername/perceiver_project.git
cd perceiver_project

# 2. Setup ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Esegui un esperimento di prova su CIFAR-10
python train.py --experiment_name quick_test --dataset cifar10 --epochs 5 --save_attention_maps

# 5. Visualizza i risultati
python visualize_results.py --output_dir logs/quick_test
python visualize_attention.py --logs_dir logs/quick_test
```

## 🏗️ Architettura e Funzionamento del Perceiver

Il **Perceiver** rappresenta una rivoluzione nell'elaborazione di dati multimodali, risolvendo il problema fondamentale della scalabilità dei Transformer tradizionali.

### 🎯 Il Problema dei Transformer Classici

I Transformer tradizionali soffrono di una limitazione critica: la **complessità quadratica** rispetto alla lunghezza della sequenza. Per un'immagine 224×224 pixels, questo significa gestire ~50,000 token, rendendo il processo computazionalmente proibitivo.

**Complessità Transformer standard**: `O(N²)` dove N è il numero di input token

### 💡 L'Innovazione del Perceiver

Il Perceiver risolve elegantemente questo problema introducendo un **bottleneck latente** che disaccoppia la complessità dai dati di input:

**Complessità Perceiver**: `O(M×N + M²)` dove M << N
- M = numero di latenti (fisso, ~100-500)  
- N = dimensione input (variabile, migliaia-milioni)

### 🔧 Componenti Architetturali

#### 1. 🧠 Latent Array (Array Latente)

Il cuore del Perceiver è un **set fisso di vettori latenti** che funzionano come una "memoria" del modello:

```python
# Esempio: 128 latenti di dimensione 512
latent_array = torch.randn(128, 512)  # [num_latents, latent_dim]
```

**Caratteristiche**:
- **Dimensione fissa**: Indipendente dalla dimensione dell'input
- **Parametri appresi**: Inizializzati casualmente e ottimizzati durante il training
- **Ruolo**: Bottleneck computazionale che "riassume" l'informazione dell'input

#### 2. 🔄 Cross-Attention (Attenzione Incrociata)

Il meccanismo che permette ai latenti di "estrarre" informazione dall'input:

```python
# Query: latenti (cosa vogliamo sapere)
# Key/Value: input + positional encoding (da cosa impariamo)
output = cross_attention(
    query=latent_array,      # [M, D]  
    key=input_with_pos,      # [N, D]
    value=input_with_pos     # [N, D]
)
# Result: [M, D] - stessa dimensione dei latenti!
```

**Funzionamento**:
1. **Query**: I latenti "chiedono" informazioni specifiche
2. **Key/Value**: L'input risponde con le informazioni rilevanti
3. **Attention**: Il modello impara automaticamente quali parti dell'input sono importanti per ogni latente

#### 3. 🔁 Self-Attention (Auto-Attenzione)

Dopo aver acquisito informazione dall'input, i latenti comunicano tra loro:

```python
# I latenti si scambiano informazioni
refined_latents = self_attention(
    query=latents,
    key=latents, 
    value=latents
)
```

**Scopo**: Permettere l'elaborazione e l'integrazione delle informazioni estratte.

#### 4. 📍 Positional Encoding

Fondamentale per preservare l'informazione spaziale/temporale:

##### Fourier Positional Encoding (Default)
```python
# Per immagini 2D
pos_enc = fourier_encode(positions, num_bands=64, max_freq=32.0)
```

**Come funziona**:
- Converte coordinate (x,y) in features ad alta dimensionalità
- Usa funzioni trigonometriche: `[sin(2πf·x), cos(2πf·x)]` per varie frequenze f
- Permette al modello di "localizzare" ogni pixel

##### Learned Positional Encoding (Alternativa)
```python
# Embedding appresi durante il training
pos_enc = learned_embedding[position_indices]
```

### 🔄 Flusso di Elaborazione Completo

Ecco come funziona il Perceiver step-by-step:

```
1. INPUT PREPROCESSING
   📸 Immagine 32×32×3 → Flatten → 3072 pixel
   📍 Aggiungi positional encoding → 3072×(3+encoding_dim)

2. CROSS-ATTENTION (Input → Latenti)
   🧠 128 latenti + 3072 input tokens → Cross-Attention → 128 latenti aggiornati
   
3. SELF-ATTENTION (Latenti ↔ Latenti)  
   🔁 128 latenti → Self-Attention Blocks → 128 latenti elaborati
   
4. OUTPUT PROCESSING
   📊 128 latenti → Pool → 1 vettore → Classification Head → 10 classi
```

### 🎯 Vantaggi Chiave

#### ✅ Scalabilità
- **Costo fisso**: O(M²) indipendente dalla dimensione input
- **Applicabilità universale**: Stessa architettura per immagini, audio, video, 3D

#### ✅ Flessibilità Modale  
- **Input arbitrari**: Qualsiasi tipo di dato può essere "flattened" e processato
- **Dimensioni variabili**: Funziona con input di dimensioni diverse

#### ✅ Efficienza Computazionale
- **Memory footprint ridotto**: Solo M latenti in memoria durante self-attention
- **Parallelizzazione**: Cross-attention facilmente parallelizzabile

### 🧮 Analisi Matematica

Per comprendere meglio l'efficienza:

#### Transformer Standard (es. ViT)
```
Input: 32×32 = 1024 token
Self-Attention: O(1024²) = ~1M operazioni
Memory: O(1024²) = ~1M elementi
```

#### Perceiver
```
Input: 32×32 = 1024 token
Latenti: 128

Cross-Attention: O(128×1024) = ~128K operazioni  
Self-Attention: O(128²) = ~16K operazioni
Total: ~144K operazioni (7x più efficiente!)
```

### 🎨 Interpretazione delle Attention Maps

Le mappe di attenzione nel Perceiver rivelano insights unici:

#### Cross-Attention Maps
- **Cosa guarda ogni latente**: Quali regioni dell'input sono importanti
- **Specializzazione**: Latenti diversi si focalizzano su caratteristiche diverse
- **Evoluzione**: Come cambia l'attenzione durante il training

#### Self-Attention Maps  
- **Comunicazione tra latenti**: Quali latenti si influenzano a vicenda
- **Gerarchie emergenti**: Pattern di dipendenza tra diversi tipi di informazione

### 🔬 Intuizione Biologica

Il Perceiver può essere visto come un modello della **percezione umana**:

1. **Retina** (Input): Milioni di fotorecettori catturano luce
2. **Nervo ottico** (Cross-Attention): Bottleneck che comprime informazione
3. **Corteccia visiva** (Self-Attention): Elaborazione e interpretazione  
4. **Riconoscimento** (Output): Classificazione degli oggetti

Questa analogia spiega perché il Perceiver funziona così bene: rispecchia come il cervello processa informazioni sensoriali complesse!

## ⚙️ Setup e Installazione

### 📋 Prerequisiti

- **Python**: 3.8 o superiore
- **CUDA**: Opzionale, per accelerazione GPU
- **RAM**: Minimo 8GB, raccomandati 16GB
- **Storage**: Almeno 5GB di spazio libero

### 🛠️ Installazione Dettagliata

1. **Clona il repository**:
```bash
git clone https://github.com/yourusername/perceiver_project.git
cd perceiver_project
```

2. **Crea ambiente virtuale**:
```bash
# Con venv
python -m venv perceiver_env
source perceiver_env/bin/activate  # Linux/Mac
# perceiver_env\Scripts\activate   # Windows

# Oppure con conda
conda create -n perceiver_env python=3.9
conda activate perceiver_env
```

3. **Installa dipendenze**:
```bash
pip install -r requirements.txt

# Per funzionalità avanzate (opzionale)
pip install plotly scipy scikit-learn tensorboard
```

4. **Verifica installazione**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import matplotlib; print('Matplotlib OK')"
```

5. **Prepara directories**:
```bash
mkdir -p data logs
```

## 🧪 Esperimenti

Il progetto include tre esperimenti completi per esplorare diverse capacità del Perceiver:

### 🎯 Esperimento 1: Baseline con Fourier Features

**Obiettivo**: Stabilire un baseline solido usando codifica posizionale fissa.

#### 📊 Configurazione
| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| Dataset | CIFAR-10 | 32×32 immagini a colori, 10 classi |
| Latents | 96 | Numero di vettori latenti |
| Latent Dim | 384 | Dimensionalità di ogni latente |
| Cross-Attention | 4 stage | Iterazioni input→latents |
| Transformer Blocks | 4 | Layer di self-attention |
| Attention Heads | 3 | Teste di attenzione parallele |
| Fourier Bands | 64 | Frequenze per PE |
| Learning Rate | 0.004 | Tasso di apprendimento |
| Epochs | 120 | Epoche di training |

#### 🚀 Comando di Esecuzione
```bash
python train.py \
    --experiment_name perceiver_cifar10_fourier \
    --dataset cifar10 \
    --data_dir ./data \
    --cifar10_fourier_bands 64 \
    --cifar10_max_freq 32.0 \
    --num_latents 96 \
    --latent_dim 384 \
    --num_cross_attend_stages 4 \
    --num_transformer_blocks 4 \
    --num_heads 3 \
    --dropout 0.2 \
    --output_pooling mean \
    --optimizer lamb \
    --lr 0.004 \
    --scheduler multistep \
    --epochs 120 \
    --batch_size_cifar10 64 \
    --num_workers 4 \
    --save_attention_maps \
    --use_tensorboard
```

#### 🎯 Risultati Attesi
- **Accuratezza**: 85-90% su test set
- **Convergenza**: Intorno all'epoca 80-100
- **Pattern di Attenzione**: Focus su oggetti principali

---

### 🧠 Esperimento 2: Positional Encoding Learnable

**Obiettivo**: Valutare l'impatto dell'apprendimento della codifica posizionale.

#### 📊 Configurazione
| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| PE Type | Learnable | Embedding posizionali apprese |
| Pixel Permutation | ✅ Seed 42 | Test di robustezza spaziale |
| Cross-Attention | 1 stage | Ridotto per compensare complessità |
| Attention Save | Ogni 10 epoche | Per analisi evoluzione |

#### 🚀 Comando di Esecuzione
```bash
python train.py \
    --experiment_name perceiver_cifar10_permuted_learned_pe \
    --dataset cifar10 \
    --data_dir ./data \
    --use_learned_pe \
    --permute_pixels \
    --permute_pixels_seed 42 \
    --num_latents 96 \
    --latent_dim 384 \
    --num_cross_attend_stages 1 \
    --num_transformer_blocks 4 \
    --num_heads 3 \
    --dropout 0.2 \
    --output_pooling mean \
    --optimizer lamb \
    --lr 0.004 \
    --scheduler multistep \
    --epochs 120 \
    --batch_size_cifar10 64 \
    --num_workers 4 \
    --save_attention_maps \
    --attention_save_interval 10 \
    --save_metrics \
    --use_tensorboard
```

#### 🎯 Risultati Attesi
- **Adattabilità**: Capacità di apprendere pattern spaziali disrutti
- **Performance**: Possibile miglioramento del 2-5%
- **Convergenza**: Più lenta inizialmente, migliore a lungo termine

---

### 🌐 Esperimento 3: Point Clouds 3D (ModelNet40)

**Obiettivo**: Dimostrare la versatilità su dati 3D non strutturati.

#### 📊 Configurazione
| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| Dataset | ModelNet40 | Nuvole di punti 3D, 40 categorie |
| Points | 2048 | Punti per nuvola |
| Coordinates | (x,y,z) | Coordinate 3D |
| Fourier Bands | 64 | Per coordinate 3D |
| Max Frequency | 1120.0 | Adattata per range 3D |
| Latents | 128 | Aumentati per complessità |
| Latent Dim | 512 | Dimensionalità maggiore |
| Cross-Attention | 2 stage | Bilanciamento efficienza/performance |
| Transformer Blocks | 6 | Elaborazione profonda |
| Attention Heads | 8 | Parallelismo maggiore |

#### 🚀 Comando di Esecuzione
```bash
python train.py \
    --experiment_name perceiver_modelnet40_fourier \
    --dataset modelnet40 \
    --data_dir ./data \
    --modelnet40_num_points 2048 \
    --modelnet40_fourier_bands 64 \
    --modelnet40_max_freq 1120.0 \
    --num_latents 128 \
    --latent_dim 512 \
    --num_cross_attend_stages 2 \
    --num_transformer_blocks 6 \
    --num_heads 8 \
    --dropout 0.1 \
    --output_pooling mean \
    --optimizer lamb \
    --lr 0.001 \
    --scheduler multistep \
    --epochs 120 \
    --batch_size_modelnet40 8 \
    --num_workers 0 \
    --save_attention_maps \
    --attention_save_interval 10 \
    --save_metrics \
    --use_tensorboard
```

#### 🎯 Risultati Attesi
- **Accuratezza**: 85-92% su ModelNet40
- **Robustezza**: Invarianza a rotazioni e traslazioni
- **Attenzione**: Focus su caratteristiche geometriche distintive

## 📊 Strumenti di Visualizzazione Avanzati

Gli strumenti di visualizzazione rappresentano il cuore dell'analisi, offrendo insights profondi sia sui pattern di apprendimento che sui meccanismi di attenzione.

### 🔍 Visualizzazione Mappe di Attenzione

#### 🌟 Caratteristiche Principali

- **🎨 Multi-Colormap**: 7 palette colore professionali
- **📈 Analisi Statistica**: Media, deviazione, entropia, picchi
- **🔄 Tracking Evoluzione**: Cambiamenti attraverso le epoche
- **🎯 Robustezza**: Gestione automatica di formati tensor diversi

#### 💻 Utilizzo Base
```bash
# Analisi automatica di tutte le mappe disponibili
python visualize_attention.py

# Personalizzazione avanzata
python visualize_attention.py \
    --colormap viridis \
    --alpha 0.7 \
    --create_evolution \
    --logs_dir logs/my_experiment
```

#### 🎨 Colormaps Disponibili

| Colormap | Uso Ottimale | Caratteristiche |
|----------|--------------|-----------------|
| `viridis` | Analisi generale | Uniforme percettivamente |
| `plasma` | Evidenziare picchi | Alto contrasto |
| `inferno` | Dati con range ampio | Buona per stampa |
| `jet` | Visualizzazione classica | Massimo contrasto |
| `hot` | Mappe di calore | Intuizione temperature |
| `cool` | Analisi dettagliata | Toni freddi rilassanti |
| `seismic` | Dati bipolari | Rosso-blu divergente |

#### 📁 Output Generati

- **`comprehensive_epoch_X_analysis.png`**: 
  - 6 pannelli di analisi completa
  - Immagine originale + heatmap + overlay
  - Statistiche + distribuzione + picchi

- **`attention_evolution_analysis.png`**:
  - Evoluzione side-by-side
  - Comparazione tra epoche
  - Trend di apprendimento visibile

### 📈 Visualizzazione Risultati Training

#### 🎯 Dashboard Completo

Il dashboard fornisce una vista a 360° del processo di training:

##### 🏆 Pannello Principale (Row 1)
- **Curve di Accuratezza**: Train vs Test con trend polinomiali
- **Curve di Loss**: Scala auto-logaritmica, annotazioni punti chiave
- **Statistiche Live**: Metriche in tempo reale

##### 🔬 Analisi Avanzata (Row 2)
- **Gap Analysis**: Rilevamento overfitting/underfitting automatico
- **Convergenza**: Statistiche rolling con bande di confidenza
- **Performance Metrics**: Stabilità, generalizzazione, miglioramento

##### 📊 Summary (Row 3)
- **Tabella Riassuntiva**: Best accuracy, min loss, epoche, gap analysis

#### 💻 Utilizzo
```bash
# Analisi standard
python visualize_results.py --output_dir logs/my_experiment

# Analisi completa con attenzione
python visualize_results.py \
    --output_dir logs/my_experiment \
    --experiment_name "Esperimento_Principale" \
    --epochs 1 10 20 30 40 50 \
    --enhanced_attention
```

#### 📊 Metriche Calcolate

| Metrica | Formula | Interpretazione |
|---------|---------|-----------------|
| **Stability** | `1 - std(last_10_epochs)` | Quanto è stabile il training |
| **Convergence** | `1 - abs(final_loss - prev_loss)` | Grado di convergenza |
| **Generalization** | `1 - abs(train_acc - test_acc)` | Capacità di generalizzazione |
| **Improvement** | `final_acc - initial_acc` | Miglioramento totale |
| **Overfitting Score** | `max(0, train_acc - test_acc)` | Livello di overfitting |

### 🎮 Demo Interattivo

Abbiamo aggiunto uno script demo per esplorare le visualizzazioni in modo interattivo:

```bash
# Demo completo delle visualizzazioni
python demo_visualizations.py

# Demo con dataset personalizzato
python demo_visualizations.py --logs_dir logs/my_experiment
```

#### 🌟 Funzionalità Demo
- **Visualizzazione Real-time**: Aggiornamento automatico delle mappe
- **Confronti Interattivi**: Side-by-side tra epoche diverse
- **Esportazione Avanzata**: Salvataggio in formati multipli (PNG, SVG, PDF)
- **Analisi Parametrica**: Variazione dinamica di parametri di visualizzazione

### 🖼️ Esempi di Visualizzazioni

#### 📸 Screenshot Tipici

**Attention Heatmap Analysis**:
- Top-left: Immagine originale CIFAR-10
- Top-center: Heatmap di attenzione sovrapposta
- Top-right: Regioni di attenzione top-5%
- Bottom-left: Statistiche per riga (target positions)
- Bottom-center: Statistiche per colonna (source positions)
- Bottom-right: Analisi entropia e sparsità

**Training Dashboard**:
- Curve smooth con trend lines
- Annotazioni automatiche dei punti salienti
- Color coding per stati (converged/converging/diverging)
- Tabelle formattate professionalmente

## 📈 Risultati Attesi

### 🏆 Performance Benchmark

| Esperimento | Dataset | Accuratezza Attesa | Tempo Training* | GPU Memory** |
|-------------|---------|-------------------|-----------------|--------------|
| Fourier PE | CIFAR-10 | 87-91% | 2-3 ore | 4-6 GB |
| Learned PE | CIFAR-10 | 85-93% | 3-4 ore | 4-6 GB |
| 3D Point Clouds | ModelNet40 | 88-92% | 4-6 ore | 6-8 GB |

*\*Con GPU RTX 3080/4070*  
*\*\*Batch size ottimale*

### 📊 Convergenza Tipica

- **Epoche 1-20**: Apprendimento iniziale rapido (40-60% accuracy)
- **Epoche 20-60**: Miglioramento steady (60-80% accuracy)  
- **Epoche 60-100**: Fine-tuning e stabilizzazione (80-90% accuracy)
- **Epoche 100+**: Convergenza finale (plateau intorno al massimo)

### 🎯 Pattern di Attenzione Attesi

#### CIFAR-10
- **Oggetti centrali**: Focus su airplane wings, car body, etc.
- **Bordi distintivi**: Contorni e edge features
- **Evoluzione**: Da pattern globali a dettagli specifici

#### ModelNet40  
- **Caratteristiche geometriche**: Vertici, spigoli, superfici curve
- **Simmetrie**: Riconoscimento di pattern simmetrici
- **Invarianza**: Robustezza a rotazioni

## 🔧 Parametri Chiave

### 🧠 Architettura

| Parametro | Descrizione | Range Tipico | Impatto |
|-----------|-------------|--------------|---------|
| `--num_latents` | Numero vettori latenti | 32-512 | Capacità modello |
| `--latent_dim` | Dimensione ogni latente | 128-1024 | Espressività |
| `--num_cross_attend_stages` | Stage cross-attention | 1-8 | Input processing |
| `--num_transformer_blocks` | Blocchi self-attention | 2-12 | Profondità elaborazione |
| `--num_heads` | Teste attenzione | 1-16 | Parallelismo |

### 📊 Training

| Parametro | Descrizione | Valori Consigliati | Note |
|-----------|-------------|-------------------|------|
| `--optimizer` | Algoritmo ottimizzazione | `lamb`, `adamw` | LAMB per batch grandi |
| `--lr` | Learning rate | 0.0001-0.01 | Dipende da optimizer |
| `--scheduler` | LR scheduler | `multistep`, `cosine` | Per convergenza stabile |
| `--dropout` | Dropout rate | 0.0-0.3 | Regularizzazione |
| `--batch_size` | Dimensione batch | 16-128 | Limitato da GPU memory |

### 🎨 Codifica Posizionale

| Parametro | Tipo | Descrizione | Quando Usare |
|-----------|------|-------------|--------------|
| `--use_learned_pe` | Learned | PE apprese dal modello | Dati con pattern spaziali complessi |
| `--fourier_bands` | Fourier | Numero bande frequenza | Default per immagini |
| `--max_freq` | Fourier | Frequenza massima | Adattare alla risoluzione |

### 📈 Monitoraggio

| Parametro | Descrizione | Raccomandazione |
|-----------|-------------|-----------------|
| `--save_attention_maps` | Salva mappe attenzione | Sempre abilitare |
| `--attention_save_interval` | Intervallo salvataggio | 10 epoche |
| `--save_metrics` | Salva metriche training | Sempre abilitare |
| `--use_tensorboard` | Logging TensorBoard | Per monitoring real-time |

## 💡 Utilizzo Avanzato

### 🎛️ Learning Rate Schedulers

#### Cosine Annealing (Raccomandato)
```bash
python train.py \
    --scheduler cosine \
    --eta_min_cosine 0.00001 \
    --T_max_cosine 100 \
    [altri parametri]
```
**Vantaggi**: Convergenza smooth, evita plateau locali

#### MultiStep (Classico)
```bash
python train.py \
    --scheduler multistep \
    --lr_milestones 30 60 90 \
    --lr_gamma 0.1 \
    [altri parametri]
```
**Vantaggi**: Controllo preciso, boost a epoche specifiche

#### Step Decay
```bash
python train.py \
    --scheduler step \
    --lr_step_size 30 \
    --lr_gamma 0.5 \
    [altri parametri]
```
**Vantaggi**: Semplice, prevedibile

### ⚙️ Strategie Weight Sharing

#### Condiviso (Default)
- **Pro**: Meno parametri, training più veloce
- **Contro**: Capacità limitata

#### Non Condiviso
```bash
python train.py --no_weight_sharing [altri parametri]
```
- **Pro**: Maggiore espressività, performance migliori
- **Contro**: Più parametri, training più lento

### 🎯 Output Pooling

#### Mean Pooling (Default)
```bash
--output_pooling mean
```
**Uso**: General purpose, stabile

#### CLS Token
```bash
--output_pooling cls
```
**Uso**: Quando serve rappresentazione globale specifica

### 🚀 Ottimizzazioni Performance

#### Batch Size Dinamico
```bash
# Per GPU con memoria limitata
--batch_size_cifar10 32 --batch_size_modelnet40 4

# Per GPU potenti
--batch_size_cifar10 128 --batch_size_modelnet40 16
```

#### Mixed Precision Training
```bash
# Aggiungere al codice train.py
--use_amp  # Se implementato
```

#### Gradient Accumulation
```bash
# Simulare batch size maggiori
--gradient_accumulation_steps 4
```

## 📦 Dipendenze

### 🛠️ Core Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.3.0
tqdm>=4.62.0
```

###
