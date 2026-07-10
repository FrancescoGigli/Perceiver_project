/* ════════════════════════════════════════════════════════════════════
   images.js — mappa card-id → array di {src, caption, credit}
   Le immagini sono in interactive_trainer/img/. Il renderer di app.js
   inserisce un blocco <figure> in cima al body della card se trova match.

   Regola: ogni immagine DEVE essere allineata al titolo della card.
   Niente immagini su quiz/review se non specificamente pertinenti.
   ════════════════════════════════════════════════════════════════════ */
const CARD_IMAGES = {

  // ═════════════════════════════════════════════════════════════════
  //   M0 — Intro & Motivazione (Perceiver originale)
  // ═════════════════════════════════════════════════════════════════
  'm0-c1': [{                                       // Frammentazione architetturale
    src: 'img/fig2_modalities.png',
    caption: 'Il Perceiver applicato a 4 modalità: immagini, audio, video, point cloud 3D. Stessa architettura per tutte.',
    credit: 'Perceiver paper (Jaegle et al., ICML 2021) — Fig. 2'
  }],
  'm0-c7': [{                                       // CNN vs Transformer vs Perceiver
    src: 'img/docx_cnn_pipeline.jpg',
    caption: 'Pipeline CNN: pixel → filtri locali + pooling → feature maps gerarchiche. Inductive bias: località + invarianza traslazionale.',
    credit: 'Dispensa interna'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M2 — Architettura Perceiver (originale, NON Perceiver IO)
  // ═════════════════════════════════════════════════════════════════
  'm2-c1': [{                                       // Panoramica: due componenti
    src: 'img/fig1_architecture.png',
    caption: 'Architettura del Perceiver originale: input array → cross-attention → latent transformer (L self-attn), iterato T volte con weight sharing. Output via average pooling.',
    credit: 'Perceiver paper (Jaegle et al., ICML 2021) — Fig. 1'
  }],
  'm2-c2': [{                                       // Pipeline del forward pass
    src: 'img/fig1_architecture_diagram.png',
    caption: 'Diagramma dettagliato del Perceiver: prima cross-attention separata, successive condividono i pesi (weight sharing).',
    credit: 'Perceiver paper — Fig. 1 (variante)'
  }],
  'm2-c3': [{                                       // Input Processing: byte array + flatten
    src: 'img/docx_byte_latent_arrays.png',
    caption: 'Input array (M×C, byte/pixel) e Latent array (N×D, N≪M): il bottleneck informativo del Perceiver.',
    credit: 'Dispensa interna'
  }],
  'm2-c5': [
    {                                                // Cross-Attention formule
      src: 'img/docx_cross_attention_steps.png',
      caption: 'Pipeline della cross-attention: Q dai latenti, K e V dall\'input. Matrice di attenzione N×M.',
      credit: 'Dispensa interna'
    },
    {
      src: 'img/docx_qkv_projections.png',
      caption: 'Proiezioni Q = z·W_Q (dai latenti), K = x·W_K, V = x·W_V (dall\'input). Le proiezioni mappano in d_k = d_v.',
      credit: 'Dispensa interna'
    }
  ],

  // ═════════════════════════════════════════════════════════════════
  //   M3 — Fourier PE & Forward Pass
  // ═════════════════════════════════════════════════════════════════
  'm3-c4': [{                                       // Esempio numerico pixel (112, 56)
    src: 'img/fig4_crop_coords.png',
    caption: 'Visualizzazione delle Fourier features sulle coordinate dei pixel: ogni posizione viene mappata in un vettore di sin/cos a frequenze diverse.',
    credit: 'Perceiver paper — Fig. 4 (concept)'
  }],
  'm3-c10': [{                                      // Forward A5 cross-attention zoom
    src: 'img/docx_cross_attention_steps.png',
    caption: 'Forward A5: zoom completo del blocco cross-attention (Pre-LN → Q/K/V → Scaled dot → Softmax → Comb → Proj → Residual).',
    credit: 'Dispensa interna'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M5 — Esperimenti (paper Perceiver + nostri)
  // ═════════════════════════════════════════════════════════════════
  'm5-c1': [{                                       // ImageNet del paper
    src: 'img/fig3_attention_maps.png',
    caption: 'Attention maps del Perceiver su ImageNet (paper originale, Fig.3): visualizzazione di dove i latenti "guardano" nell\'immagine.',
    credit: 'Perceiver paper (Jaegle et al., 2021) — Fig. 3'
  }],
  'm5-c5': [{                                       // Ablation paper: cross-attend + WS
    src: 'img/fig5_hyperparameter_sweeps.png',
    caption: 'Sweep degli iperparametri del Perceiver originale: numero di latenti N, iterazioni T, blocchi L per self-attn.',
    credit: 'Perceiver paper — Fig. 5'
  }],
  'm5-c7b': [{                                      // Perceiver vs paper: PE essenziale
    src: 'img/cifar10_chart.png',
    caption: 'Risultati CIFAR-10 dell\'ablation study sul PE: 72.02% con Fourier PE vs 61.34% senza PE (-10.68pp).',
    credit: 'Nostri esperimenti'
  }],
  'm5-c9b': [
    {                                                // Fourier vs Learned su permutato, WS
      src: 'img/accuracy_chart.png',
      caption: 'Riepilogo accuracy degli esperimenti Perceiver su CIFAR-10: Fourier vs Learned PE, input permutato e weight sharing.',
      credit: 'Nostri esperimenti'
    },
    {
      src: 'img/our_exp6_attention_evolution.png',
      caption: 'Attention map dell\'esperimento 6 (Fourier PE su input permutato): la struttura rimane grazie alle frequenze.',
      credit: 'Nostri esperimenti'
    }
  ],
  'm5-c10b': [
    {                                                // Riepilogo Perceiver originale vs paper
      src: 'img/modelnet40_chart.png',
      caption: 'Risultati ModelNet40: baseline 84.16% vs paper 85.7% (gap 1.54pp). Augmentation peggiora.',
      credit: 'Nostri esperimenti'
    },
    {
      src: 'img/convergence_chart.png',
      caption: 'Convergenza dei nostri esperimenti Perceiver: immagini e point cloud sono letti come controlli rispetto ai trend del paper.',
      credit: 'Nostri esperimenti'
    }
  ],
  'm5-c12': [{                                      // Encode-Process-Decode formula
    src: 'img/perceiver_io_fig2_architecture.png',
    caption: 'Encode-Process-Decode: il paradigma del Perceiver IO. Cross-attention encoder + processor self-attn + decoder cross-attn con output query.',
    credit: 'Perceiver IO paper (Jaegle et al., ICLR 2022) — Fig. 2'
  }],
  'm5-c13': [{                                      // Output Query Array PIO
    src: 'img/perceiver_io_fig3_output_queries.png',
    caption: 'Output Query Array: design per task (classification, optical flow, MLM, multimodal).',
    credit: 'Perceiver IO paper — Fig. 3'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M6 — PIO Decoder & Output Queries
  // ═════════════════════════════════════════════════════════════════
  'm6-c3': [{                                       // Read-Process-Write
    src: 'img/perceiver_io_fig2_architecture.png',
    caption: 'Encode-Process-Decode: il paradigma del Perceiver IO. Decoder cross-attention "scrive" l\'output dalle latenti.',
    credit: 'Perceiver IO paper — Fig. 2'
  }],
  'm6-c9': [{                                       // Output Query Array
    src: 'img/perceiver_io_fig3_output_queries.png',
    caption: 'Output query design per task: classification (1 query), optical flow (H×W), MLM (N_masked), multimodal (eterogenee).',
    credit: 'Perceiver IO paper — Fig. 3'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M7 — Esperimenti PIO
  // ═════════════════════════════════════════════════════════════════
  'm7-c1': [{                                       // GLUE setup byte-level
    src: 'img/glue_chart.png',
    caption: 'Performance sul benchmark GLUE: 8 task fine-tuned dal modello PIO byte-level pre-allenato MLM.',
    credit: 'Perceiver IO paper / Nostri esperimenti'
  }],
  'm7-c6': [{                                       // Optical Flow setup Sintel/KITTI
    src: 'img/optical_flow_eg.png',
    caption: 'Esempio di optical flow: movimento per-pixel come campo di vettori 2D. Loss EPE (L2). Confronto col SOTA RAFT.',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],
  'm7-c14': [{                                      // Tabella riassuntiva Table 5
    src: 'img/table5_cross_attends.png',
    caption: 'Tabella 5 — Numero di cross-attend per task (Encoder/Decoder) nel Perceiver IO.',
    credit: 'Perceiver IO paper — Table 5'
  }],
  'm7-c15a': [{                                     // PIO vs paper
    src: 'img/glue_chart.png',
    caption: 'Confronto dei nostri risultati linguistici con il riferimento del paper Perceiver IO: stesso paradigma byte-level, budget molto diverso.',
    credit: 'Perceiver IO paper / Nostri esperimenti'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M8 — Loss & Training PIO
  // ═════════════════════════════════════════════════════════════════
  'm8-c4': [{                                       // EPE per Optical Flow
    src: 'img/optical_flow_eg.png',
    caption: 'Optical flow per-pixel: campo di vettori 2D (Δx, Δy). EPE = ||pred − target||_2 calcolata pixel per pixel.',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],
  'm8-c14': [{                                      // Walkthrough Optical Flow + Autoencoding
    src: 'img/page7_attention_maps.png',
    caption: 'Attention maps multimodali del Perceiver IO: walkthrough delle pipeline Sintel/Kinetics.',
    credit: 'Perceiver IO paper — Fig. 7'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M9 — Transformer & ViT (App H + I)
  // ═════════════════════════════════════════════════════════════════
  'm9-c2': [{                                       // Architettura encoder-decoder N=6
    src: 'img/docx_transformer_compact.png',
    caption: 'Architettura del Transformer encoder-decoder: 6 layer encoder + 6 decoder (Vaswani et al., 2017).',
    credit: 'Attention is All You Need — Fig. 1'
  }],
  'm9-c5': [{                                       // Multi-Head h=8
    src: 'img/docx_transformer_encoder_highlight.png',
    caption: 'Encoder Transformer (evidenziato): multi-head self-attention (h=8 teste) + FFN, residual + LayerNorm.',
    credit: 'Dispensa interna'
  }],
  'm9-c7': [{                                       // Masked Self-Attention decoder
    src: 'img/docx_sdpa_mask.png',
    caption: 'Scaled dot-product attention con maschera causale: usata nel decoder per impedire ai token futuri di influenzare il presente.',
    credit: 'Dispensa interna'
  }],
  'm9-c8': [{                                       // Cross-attention encoder-decoder
    src: 'img/docx_transformer_decoder_highlight.png',
    caption: 'Decoder Transformer: masked self-attention + cross-attention con encoder (Q dal decoder, K/V dall\'encoder) + FFN.',
    credit: 'Dispensa interna'
  }],
  'm9-c11': [{                                      // ViT motivazione e pipeline
    src: 'img/docx_vit_architecture.png',
    caption: 'Vision Transformer (ViT): split immagine in patch 16×16 → linear embedding → CLS token + PE → encoder Transformer → MLP head.',
    credit: 'Dosovitskiy et al., 2021'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M10 — RNN / LSTM / GRU
  // ═════════════════════════════════════════════════════════════════
  'm10-c1': [{                                      // RNN loop nello stato nascosto
    src: 'img/rnn_architecture_original.png',
    caption: 'Architettura RNN: stato nascosto h_t che evolve nel tempo. Loop con condivisione dei pesi tra timestep.',
    credit: 'Dispensa interna'
  }],
  'm10-c3': [{                                      // Srotolamento e weight sharing
    src: 'img/rnn_unfold.svg',
    caption: 'Srotolamento (unrolling) di una RNN nel tempo: stessa cella, parametri W condivisi tra timestep.',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],
  'm10-c8': [{                                      // LSTM: cella di memoria
    src: 'img/lstm_architecture_original.png',
    caption: 'Cella LSTM: 4 gate (input, forget, output, candidate) + cell state c_t come "autostrada" per il gradiente.',
    credit: 'Dispensa interna'
  }],
  'm10-c13': [{                                     // GRU: 2 porte invece di 3
    src: 'img/gru_cell_type1.svg',
    caption: 'Cella GRU: 2 gate (reset r_t, update z_t). Architettura più semplice della LSTM ma performance comparabili.',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M11 — CNN & ResNet
  // ═════════════════════════════════════════════════════════════════
  'm11-c1': [{                                      // Input: immagine come matrice
    src: 'img/cnn_architecture_clean.jpg',
    caption: 'Architettura CNN classica: input (H×W×3) → conv + ReLU + pool ripetuti → flatten → FC layers → softmax.',
    credit: 'Dispensa interna'
  }],
  'm11-c2': [{                                      // Strato convoluzionale
    src: 'img/cnn_03_kernel_convolution.png',
    caption: 'Convoluzione: il kernel K×K scorre sull\'input applicando weight sharing. Parametri stride S, padding P.',
    credit: 'Dispensa interna'
  }],
  'm11-c3': [{                                      // Esempio numerico conv 5x5
    src: 'img/cnn_05_feature_maps.png',
    caption: 'Feature maps prodotte da un layer convoluzionale: ogni filtro produce un canale di output.',
    credit: 'Dispensa interna'
  }],
  'm11-c5': [{                                      // Pooling max e avg
    src: 'img/cnn_06_pooling_maxpool.png',
    caption: 'Max-pooling 2×2: riduce le dimensioni spaziali (H/2 × W/2) e fornisce invarianza traslazionale locale.',
    credit: 'Dispensa interna'
  }],
  'm11-c6': [{                                      // Gerarchia features
    src: 'img/cnn_feature_hierarchy_clean.jpeg',
    caption: 'Gerarchia di feature nelle CNN: layer 1 → edges, 2 → textures, 3 → parts (occhi, ruote), 4 → objects.',
    credit: 'Dispensa interna'
  }],
  'm11-c16': [{                                     // Bottleneck block ResNet-50
    src: 'img/docx_resnet50_arch.png',
    caption: 'Architettura ResNet-50: bottleneck block 1×1 → 3×3 → 1×1 con riduzione/espansione canali. Skip connection esterna.',
    credit: 'He et al., 2016'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M4 — Training (attivazioni, backward, optimizer)
  // ═════════════════════════════════════════════════════════════════
  'm4-c2': [{                                       // GELU: l'attivazione del Perceiver
    src: 'img/relu_gelu_plot.svg',
    caption: 'Confronto ReLU vs GELU: GELU è liscia (gradiente continuo) e ha valori non-zero anche per x<0 (no dying neurons).',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],

  // ═════════════════════════════════════════════════════════════════
  //   M13 — Approfondimenti (Softmax, LayerNorm, Attivazioni, Optimizer)
  // ═════════════════════════════════════════════════════════════════
  'm13-c9': [{                                      // Sigmoid/Tanh/ReLU/GELU tabella
    src: 'img/relu_gelu_plot.svg',
    caption: 'Plot delle attivazioni più comuni: ReLU (rosso, piecewise) vs GELU (blu, liscia). Domain, derivata e problemi a confronto.',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],
  'm13-c10': [{                                     // Perché GELU vince
    src: 'img/gelu_only.png',
    caption: 'GELU: x · Φ(x) ≈ x · σ(1.702·x). Self-gating + smooth gradient = standard nei Transformer e Perceiver.',
    credit: 'Wikimedia Commons (CC BY-SA)'
  }],
};

if (typeof window !== 'undefined') window.CARD_IMAGES = CARD_IMAGES;
