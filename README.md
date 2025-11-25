# 🤖 XT-404 Skynet : Wan 2.2 Sentinel Suite
### Cyberdyne Systems Corp. | Series T-800 | Model 101

<p align="center">
  <img src="https://img.shields.io/badge/Version-v3.4_Sentinel-red?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Architecture-Wan_2.2-blue?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/GGUF-Native_Support-green?style=for-the-badge" alt="GGUF Support">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License">
</p>

> *"The future is not set. There is no fate but what we make for ourselves."*

---

## ⚠️ CRITICAL SYSTEM DEPENDENCY / DÉPENDANCE CRITIQUE

> [!CAUTION]
> **INFILTRATION PROTOCOL (GGUF):**
> To utilize GGUF Quantized Models with this suite, the **ComfyUI-GGUF** engine is **REQUIRED**.
>
> 📥 **Download Engine:** `city96/ComfyUI-GGUF`
>
> *Without this engine, the Cyberdyne Model Hub will operate in Safetensors-only mode.*

---

## 🌍 NEURAL NET NAVIGATION / NAVIGATION DU RÉSEAU

### 🇺🇸 [ENGLISH DOCUMENTATION](#-english-documentation)
1. [Latest Intel (Changelog)](#-latest-intel-v34--v152-sentinel)
2. [Phase 1: Infiltration (Loaders)](#%EF%B8%8F-phase-1-infiltration-loaders)
3. [Phase 2: Neural Net Core (Samplers)](#-phase-2-neural-net-core-samplers-xt-404)
4. [Phase 3: Hardware Optimization](#-phase-3-hardware-optimization)
5. [Phase 4: Post-Processing & Tools](#%EF%B8%8F-phase-4-post-processing--tools)

### 🇫🇷 [DOCUMENTATION FRANÇAISE](#-documentation-française)
1. [Dernières Infos (Mise à jour)](#-dernières-infos-v34--v152-sentinel)
2. [Phase 1 : Infiltration (Chargement)](#%EF%B8%8F-phase-1--infiltration-chargement)
3. [Phase 2 : Cœur Neuronal (Samplers)](#-phase-2--cœur-neuronal-samplers-xt-404)
4. [Phase 3 : Optimisation Matérielle](#-phase-3--optimisation-matérielle)
5. [Phase 4 : Post-Production & Outils](#%EF%B8%8F-phase-4--post-production--outils)

---

# 🇺🇸 ENGLISH DOCUMENTATION

## 📡 Latest Intel (v3.4 / v15.2 Sentinel)

XT-404 Skynet is an elite engineering suite for ComfyUI, specifically architected for the Wan 2.2 video generation model.

### 🆕 System Status Update (v3.4):
*   **Protocol Update (Passive Sentinel):** The "Vector Amplification" (Signal Boost) has been **decommissioned**. The Sentinel now operates in **Passive Monitoring Mode** only. It analyzes CFG levels but no longer interferes with the signal, resolving "burn" issues in Chained/Refined workflows.
*   **Atomic Guard v14.2 (Stability):** Fixed a critical backend crash (`record_stream`) when decoding on CPU/Hybrid modes. Added robust async synchronization to prevent memory corruption.
*   **Vision Cache v8.1:** Enhanced hashing algorithm (Stride 16 + Byte Exactness) to eliminate collision risks on similar frames.
*   **Adaptive Bongmath v3:** The "Anti-Plastic" texture engine remains active, respecting dark scenes while expanding dynamic range.

---

## 🛡️ Phase 1: Infiltration (Loaders)

### 🤖 Cyberdyne Model Hub
**Class:** `CyberdyneModelHub`

A unified, intelligent loader that bridges the gap between Analog (Safetensors) and Quantized (GGUF) architectures. It specifically handles the Wan 2.2 Dual-Model requirement (High Context + Low Context).

| Parameter | Description |
| :--- | :--- |
| `model_high_name` | The main UNet model. Supports `.safetensors` AND `.gguf`. |
| `dtype_high` | Precision override (`fp16`, `bf16`, `fp8_e4m3fn`, etc.). |
| `model_low_name` | The secondary UNet model (Wan 2.2 requirement). |
| `enable_checksum` | Performs a SHA256 integrity scan (Security Protocol). |
| `offload_inactive` | **"Skynet Protocol":** Aggressively purges VRAM of unused models before loading new ones to prevent OOM. |

---

## 🧠 Phase 2: Neural Net Core (Samplers XT-404)

The "Sentinel" engine powers three specialized sampling nodes designed for chained workflows.

### 🔴 XT-404 Skynet 1 (Master)
**The Commander node.** Initializes generation and defines the global noise schedule.
*   **Prompt Authority:** Active signal monitoring (Passive Mode).
*   **Outputs:** Latent, Denoised Latent, Options (for chaining), Seed.

| Parameter | Description |
| :--- | :--- |
| `sampler_name` | Combo selection (e.g., `linear/euler`, `beta/dpmpp_2m`). |
| `cfg` | Guidance Scale. Monitored by Sentinel. |
| `bongmath` | Texture Engine. `True` = Film/Analog look. `False` = Digital/Smooth. |
| `sampler_mode` | Standard or Resample (injects fresh noise). |

### 🟡 XT-404 Skynet 2 (Chain)
**The Relay node.** Designed for split-sampling (e.g., first 50% on Master, next 30% on Chain).
*   **Logic:** Hides the Seed widget (uses internal deterministic inheritance).
*   **VRAM:** Dynamic unloading based on model type (Keep GGUF loaded / Unload FP16).

### 🟢 XT-404 Skynet 3 (Refiner)
**The Terminator node.** Finalizes the image structure.
*   **Configuration:** `steps_to_run` defaults to `-1` (finish the schedule).
*   **Focus:** High-frequency detail recovery.

---

## ⚡ Phase 3: Hardware Optimization

### 🚀 Wan Hardware Accelerator
**Class:** `Wan_Hardware_Accelerator`
Enables low-level PyTorch optimizations (TF32) for NVIDIA Ampere+ GPUs. Increases matrix multiplication speed.

### ✂️ Wan Attention Slicer (SDPA)
**Class:** `Wan_Attention_Slicer`
Manages the Attention mechanism.
*   **0 (Auto):** Activates Flash Attention (SDPA) for maximum speed.
*   **1-4:** Forces slicing to drastically reduce VRAM peaks (slower, but safer for <12GB cards).

### 🧩 Wan Hybrid VRAM Guard
**Class:** `Wan_Hybrid_VRAM_Guard`
Essential for VAE Decoding. Replaces the standard VAE Decode.
*   **Atomic Decoding:** Decodes 1 frame at a time.
*   **Tiling:** Forces spatial tiling (512px chunks).
*   **Async Offload:** Streams decoded data to CPU RAM immediately. **Zero VRAM Spikes.**

### 🍵 Wan TeaCache (Obsidian Engine)
**Class:** `Wan_TeaCache_Patch`
Implements caching to skip U-Net calculations if frame difference is minimal.
*   `rel_l1_threshold`: **0.15** (Recommended). Higher = Faster generation, lower quality.

---

## 🛠️ Phase 4: Post-Processing & Tools

### 💾 Wan Video Compressor (H.265)
**Class:** `Wan_Video_Compressor`
Encodes output to H.265 10-bit.
*   **Modes:** Web/Discord (<5MB target), Master (High Fidelity), Archival.
*   **Psy-RD:** Preserves grain texture without bloating file size.

### 🧹 Wan Cycle Terminator
**Class:** `Wan_Cycle_Terminator`
Surgical memory cleaning. Uses Windows API `EmptyWorkingSet` to flush Physical RAM + VRAM.
*   **Fun:** Displays Skynet/Terminator quotes in the console on activation.

### 📐 Resolution Savant & OneShot Cache
*   **Resolution Savant:** Resizes images ensuring dimensions are strictly divisible by 16 (Wan Requirement), using Lanczos (CPU) or Area (GPU).
*   **OneShot Cache:** Caches Text (CLIP) and Vision (I2V) encodings in Pinned Memory to prevent re-calculation.

---
---

# 🇫🇷 DOCUMENTATION FRANÇAISE

## 📡 Dernières Infos (v3.4 / v15.2 Sentinel)

XT-404 Skynet est une suite d'ingénierie d'élite pour ComfyUI, architecturée spécifiquement pour le modèle de génération vidéo Wan 2.2.

### 🆕 Mise à jour État Système (v3.4) :
*   **Mise à jour Protocole (Sentinel Passif) :** L'amplification vectorielle (Signal Boost) a été **désactivée**. Le Sentinel opère désormais en **Mode Monitoring Passif**. Il analyse le CFG mais n'intervient plus sur le signal, résolvant les problèmes de sur-saturation dans les chaînes (Chain/Refiner).
*   **Atomic Guard v14.2 (Stabilité) :** Correction d'un crash critique de backend (`record_stream`) lors du décodage CPU/Hybride. Synchronisation asynchrone renforcée pour éviter la corruption mémoire.
*   **Cache Vision v8.1 :** Algorithme de hachage renforcé (Stride 16) pour éliminer les risques de collisions sur des frames similaires.
*   **Bongmath Adaptatif v3 :** Le moteur de texture reste actif, respectant les scènes sombres tout en optimisant le grain argentique.

---

## 🛡️ Phase 1 : Infiltration (Chargement)

### 🤖 Cyberdyne Model Hub
**Classe :** `CyberdyneModelHub`

Un chargeur unifié et intelligent qui fait le pont entre les architectures Analogiques (Safetensors) et Quantifiées (GGUF). Il gère spécifiquement l'exigence Wan 2.2 Dual-Model (Contexte Haut + Contexte Bas).

| Paramètre | Description |
| :--- | :--- |
| `model_high_name` | Le modèle UNet principal. Supporte `.safetensors` ET `.gguf`. |
| `dtype_high` | Forçage de la précision (`fp16`, `bf16`, `fp8_e4m3fn`, etc.). |
| `model_low_name` | Le modèle UNet secondaire (Requis par Wan 2.2). |
| `enable_checksum` | Effectue un scan d'intégrité SHA256 (Protocole de Sécurité). |
| `offload_inactive` | **"Protocole Skynet" :** Purge agressivement la VRAM des modèles inutilisés avant d'en charger de nouveaux pour éviter les erreurs OOM. |

---

## 🧠 Phase 2 : Cœur Neuronal (Samplers XT-404)

Le moteur "Sentinel" propulse trois nœuds d'échantillonnage spécialisés conçus pour les workflows en chaîne.

### 🔴 XT-404 Skynet 1 (Master)
**Le Commandant.** Initialise la génération et définit le planning de bruit global.
*   **Autorité de Prompt :** Surveillance active du signal (Mode Passif).
*   **Sorties :** Latent, Latent Débruité, Options (pour le chaînage), Seed.

| Paramètre | Description |
| :--- | :--- |
| `sampler_name` | Sélection combinée (ex: `linear/euler`, `beta/dpmpp_2m`). |
| `cfg` | Échelle de guidage. Surveillé par Sentinel. |
| `bongmath` | Moteur de Texture. `True` = Look Film/Analogique. `False` = Numérique/Lisse. |
| `sampler_mode` | Standard ou Resample (injecte du bruit frais). |

### 🟡 XT-404 Skynet 2 (Chain)
**Le Relais.** Conçu pour l'échantillonnage fractionné (ex: 50% sur Master, 30% sur Chain).
*   **Logique :** Masque le widget Seed (utilise l'héritage déterministe interne).
*   **VRAM :** Déchargement dynamique basé sur le type de modèle (Garde GGUF / Décharge FP16).

### 🟢 XT-404 Skynet 3 (Refiner)
**Le Terminator.** Finalise la structure de l'image.
*   **Configuration :** `steps_to_run` par défaut à `-1` (termine le planning).
*   **Focus :** Récupération des détails haute fréquence.

---

## ⚡ Phase 3 : Optimisation Matérielle

### 🚀 Wan Hardware Accelerator
**Classe :** `Wan_Hardware_Accelerator`
Active les optimisations bas niveau PyTorch (TF32) pour les GPU NVIDIA Ampere+. Accélère les multiplications matricielles.

### ✂️ Wan Attention Slicer (SDPA)
**Classe :** `Wan_Attention_Slicer`
Gère le mécanisme d'Attention.
*   **0 (Auto) :** Active Flash Attention (SDPA) pour une vitesse maximale.
*   **1-4 :** Force le découpage (slicing) pour réduire drastiquement les pics de VRAM (plus lent, mais plus sûr pour les cartes <12Go).

### 🧩 Wan Hybrid VRAM Guard
**Classe :** `Wan_Hybrid_VRAM_Guard`
Essentiel pour le Décodage VAE. Remplace le Decode VAE standard.
*   **Décodage Atomique :** Décode 1 frame à la fois.
*   **Tuilage (Tiling) :** Force le tuilage spatial (blocs de 512px).
*   **Déchargement Async :** Transfère les données décodées vers la RAM CPU immédiatement. **Zéro Pic de VRAM.**

### 🍵 Wan TeaCache (Obsidian Engine)
**Classe :** `Wan_TeaCache_Patch`
Implémente un cache pour sauter les calculs U-Net si la différence entre les frames est minime.
*   `rel_l1_threshold` : **0.15** (Recommandé). Plus haut = Génération plus rapide, qualité moindre.

---

## 🛠️ Phase 4 : Post-Production & Outils

### 💾 Wan Video Compressor (H.265)
**Classe :** `Wan_Video_Compressor`
Encode la sortie en H.265 10-bits.
*   **Modes :** Web/Discord (Cible <5Mo), Master (Haute Fidélité), Archival.
*   **Psy-RD :** Préserve la texture du grain sans gonfler la taille du fichier.

### 🧹 Wan Cycle Terminator
**Classe :** `Wan_Cycle_Terminator`
Nettoyage chirurgical de la mémoire. Utilise l'API Windows `EmptyWorkingSet` pour vider la RAM Physique + VRAM.
*   **Fun :** Affiche des citations Skynet/Terminator dans la console lors de l'activation.

### 📐 Resolution Savant & OneShot Cache
*   **Resolution Savant :** Redimensionne les images en assurant que les dimensions sont strictement divisibles par 16 (Exigence Wan), utilisant Lanczos (CPU) ou Area (GPU).
*   **OneShot Cache :** Met en cache les encodages Texte (CLIP) et Vision (I2V) en mémoire "Pinned" pour éviter le re-calcul.

---

<p align="center">
  <i>Architected by Cyberdyne Systems. No fate but what we make.</i>
</p>
