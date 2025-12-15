# 🤖 XT-404 Skynet : Wan 2.2 Sentinel Suite
### Cyberdyne Systems Corp. | Series T-800 | Model 101

<p align="center">
  <img src="https://img.shields.io/badge/Version-v29.0_GOLD_MASTER-yellow?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Architecture-Wan_2.2-blue?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Engine-T_3000_Genisys-red?style=for-the-badge" alt="Engine">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License">
</p>

> *"The future is not set. There is no fate but what we make for ourselves."*

---

## ⚠️ CRITICAL SYSTEM DEPENDENCY / DÉPENDANCE CRITIQUE

> [!CAUTION]
> **INFILTRATION PROTOCOL (GGUF):**
> To utilize GGUF Quantized Models with the **Cyberdyne Model Hub**, the **ComfyUI-GGUF** engine is **REQUIRED**.
>
> 📥 **Download Engine:** `city96/ComfyUI-GGUF`
>
> *Without this engine, the Cyberdyne Model Hub will operate in Safetensors-only mode.*

---

## 🌍 NEURAL NET NAVIGATION / NAVIGATION DU RÉSEAU

### 🇺🇸 [ENGLISH DOCUMENTATION](#-english-documentation)
1. [Phase 1: Infiltration (Model Loader)](#-phase-1-infiltration-cyberdyne-model-hub)
2. [Phase 2: Neural Net Core (XT-404 Samplers)](#-phase-2-neural-net-core-xt-404-samplers)
3. [Phase 3: T-3000 Genisys (Omniscient Cache)](#-phase-3-t-3000-genisys-omniscient-cache)
4. [Phase 4: Mimetic Rendering (I2V Ultra & Fidelity)](#-phase-4-mimetic-rendering-i2v-ultra--fidelity)
5. [Phase 5: Sensors & Accelerators (Omega Tools)](#-phase-5-sensors--accelerators-omega-tools)
6. [Phase 6: Post-Processing & Automation](#-phase-6-post-processing--automation)

### 🇫🇷 [DOCUMENTATION FRANÇAISE](#-documentation-française)
1. [Phase 1 : Infiltration (Chargement Modèles)](#-phase-1--infiltration-cyberdyne-model-hub)
2. [Phase 2 : Cœur Neuronal (Samplers XT-404)](#-phase-2--cœur-neuronal-samplers-xt-404)
3. [Phase 3 : T-3000 Genisys (Cache Omniscient)](#-phase-3--t-3000-genisys-cache-omniscient)
4. [Phase 4 : Rendu Mimétique (I2V Ultra & Fidelity)](#-phase-4--rendu-mimétique-i2v-ultra--fidelity)
5. [Phase 5 : Capteurs & Accélérateurs (Outils Omega)](#-phase-5--capteurs--accélérateurs-outils-omega)
6. [Phase 6 : Post-Production & Automatisation](#-phase-6--post-production--automatisation)

---

# 🇺🇸 ENGLISH DOCUMENTATION

## 🛡️ Phase 1: Infiltration (Cyberdyne Model Hub)

### 🤖 Cyberdyne Model Hub
**File:** `cyberdyne_model_hub.py`

A unified, intelligent loader that bridges the gap between Analog (Safetensors) and Quantized (GGUF) architectures. It specifically handles the Wan 2.2 Dual-Model requirement (High Context + Low Context) and includes a recursive file scanner.

| Parameter | Description |
| :--- | :--- |
| `model_high_name` | The main UNet model. Supports `.safetensors` AND `.gguf`. |
| `dtype_high` | Precision override (`fp16`, `bf16`, `fp8_e4m3fn`, etc.). |
| `model_low_name` | The secondary UNet model (Wan 2.2 requirement). |
| `enable_checksum` | Performs a SHA256 integrity scan (Security Protocol). |
| `offload_inactive` | **"Skynet Protocol":** Aggressively purges VRAM of unused models before loading new ones. |

---

## 🧠 Phase 2: Neural Net Core (XT-404 Samplers)

**File:** `XT404_Skynet_Nodes.py`

The "Sentinel" engine powers three specialized sampling nodes designed for chained workflows (Master -> Chain -> Refiner). They utilize a specialized noise scheduler (`simple`/`Linear`) mandatory for Wan 2.2.

### 🔴 XT-404 Skynet 1 (Master)
**The Commander node.** Initializes generation and defines the global noise schedule.
*   **shift_val:** **5.0** (Default). Critical for Wan 2.2. Controls the noise schedule curve.
*   **bongmath:** Texture Engine. `True` = Film/Analog look. `False` = Digital/Smooth.
*   **sampler_mode:** `standard` (Default).

### 🟡 XT-404 Skynet 2 (Chain)
**The Relay node.** Designed for split-sampling.
*   **Logic:** Hides the Seed widget (uses internal deterministic inheritance from Master).
*   **steps_to_run:** Defines how many steps this specific node executes before passing to the next.

### 🟢 XT-404 Skynet 3 (Refiner)
**The Terminator node.** Finalizes the image structure.
*   **sampler_mode:** `resample` (Default). Injects fresh noise to refine details.

---

## 💀 Phase 3: T-3000 Genisys (Omniscient Cache)

**File:** `wan_genisys.py`

**The "Omniscient" Edition.** A highly advanced caching system that replaces standard TeaCache. It visualizes "Signal Health" in the console and uses Kinetic Momentum to prevent static video freezing.

| Parameter | Description |
| :--- | :--- |
| `system_status` | Master switch for the T-3000 engine. |
| `security_level` | **7** (Default). Controls the cache threshold. 1=Lax, 10=Strict. Adjusts how much change triggers a recalc. |
| `warmup_steps` | **6** (Default). Number of initial steps where caching is **forbidden**. Crucial for establishing the prompt's subject. |
| `kinetic_momentum` | **2** (Default). Forces calculation for X frames after a movement is detected to maintain motion inertia. |
| `hud_display` | Activates the **Cyberdyne HUD** in the console (Visualizes Drift, Tao, Mag, Signal Integrity). |

---

## 🎭 Phase 4: Mimetic Rendering (I2V Ultra & Fidelity)

### 🌟 Wan Image To Video Ultra
**File:** `nodes_wan_ultra.py`
The definitive engine for Image-to-Video. Features a "Mouchard" (Snitch) for performance monitoring.

*   **FP32 Forced Pipeline:** All math runs in 32-bit floating point to eliminate color banding.
*   **detail_boost:** A GPU-sharpening matrix applied *before* encoding to counteract VAE blur.
*   **motion_amp:** Dynamic booster. **1.0** = Normal. **1.15** = Forced movement dynamics.
*   **force_ref:** Injects the source image as a hard reference (Identity Lock).

### ⚡ Wan Image To Video Fidelity
**File:** `wan_fast.py`
Optimized version for speed while maintaining FP32 precision on the latent canvas. Uses `torch.full` for memory efficiency.

---

## ⚡ Phase 5: Sensors & Accelerators (Omega Tools)

### 🚀 Wan Hardware Accelerator (Omega)
**File:** `wan_accelerator.py`
*   **enable_tf32:** Activates TensorFloat-32 on Ampere+ GPUs (~30% speedup).
*   **Attention Slicer:** Smart management of SDPA (Flash Attention) vs Manual Slicing for Low VRAM.

### 👁️ Wan Vision & Text OneShot Cache
**File:** `wan_i2v_tools.py` & `wan_text_encoder.py`
*   **Vision Cache:** Hashes the input image (including stride sampling) to prevent re-encoding the same CLIP Vision data.
*   **Text Cache:** Uses **Pinned Memory (DMA)** to transfer text embeddings from CPU to GPU instantly.

### 📐 Wan Resolution Savant (FP32)
**File:** `wan_i2v_tools.py`
Resizes images ensuring dimensions are strictly divisible by 16 (Required by Wan). Uses **FP32 interpolation** (Lanczos/Bicubic) to prevent aliasing.

---

## 🛠️ Phase 6: Post-Processing & Automation

### 💾 Wan Video Compressor (H.265)
**File:** `wan_compressor.py`
Encodes output to H.265 10-bit.
*   **Thread Safe:** Auto-limits threads (Max 16) to prevent x265 crashes on high-end CPUs (Threadripper/i9).
*   **Modes:** Web/Discord (CRF 26), Master (CRF 22), Archival (CRF 18).

### 🧹 Wan Cycle Terminator
**File:** `wan_cleanup.py`
Surgical memory cleaning using Windows API `EmptyWorkingSet`. Use only when switching heavy workflows to flush RAM/VRAM without crashing.

### 📉 Auto Image Optimizers
**File:** `auto_wan_node.py` & `auto_half_node.py`
*   **Auto Wan Optimizer:** Smartly resizes images to safeguard against OOM (Max 1024px) while respecting Modulo 16.
*   **Auto Half Size:** Quick 50% downscaler with bicubic antialiasing.

---
---

# 🇫🇷 DOCUMENTATION FRANÇAISE

## 🛡️ Phase 1 : Infiltration (Cyberdyne Model Hub)

### 🤖 Cyberdyne Model Hub
**Fichier :** `cyberdyne_model_hub.py`

Un chargeur unifié qui gère l'exigence Wan 2.2 Dual-Model (High + Low Context) et supporte nativement les fichiers GGUF via un scan récursif.

| Paramètre | Description |
| :--- | :--- |
| `model_high_name` | Modèle principal. Supporte `.safetensors` ET `.gguf`. |
| `dtype_high` | Forçage précision (`fp16`, `bf16`, `fp8_e4m3fn`, etc.). |
| `model_low_name` | Modèle secondaire (Requis par Wan 2.2). |
| `enable_checksum` | Scan d'intégrité SHA256 (Sécurité). |
| `offload_inactive` | **"Protocole Skynet" :** Purge la VRAM avant chargement. |

---

## 🧠 Phase 2 : Cœur Neuronal (Samplers XT-404)

**Fichier :** `XT404_Skynet_Nodes.py`

Le moteur "Sentinel" propulse trois nœuds de sampling conçus pour les workflows chaînés (Master -> Chain -> Refiner). Ils utilisent le scheduler spécifique `simple` (Linear) obligatoire pour Wan 2.2.

### 🔴 XT-404 Skynet 1 (Master)
**Le Commandant.** Initialise la génération et définit la courbe de bruit.
*   **shift_val :** **5.0** (Défaut). Crucial pour Wan 2.2.
*   **bongmath :** Moteur de Texture. `True` = Grain Film/Analogique. `False` = Numérique/Lisse.
*   **sampler_mode :** `standard` (Défaut).

### 🟡 XT-404 Skynet 2 (Chain)
**Le Relais.** Conçu pour l'échantillonnage fractionné.
*   **Logique :** Masque le widget Seed (utilise l'héritage déterministe interne du Master).
*   **steps_to_run :** Définit le nombre d'étapes exécutées par ce nœud avant de passer la main.

### 🟢 XT-404 Skynet 3 (Refiner)
**Le Terminator.** Finalise la structure de l'image.
*   **sampler_mode :** `resample` (Défaut). Réinjecte du bruit frais pour affiner les détails.

---

## 💀 Phase 3 : T-3000 Genisys (Cache Omniscient)

**Fichier :** `wan_genisys.py`

**L'Édition "Omnisciente".** Un système de cache ultra-avancé remplaçant le TeaCache. Il visualise la "Santé du Signal" dans la console et utilise le "Kinetic Momentum" pour empêcher le gel des vidéos.

| Paramètre | Description |
| :--- | :--- |
| `system_status` | Interrupteur principal du moteur T-3000. |
| `security_level` | **7** (Défaut). Contrôle le seuil du cache. 1=Laxiste, 10=Strict. Ajuste la sensibilité au changement. |
| `warmup_steps` | **6** (Défaut). Nombre d'étapes initiales où le cache est **interdit**. Vital pour imprimer le sujet du prompt. |
| `kinetic_momentum` | **2** (Défaut). Force le calcul pour X frames après une détection de mouvement (Inertie). |
| `hud_display` | Active le **HUD Cyberdyne** dans la console (Visualise Drift, Tao, Mag, Intégrité Signal). |

---

## 🎭 Phase 4 : Rendu Mimétique (I2V Ultra & Fidelity)

### 🌟 Wan Image To Video Ultra
**Fichier :** `nodes_wan_ultra.py`
Le moteur définitif pour l'Image-to-Video. Intègre un "Mouchard" pour le monitoring de performance.

*   **Pipeline FP32 Forcé :** Tous les calculs sont en 32 bits pour éliminer les bandes de couleurs (banding).
*   **detail_boost :** Matrice de netteté GPU appliquée *avant* l'encodage pour contrer le flou du VAE.
*   **motion_amp :** Booster dynamique. **1.0** = Normal. **1.15** = Dynamique de mouvement forcée.
*   **force_ref :** Injecte l'image source comme référence dure (Verrouillage d'Identité).

### ⚡ Wan Image To Video Fidelity
**Fichier :** `wan_fast.py`
Version optimisée pour la vitesse tout en maintenant la précision FP32 sur le canvas latent. Utilise `torch.full` pour l'efficacité mémoire.

---

## ⚡ Phase 5 : Capteurs & Accélérateurs (Outils Omega)

### 🚀 Wan Hardware Accelerator (Omega)
**Fichier :** `wan_accelerator.py`
*   **enable_tf32 :** Active TensorFloat-32 sur GPU Ampere+ (Gain vitesse ~30%).
*   **Attention Slicer :** Gestion intelligente de SDPA (Flash Attention) vs Slicing Manuel pour faible VRAM.

### 👁️ Wan Vision & Text OneShot Cache
**Fichiers :** `wan_i2v_tools.py` & `wan_text_encoder.py`
*   **Vision Cache :** Hash l'image d'entrée (incluant l'échantillonnage) pour éviter de ré-encoder le CLIP Vision.
*   **Text Cache :** Utilise la **Mémoire Épinglée (Pinned Memory/DMA)** pour transférer les embeddings texte du CPU au GPU instantanément.

### 📐 Wan Resolution Savant (FP32)
**Fichier :** `wan_i2v_tools.py`
Redimensionne les images pour qu'elles soient divisibles par 16 (Requis par Wan). Utilise l'interpolation **FP32** (Lanczos/Bicubic) pour éviter l'aliasing.

---

## 🛠️ Phase 6 : Post-Production & Automatisation

### 💾 Wan Video Compressor (H.265)
**Fichier :** `wan_compressor.py`
Encode la sortie en H.265 10-bits.
*   **Thread Safe :** Limite auto les threads (Max 16) pour éviter les crashs x265 sur les gros CPU (Threadripper/i9).
*   **Modes :** Web/Discord (CRF 26), Master (CRF 22), Archival (CRF 18).

### 🧹 Wan Cycle Terminator
**Fichier :** `wan_cleanup.py`
Nettoyage chirurgical de la mémoire via API Windows `EmptyWorkingSet`. À utiliser lors du changement de workflow lourd pour purger RAM/VRAM sans crash.

### 📉 Auto Image Optimizers
**Fichiers :** `auto_wan_node.py` & `auto_half_node.py`
*   **Auto Wan Optimizer :** Redimensionne intelligemment pour protéger contre le OOM (Max 1024px) tout en respectant le Modulo 16.
*   **Auto Half Size :** Downscaler rapide 50% avec antialiasing bicubique.

---

<p align="center">
  <i>Architected by Cyberdyne Systems. No fate but what we make.</i>
</p>
