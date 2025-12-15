import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import time
import sys
import gc

class WanImageToVideoUltra:
    """
    WanImageToVideoUltra - REPORTER V18 (Ultimate + Smart VRAM Scan + Full Verbose).
    
    Améliorations V18 :
    - SMART VRAM SCAN : Rapport complet (Nom GPU, VRAM Totale, Logique de décision).
    - FULL VERBOSE : Chaque étape détaille la résolution, le format et l'usage mémoire.
    - Toujours inclus : GPU Lock, FP16/BF16, Async Transfer, Nuclear Norm.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "video_frames": ("INT", {"default": 114, "min": 1, "max": 4096, "step": 1, "tooltip": "Nombre exact de frames."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05}),
                "force_ref": ("BOOLEAN", {"default": True}),
                # OPTIMISATION
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32", "tooltip": "fp16 = Gain VRAM massif. bf16 = Recommandé pour RTX 3000/4000."}),
                "tile_strategy": (["Auto (Smart VRAM)", "512x512 (Safe)", "1024x1024 (Fast)", "1280x1280 (Ultra)"], {"default": "Auto (Smart VRAM)"}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "WanVideo/Ultra"

    def _log(self, step_name, info=""):
        """Mouchard Cafard V4: Logs ultra-détaillés"""
        if step_name == "Init":
            self.t0 = time.time()
            self.step_t0 = self.t0
            print(f"\n╔══════════════════════════════════════════════════════════════╗", flush=True)
            print(f"║   [WanUltra V18] DEMARRAGE (REPORTER EDITION)                ║", flush=True)
            print(f"╚══════════════════════════════════════════════════════════════╝", flush=True)
            return
        
        current_time = time.time()
        dt = current_time - self.step_t0
        total_t = current_time - self.t0
        
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        info_str = f" | {info}" if info else ""
        print(f"👉 [WanUltra] {step_name}{info_str}", flush=True)
        print(f"   └── Stats: {dt:.2f}s (Tot: {total_t:.2f}s) | VRAM Alloc: {mem_alloc:.2f}GB (Res: {mem_reserved:.2f}GB)", flush=True)
        self.step_t0 = current_time

    def _get_dtype(self, precision_str):
        if precision_str == "fp16": return torch.float16
        if precision_str == "bf16": return torch.bfloat16
        return torch.float32

    def _determine_tile_size(self, strategy_str):
        """Scan Matériel et Décision Logique"""
        print(f"\n🔎 [WanUltra] --- SMART VRAM SCANNER ---", flush=True)
        
        # 1. Détection Force Manuelle
        if "512" in strategy_str: 
            print(f"   └── Mode: MANUEL (Safe 512px) - Ignorer VRAM", flush=True)
            return 512
        if "1024" in strategy_str:
            print(f"   └── Mode: MANUEL (Fast 1024px) - Ignorer VRAM", flush=True)
            return 1024
        if "1280" in strategy_str:
            print(f"   └── Mode: MANUEL (Ultra 1280px) - Ignorer VRAM", flush=True)
            return 1280
        
        # 2. Mode AUTO (Smart Scan)
        try:
            device = comfy.model_management.get_torch_device()
            total_mem_bytes = comfy.model_management.get_total_memory(device)
            total_mem_gb = total_mem_bytes / (1024**3)
            gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "Unknown Device"
            
            print(f"   ├── GPU Detecté : {gpu_name}", flush=True)
            print(f"   ├── VRAM Totale : {total_mem_gb:.2f} GB", flush=True)
            
            # Logique de décision
            if total_mem_gb > 22.0: 
                print(f"   ├── Analyse : VRAM Massive (>22GB). Mode ULTRA activé.", flush=True)
                print(f"   └── DECISION : Tuiles 1280x1280", flush=True)
                return 1280 # RTX 3090/4090
            elif total_mem_gb > 10.0: 
                print(f"   ├── Analyse : VRAM Confortable (>10GB). Mode FAST activé.", flush=True)
                print(f"   └── DECISION : Tuiles 1024x1024", flush=True)
                return 1024 # RTX 3060/4070/2080Ti
            else: 
                print(f"   ├── Analyse : VRAM Limitée (<10GB). Mode SAFE activé.", flush=True)
                print(f"   └── DECISION : Tuiles 512x512", flush=True)
                return 512 # 8GB Cards
                
        except Exception as e:
            print(f"   └── ⚠️ ERREUR SCAN ({e}) -> Fallback Safe 512px", flush=True)
            return 512

    def _sanitize_tensor(self, tensor, target_w, target_h, device, dtype):
        """Nettoyage et Interpolation (Optimisé)"""
        # Transfert Asynchrone
        if tensor.device != device or tensor.dtype != dtype:
            tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

        if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
            
        tensor_bchw = tensor.movedim(-1, 1)
        
        # Interpolation
        tensor_resized = torch.nn.functional.interpolate(
            tensor_bchw, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
        )
        
        # Clamping
        tensor_resized = torch.clamp(tensor_resized, 0.0, 1.0)
        return tensor_resized.movedim(1, -1)

    def _enhance_details(self, image_tensor, factor=0.5):
        if factor <= 0: return image_tensor
        device = image_tensor.device
        img_bchw = image_tensor.movedim(-1, 1)
        
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
                            dtype=image_tensor.dtype, device=device).view(1, 1, 3, 3)
        
        b, c, h, w = img_bchw.shape
        enhanced = torch.nn.functional.conv2d(img_bchw.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        
        result_bchw = torch.lerp(img_bchw, enhanced, factor * 0.2)
        result_bchw = torch.clamp(result_bchw, 0.0, 1.0)
        return result_bchw.movedim(1, -1)

    def _smart_encode(self, vae, pixels, tile_size, description="Unknown", device=None):
        """Mouchard VAE avec Rapport de Décision"""
        # Sécurité GPU
        if device and pixels.device != device:
             pixels = pixels.to(device, non_blocking=True)
        
        use_tiled = hasattr(vae, "encode_tiled")
        
        print(f"   ├── {description} : Input Shape {list(pixels.shape)}", flush=True)
        
        if use_tiled:
            try:
                overlap = tile_size // 8
                print(f"   ├── Méthode : TILED (Taille: {tile_size}, Overlap: {overlap})", flush=True)
                return vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
            except Exception as e:
                print(f"   └── 🛑 ECHEC TILED ({e}) -> BASCULE STANDARD", flush=True)
                return vae.encode(pixels)
        else:
            print(f"   └── Méthode : STANDARD (Tiled non supporté par ce VAE)", flush=True)
            return vae.encode(pixels)

    def execute(self, positive, negative, vae, video_frames, width, height, batch_size, detail_boost, motion_amp, force_ref, precision, tile_strategy, start_image=None, clip_vision_output=None):
        # 0. Initialisation
        self._log("Init")
        
        device = comfy.model_management.get_torch_device()
        target_dtype = self._get_dtype(precision)
        
        # 1. SCAN MATERIEL & DECISION TUILES
        tile_size = self._determine_tile_size(tile_strategy)
        
        self._log("Configuration", f"Precision: {precision.upper()} | Device: {device}")

        # 2. GPU LOCK
        try:
            if hasattr(comfy.model_management, "load_models_gpu"):
                 models_to_load = [vae.patcher] if hasattr(vae, "patcher") else [vae]
                 comfy.model_management.load_models_gpu(models_to_load)
                 print(f"🔐 [WanUltra] GPU LOCK ACTIVÉ : Modèles verrouillés en VRAM.", flush=True)
        except: pass

        gc.collect()
        comfy.model_management.soft_empty_cache()

        length = video_frames
        latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=device, dtype=torch.float32)

        if start_image is not None:
            # 3. SANITIZATION (Nuclear Norm)
            img_final = self._sanitize_tensor(start_image, width, height, device, target_dtype)
            
            # 4. Detail Boost
            if detail_boost > 0:
                img_final = self._enhance_details(img_final, detail_boost)
            
            self._log(f"Preparation Image HD", f"Shape: {list(img_final.shape)} | Dtype: {img_final.dtype}")

            # 5. Encodage Reference
            ref_latent = None
            if force_ref:
                try:
                    print(f"🚀 [WanUltra] Lancement Encodage Reference...", flush=True)
                    full_ref_latent = self._smart_encode(vae, img_final, tile_size, description="Reference Image", device=device)
                    ref_latent = full_ref_latent[:, :, 0:1, :, :]
                    self._log("Reference Encoded")
                except Exception as e:
                    print(f"❌ [WanUltra] ERREUR REF ENCODE: {e}", flush=True)
                    ref_latent = None
            
            # 6. Préparation Volume Vidéo
            valid_frames = min(img_final.shape[0], length)
            video_input = torch.full((length, height, width, 3), 0.5, dtype=target_dtype, device=device)
            video_input[:valid_frames] = img_final[:valid_frames]
            
            del start_image
            self._log("Volume Video Created", f"Frames: {length}")

            # 7. Encodage VAE Principal
            try:
                print(f"🚀 [WanUltra] Lancement Encodage Video Principal...", flush=True)
                concat_latent_image = self._smart_encode(vae, video_input, tile_size, description="Video Volume", device=device)
            except Exception as e:
                print(f"❌ [WanUltra] ERREUR CRITIQUE VAE: {e}", flush=True)
                raise e
            
            del video_input, img_final
            self._log("VAE Video Encoded")

            # 8. MOTION AMPLIFICATION
            if motion_amp > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                
                diff = gray_latent - base_latent
                std = diff.std()
                if std > 0 and std > 1.0: diff = diff / std
                
                limit = 2.5 
                boosted = diff * motion_amp
                soft_diff = torch.tanh(boosted / limit) * limit
                
                scaled_latent = base_latent + soft_diff
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
                self._log(f"Motion Amp Applied", f"Factor: {motion_amp}")

            # 9. Conditioning
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=device, dtype=torch.float32)
            start_latent_end_index = ((valid_frames - 1) // 4) + 1
            mask[:, :, :start_latent_end_index] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

            if ref_latent is not None:
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
            
            del mask, concat_latent_image, ref_latent
            self._log("Conditioning Finalized")

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        
        gc.collect()
        comfy.model_management.soft_empty_cache()
        
        print(f"\n╚══════════════════════════════════════════════════════════════╝", flush=True)
        self._log("Job Terminé & Nettoyé")
        
        return (positive, negative, out_latent)
