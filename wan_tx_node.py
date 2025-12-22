"""
CYBERDYNE SYSTEMS CORP.
MODULE: T-X POLYMETRIC INTERPOLATOR
FUNCTION: DUAL PHASE LATENT BRIDGING
"""

import torch
import nodes
import node_helpers
import comfy.model_management as mm
import comfy.ldm.wan.vae
import comfy.utils
import sys
import gc
import time

# --- CORE OVERRIDE PROTOCOLS ---
# Modification temporaire du VAE pour gérer la cohérence temporelle
# sur des séquences vides (Timeline Bridging).

original_encode = comfy.ldm.wan.vae.WanVAE.encode
original_decode = comfy.ldm.wan.vae.WanVAE.decode

def tx_encode_override(self, x):
    # Allocation dynamique du cache pour éviter les débordements mémoire
    self._enc_feat_map = [None] * 64
    self._enc_conv_idx = [0]
    self._enc_conv_num = 64

    t = x.shape[2]
    iter_ = 2 + (t - 2) // 4

    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
        elif i == iter_ - 1:
            out_ = self.encoder(x[:, :, -1:, :, :],
                                feat_cache=[None] * self._enc_conv_num,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
        else:
            out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
            
    out_head = out[:, :, :iter_ - 1, :, :]
    out_tail = out[:, :, -1, :, :].unsqueeze(2)
    mu, log_var = torch.cat([self.conv1(out_head), self.conv1(out_tail)], dim=2).chunk(2, dim=1)
    return mu

def tx_decode_override(self, z):
    self._feat_map = [None] * 64
    self._conv_idx = [0]
    self._dec_conv_num = 64
    
    iter_ = z.shape[2]
    z_head = z[:, :, :-1, :, :]
    z_tail = z[:, :, -1, :, :].unsqueeze(2)
    x = torch.cat([self.conv2(z_head), self.conv2(z_tail)], dim=2)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx)
        elif i == iter_ - 1:
            out_ = self.decoder(x[:, :, -1, :, :].unsqueeze(2),
                                feat_cache=None,
                                feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
        else:
            out_ = self.decoder(x[:, :, i:i + 1, :, :],
                                feat_cache=self._feat_map,
                                feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    return out

class Wan_TX_Interpolator:
    """
    T-X SERIES: POLYMETRIC INTERPOLATOR
    Capacité : Transition liquide entre deux états (Start/End).
    Moteur : Injection Native VAE + Optimisations Ultra.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "video_frames": ("INT", {"default": 81, "min": 5, "max": 4096, "step": 4, "tooltip": "Taille temporelle exacte."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "start_image": ("IMAGE", ),
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
                "tile_strategy": (["Auto (Smart VRAM)", "512x512 (Safe)", "1024x1024 (Fast)", "1280x1280 (Ultra)"], {"default": "Auto (Smart VRAM)"}),
            },
            "optional": {
                "end_image": ("IMAGE", ),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "XT-404/Wan2.2"

    def _log(self, step_name, info=""):
        if step_name == "Init":
            self.t0 = time.time()
            self.step_t0 = self.t0
            print(f"\n╔══════════════════════════════════════════════════════════════╗", flush=True)
            print(f"║   [CYBERDYNE T-X] POLYMETRIC INTERPOLATOR (ONLINE)           ║", flush=True)
            print(f"╚══════════════════════════════════════════════════════════════╝", flush=True)
            return
        
        current_time = time.time()
        dt = current_time - self.step_t0
        total_t = current_time - self.t0
        
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        info_str = f" | {info}" if info else ""
        print(f"👉 [T-X] {step_name}{info_str}", flush=True)
        print(f"   └── Stats: {dt:.2f}s (Tot: {total_t:.2f}s) | VRAM Alloc: {mem_alloc:.2f}GB (Res: {mem_reserved:.2f}GB)", flush=True)
        self.step_t0 = current_time

    def _get_dtype(self, precision_str):
        if precision_str == "fp16": return torch.float16
        if precision_str == "bf16": return torch.bfloat16
        return torch.float32

    def _determine_tile_size(self, strategy_str):
        print(f"\n🔎 [T-X] --- HARDWARE SCAN ---", flush=True)
        
        if "512" in strategy_str: return 512
        if "1024" in strategy_str: return 1024
        if "1280" in strategy_str: return 1280
        
        try:
            device = mm.get_torch_device()
            total_mem_gb = mm.get_total_memory(device) / (1024**3)
            gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "Unknown Device"
            
            print(f"   ├── GPU: {gpu_name}", flush=True)
            print(f"   ├── VRAM: {total_mem_gb:.2f} GB", flush=True)
            
            if total_mem_gb > 22.0: return 1280
            elif total_mem_gb > 10.0: return 1024
            else: return 512
        except Exception: return 512

    def _sanitize_tensor(self, tensor, target_w, target_h, device, dtype):
        if tensor.device != device or tensor.dtype != dtype:
            tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
        if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
        tensor_bchw = tensor.movedim(-1, 1)
        tensor_resized = torch.nn.functional.interpolate(tensor_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True)
        return torch.clamp(tensor_resized, 0.0, 1.0).movedim(1, -1)

    def _enhance_details(self, image_tensor, factor=0.5):
        if factor <= 0: return image_tensor
        device = image_tensor.device
        img_bchw = image_tensor.movedim(-1, 1)
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=image_tensor.dtype, device=device).view(1, 1, 3, 3)
        b, c, h, w = img_bchw.shape
        enhanced = torch.nn.functional.conv2d(img_bchw.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        return torch.lerp(img_bchw, enhanced, factor * 0.2).clamp(0,1).movedim(1, -1)

    def _smart_encode(self, vae, pixels, tile_size, description="Unknown", device=None):
        if device and pixels.device != device:
             pixels = pixels.to(device, non_blocking=True)
        
        use_tiled = hasattr(vae, "encode_tiled")
        print(f"   ├── {description} : Input Shape {list(pixels.shape)}", flush=True)
        
        if use_tiled:
            try:
                overlap = tile_size // 8
                return vae.encode_tiled(pixels, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
            except Exception as e:
                print(f"   └── ⚠️ Tiling Failed ({e}) -> Standard Fallback", flush=True)
                return vae.encode(pixels)
        else:
            return vae.encode(pixels)

    def execute(self, positive, negative, vae, video_frames, width, height, batch_size, start_image, detail_boost, motion_amp, precision, tile_strategy, end_image=None, clip_vision_output=None):
        self._log("Init")
        
        valid_end_image = end_image is not None
        device = mm.get_torch_device()
        target_dtype = self._get_dtype(precision)
        tile_size = self._determine_tile_size(tile_strategy)

        # 1. Activation du Patch VAE (Mode Injection Native)
        if valid_end_image:
            comfy.ldm.wan.vae.WanVAE.encode = tx_encode_override
            comfy.ldm.wan.vae.WanVAE.decode = tx_decode_override
            print("   ├── 🛠️ [T-X Protocol] VAE Overridden (Start/End Injection).", flush=True)
        else:
            comfy.ldm.wan.vae.WanVAE.encode = original_encode
            comfy.ldm.wan.vae.WanVAE.decode = original_decode

        try:
            if hasattr(mm, "load_models_gpu"):
                 models_to_load = [vae.patcher] if hasattr(vae, "patcher") else [vae]
                 mm.load_models_gpu(models_to_load)
            gc.collect()
            mm.soft_empty_cache()

            # 2. Préparation
            resized_start = self._sanitize_tensor(start_image, width, height, device, target_dtype)
            if detail_boost > 0: 
                resized_start = self._enhance_details(resized_start, detail_boost)
                self._log("Start Image", f"Detail Boost: {detail_boost}")

            # 3. Construction du Volume Temporel
            if valid_end_image:
                empty_count = video_frames - 2
                empty_frames = torch.ones(empty_count, height, width, 3, device=device, dtype=target_dtype) * 0.5
                
                resized_end = self._sanitize_tensor(end_image, width, height, device, target_dtype)
                if detail_boost > 0: resized_end = self._enhance_details(resized_end, detail_boost)
                
                concatenated = torch.cat([resized_start, empty_frames, resized_end], dim=0)
                self._log("Volume Build", f"Start + {empty_count} Void + End")
            else:
                empty_count = video_frames - 1
                empty_frames = torch.ones(empty_count, height, width, 3, device=device, dtype=target_dtype) * 0.5
                
                concatenated = torch.cat([resized_start, empty_frames], dim=0)
                self._log("Volume Build", f"Start + {empty_count} Void")

            # 4. Encodage
            print(f"🚀 [T-X] Encoding Sequence...", flush=True)
            la = self._smart_encode(vae, concatenated, tile_size, description="Video Volume", device=device)
            self._log("Encoding", f"Latent Shape: {list(la.shape)}")

            # 5. Motion Amp (Post-Process)
            if motion_amp > 1.0:
                base_latent = la[:, :, 0:1]
                diff = la - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                scaled_latent = base_latent + diff_centered * motion_amp + diff_mean
                la = torch.clamp(scaled_latent, -6.0, 6.0)
                self._log("Motion Amp", f"Dynamics: {motion_amp}")

            # 6. Masque Temporel (Synchronisé)
            target_t_latent = la.shape[2]
            latent = torch.zeros([batch_size, 16, target_t_latent, height // 8, width // 8], device=device)
            
            mask_len = concatenated.shape[0]
            mask = torch.ones((1, mask_len, la.shape[-2], la.shape[-1]), device=device)
            
            mask[:, 0] = 0.0
            if valid_end_image:
                mask[:, -1] = 0.0
                print("   ├── 🔒 [Timeline] End Frame Locked.", flush=True)

            start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
            
            if valid_end_image:
                end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)
                middle_mask = mask[:, 1:-1]
                mask_final = torch.cat([start_mask_repeated, middle_mask, end_mask_repeated], dim=1)
            else:
                middle_mask = mask[:, 1:]
                mask_final = torch.cat([start_mask_repeated, middle_mask], dim=1)

            # Ajustement Dimensionnel
            needed_flat_len = la.shape[2] * 4
            current_flat_len = mask_final.shape[1]
            
            if current_flat_len < needed_flat_len:
                pad = torch.ones((1, needed_flat_len - current_flat_len, mask.shape[2], mask.shape[3]), device=device)
                mask_final = torch.cat([mask_final, pad], dim=1)
            elif current_flat_len > needed_flat_len:
                mask_final = mask_final[:, :needed_flat_len]

            mask_final = mask_final.view(1, la.shape[2], 4, la.shape[-2], la.shape[-1])
            mask_final = mask_final.movedim(1, 2)
            mask_final = mask_final.repeat(1, 4, 1, 1, 1)

            # 7. Conditioning
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": la, "concat_mask": mask_final})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": la, "concat_mask": mask_final})

            if clip_vision_output is not None:
                positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
                negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

            out_latent = {"samples": torch.zeros_like(la)}

        finally:
            comfy.ldm.wan.vae.WanVAE.encode = original_encode
            comfy.ldm.wan.vae.WanVAE.decode = original_decode
            print("   ├── 🧹 [System] VAE Restored.", flush=True)

        gc.collect()
        mm.soft_empty_cache()
        self._log("Process Complete")
        
        return (positive, negative, out_latent)