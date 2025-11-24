import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
from comfy.sd import VAE
import gc

# ==============================================================================
# ARCHITECTURE: WAN OPTIMIZER V14 (VRAM FIX)
# SPECIALIZATION: ATOMIC DECODING & ZERO LAG
# ==============================================================================

class WanState:
    def __init__(self):
        self.prev_latent = None
        self.prev_output = None
        self.skipped_steps = 0

class Wan_TeaCache_Patch:
    """
    V13 OBSIDIAN: Adaptive Stride, Dynamic Resolution & Topology Guard.
    (Code inchangé car fonctionnel et performant)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_tea_cache": ("BOOLEAN", {"default": True}),
                "rel_l1_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_autocast": ("BOOLEAN", {"default": True}),
                "start_step_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_step_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("turbo_model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "ComfyWan_Architect/Performance"

    def apply_teacache(self, model, enable_tea_cache, rel_l1_threshold, force_autocast, start_step_percent, end_step_percent):
        if not enable_tea_cache: return (model,)
        m = model.clone()
        m.wan_teacache_state = WanState()
        device = mm.get_torch_device()
        
        is_fp8 = False
        try:
            if hasattr(m.model, "diffusion_model"):
                for param in m.model.diffusion_model.parameters():
                    if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        is_fp8 = True; break
        except: pass

        use_autocast = force_autocast and not is_fp8
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        def teacache_wrapper(model_function, params):
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c")
            t_val = timestep.flatten()[0].item() if isinstance(timestep, torch.Tensor) else timestep
            current_pct = max(0.0, min(1.0, 1.0 - (t_val / 1000.0)))
            
            def run_model(x, t, **kwargs):
                if use_autocast:
                    with torch.autocast(device.type, dtype=autocast_dtype):
                        return model_function(x, t, **kwargs)
                return model_function(x, t, **kwargs)

            if not (start_step_percent <= current_pct <= end_step_percent):
                result = run_model(input_x, timestep, **c)
                m.wan_teacache_state.prev_latent = input_x.detach(); m.wan_teacache_state.prev_output = result
                return result

            can_optimize = False
            if m.wan_teacache_state.prev_latent is not None and m.wan_teacache_state.prev_output is not None:
                if input_x.shape == m.wan_teacache_state.prev_latent.shape: can_optimize = True
            
            if can_optimize:
                b, ch, t, h, w = input_x.shape
                spatial_res = h * w
                stride = 3 if spatial_res < 65536 else (6 if spatial_res < 262144 else 8)
                try:
                    diff = F.l1_loss(input_x[..., ::stride, ::stride], m.wan_teacache_state.prev_latent[..., ::stride, ::stride], reduction='mean')
                    if diff < rel_l1_threshold:
                        m.wan_teacache_state.skipped_steps += 1
                        return m.wan_teacache_state.prev_output
                except: pass

            result = run_model(input_x, timestep, **c)
            m.wan_teacache_state.prev_latent = input_x.detach(); m.wan_teacache_state.prev_output = result
            return result

        m.set_model_unet_function_wrapper(teacache_wrapper)
        print(f">> [Wan Turbo V13] Obsidian Engine Active.")
        return (m,)

class Wan_Hybrid_VRAM_Guard:
    """
    V14 ATOMIC GUARD: FIX CRITICAL VRAM SPIKE & LAG.
    - Force Tiling Spatial (512px) pour éviter l'explosion mémoire.
    - Décodage atomique (1 frame latente à la fois).
    - Suppression du Lag (Async Transfer).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "tile_size_spatial": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "temporal_chunk_size": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": "GARDEZ A 1 pour une VRAM minimale. Augmenter fait exploser la VRAM."}),
                "enable_cpu_offload": ("BOOLEAN", {"default": True}),
                "output_precision": (["fp16", "fp32"], {"default": "fp16"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_hybrid"
    CATEGORY = "ComfyWan_Architect/Performance"
    
    def decode_hybrid(self, vae, samples, tile_size_spatial, temporal_chunk_size, enable_cpu_offload, output_precision):
        latents = samples["samples"]
        
        # 1. CONFIGURATION STRICTE DU TILING (Anti-OOM)
        # On force le VAE à travailler par petits carrés de 512px
        try:
            vae.first_stage_model.enable_tiling(True)
            vae.first_stage_model.tile_sample_min_size = tile_size_spatial
            vae.first_stage_model.tile_latent_min_size = int(tile_size_spatial // 8)
            # Paramètres de chevauchement pour éviter les coutures (Seams)
            vae.first_stage_model.tile_overlap_factor = 0.25 
        except Exception as e:
            print(f"! [Wan Guard] Tiling activation warning: {e}")

        device = mm.get_torch_device()
        work_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        final_dtype = torch.float32 if output_precision == "fp32" else torch.float16
        
        # Dimensions
        # Wan Latents: (Batch, Channels, Time, Height, Width) ou (B, C, H, W)
        shape = latents.shape
        is_5d = len(shape) == 5
        total_input_frames = shape[2] if is_5d else shape[0]

        print(f">> [Wan Guard V14] Atomic Decode. Input: {shape} | Chunk: {temporal_chunk_size} | Tiling: {tile_size_spatial}px")
        
        # 2. NETTOYAGE INITIAL (UNE SEULE FOIS)
        mm.soft_empty_cache()
        gc.collect()

        final_tensor = None
        current_write_idx = 0
        
        # Stream pour le transfert asynchrone (Zéro Lag)
        transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        pbar = comfy.utils.ProgressBar(total_input_frames)

        try:
            # Boucle Principale
            for i in range(0, total_input_frames, temporal_chunk_size):
                end_i = min(i + temporal_chunk_size, total_input_frames)
                
                # A. Chargement Léger (Slicing)
                if is_5d: 
                    # (B, C, T_slice, H, W)
                    chunk_latents = latents[:, :, i:end_i, :, :].to(device, non_blocking=True)
                else: 
                    chunk_latents = latents[i:end_i].to(device, non_blocking=True)
                
                # B. Décodage Protégé (Autocast + Tiling implicite)
                try:
                    with torch.autocast(device.type, dtype=work_dtype):
                        # decode_tiled est plus sûr si disponible, sinon decode standard (qui utilise le tiling configuré plus haut)
                        if hasattr(vae, "decode_tiled"):
                            chunk_image = vae.decode_tiled(chunk_latents, tile_x=tile_size_spatial, tile_y=tile_size_spatial)
                        else:
                            chunk_image = vae.decode(chunk_latents)
                except torch.cuda.OutOfMemoryError:
                    print(f"!! [Wan Guard] OOM detected on chunk {i}. Clearing Cache & Retrying...")
                    mm.soft_empty_cache()
                    torch.cuda.empty_cache()
                    chunk_image = vae.decode(chunk_latents) # Retry brutal

                # Normalisation dimensions (Remove Batch dim if 5D decoded to 5D)
                if len(chunk_image.shape) == 5: 
                    chunk_image = chunk_image.squeeze(0) # (T, H, W, C)

                out_frames = chunk_image.shape[0]
                h, w, c = chunk_image.shape[1], chunk_image.shape[2], chunk_image.shape[3]

                # C. Allocation Intelligente (Premier Tour Uniquement)
                if final_tensor is None:
                    # Estimer la taille totale
                    expansion_ratio = out_frames / (end_i - i) # Combien de frames images pour 1 frame latent ?
                    estimated_total_frames = int(total_input_frames * expansion_ratio)
                    
                    # Allocation en RAM CPU (Pinned pour vitesse)
                    try:
                        final_tensor = torch.empty((estimated_total_frames, h, w, c), dtype=final_dtype, device="cpu", pin_memory=True)
                    except:
                        # Fallback si pas assez de RAM pour Pinning
                        final_tensor = torch.empty((estimated_total_frames, h, w, c), dtype=final_dtype, device="cpu")

                # D. Transfert Zéro-Lag (GPU -> CPU)
                if enable_cpu_offload and transfer_stream:
                    # On attend que le GPU ait fini de décoder CE chunk
                    event = torch.cuda.Event()
                    event.record()
                    transfer_stream.wait_event(event)
                    
                    with torch.cuda.stream(transfer_stream):
                        # Copie asynchrone vers la RAM
                        # Gestion de la taille tampon
                        if current_write_idx + out_frames <= final_tensor.shape[0]:
                            final_tensor[current_write_idx : current_write_idx + out_frames].copy_(chunk_image, non_blocking=True)
                        else:
                            # Extension d'urgence (Rare)
                            cpu_chunk = chunk_image.to("cpu", dtype=final_dtype)
                            final_tensor = torch.cat([final_tensor[:current_write_idx], cpu_chunk], dim=0)
                else:
                    # Mode Synchrone (Plus lent mais compatible tout OS)
                    final_tensor[current_write_idx : current_write_idx + out_frames] = chunk_image.to("cpu", dtype=final_dtype)

                current_write_idx += out_frames
                pbar.update(end_i - i)
                
                # Nettoyage immédiat du tenseur GPU temporaire
                del chunk_latents
                del chunk_image
                # NOTE: Pas de empty_cache() ici pour éviter le Lag. On laisse PyTorch gérer son allocator.

            # Fin de boucle : Synchro finale
            if transfer_stream: transfer_stream.synchronize()
            
        except Exception as e:
            print(f"!! [Wan Guard Error] {e}")
            # En cas de crash, on retourne ce qu'on a pu décoder pour ne pas tout perdre
            if final_tensor is not None:
                return (final_tensor[:current_write_idx],)
            raise e

        # Trim final (ajustement taille exacte)
        result = final_tensor[:current_write_idx]
        
        # Retour (Pas de .clone(), on renvoie la vue directe)
        return (result,)

# ==============================================================================
# NODE MAPPINGS
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "Wan_TeaCache_Patch": Wan_TeaCache_Patch,
    "Wan_Hybrid_VRAM_Guard": Wan_Hybrid_VRAM_Guard
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_TeaCache_Patch": "Wan Turbo (TeaCache)",
    "Wan_Hybrid_VRAM_Guard": "Wan Hybrid Decode (VRAM Guard)"
}