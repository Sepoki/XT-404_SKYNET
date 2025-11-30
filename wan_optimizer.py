import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
from comfy.sd import VAE
import gc
import time

# ==============================================================================
# ARCHITECTURE: WAN OPTIMIZER OMEGA (V6 - TESSERACT ENGINE)
# ==============================================================================

class WanSubState:
    def __init__(self):
        self.prev_latent = None
        self.prev_output = None
        self.skipped_steps = 0
        self.step_counter = 0

class WanState:
    def __init__(self):
        self.flows = {} 
        self.debug_printed = False
        self.autocast_failed_once = False 

class Wan_TeaCache_Patch:
    """
    OMEGA EDITION V5: Chronos Sentinel (Inchangé car fonctionnel).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_tea_cache": ("BOOLEAN", {"default": True}),
                "rel_l1_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.6, "step": 0.01}),
                "start_step_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "force_autocast": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("turbo_model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "ComfyWan_Architect/Performance"

    def apply_teacache(self, model, enable_tea_cache, rel_l1_threshold, start_step_percent, force_autocast):
        if not enable_tea_cache: return (model,)
        
        m = model.clone()
        m.wan_teacache_state = WanState()
        device = mm.get_torch_device()
        autocast_dtype = torch.float16
        if force_autocast and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        
        def teacache_wrapper(model_function, params):
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})
            
            flow_id = "default"
            if "context" in c and isinstance(c["context"], torch.Tensor):
                flow_id = c["context"].data_ptr()
            elif "y" in c and isinstance(c["y"], torch.Tensor):
                flow_id = c["y"].data_ptr()
            else:
                flow_id = f"fallback_{len(c.keys())}"

            if flow_id not in m.wan_teacache_state.flows:
                m.wan_teacache_state.flows[flow_id] = WanSubState()
            
            current_state = m.wan_teacache_state.flows[flow_id]

            def run_model(x, t, **kwargs):
                if force_autocast and not m.wan_teacache_state.autocast_failed_once:
                    try:
                        with torch.autocast(device.type, dtype=autocast_dtype):
                            return model_function(x, t, **kwargs)
                    except RuntimeError as e:
                        if "scalartype" in str(e).lower() or "autocast" in str(e).lower():
                            if not m.wan_teacache_state.debug_printed:
                                print(f"🛡️ [Wan Omega] FP8/Quantization conflict. Disabling Autocast.")
                            m.wan_teacache_state.autocast_failed_once = True
                            return model_function(x, t, **kwargs)
                        raise e
                else:
                    return model_function(x, t, **kwargs)

            if current_state.prev_latent is None: current_state.step_counter = 0
            
            # HARD LOCK: 2 steps minimum
            if current_state.step_counter < 2:
                current_state.prev_latent = input_x.detach()
                result = run_model(input_x, timestep, **c)
                current_state.prev_output = result
                current_state.step_counter += 1
                return result

            if input_x.shape != current_state.prev_latent.shape:
                current_state.prev_latent = input_x.detach()
                result = run_model(input_x, timestep, **c)
                current_state.prev_output = result
                current_state.step_counter += 1
                return result

            dims = input_x.shape
            spatial_res = dims[-1] * dims[-2]
            stride = 4 if spatial_res > 262144 else 2
            
            try:
                current_slice = input_x[..., ::stride, ::stride].to(torch.float32)
                prev_slice = current_state.prev_latent[..., ::stride, ::stride].to(torch.float32)
                diff = F.l1_loss(current_slice, prev_slice, reduction='mean')
                
                if diff < rel_l1_threshold:
                    current_state.skipped_steps += 1
                    current_state.step_counter += 1
                    return current_state.prev_output
            except: pass

            result = run_model(input_x, timestep, **c)
            current_state.prev_latent = input_x.detach()
            current_state.prev_output = result
            current_state.step_counter += 1
            return result

        m.set_model_unet_function_wrapper(teacache_wrapper)
        print(f"🚀 [Wan Turbo V5] Chronos Sentinel. Threshold: {rel_l1_threshold}")
        return (m,)

class Wan_Hybrid_VRAM_Guard:
    """
    OMEGA EDITION V6: Tesseract Engine.
    
    OPTIMISATIONS :
    1. Pipeline Asynchrone : Décode le Chunk N+1 pendant que le Chunk N se copie sur la RAM.
    2. Zero-Lag GC : Suppression du Garbage Collector dans la boucle critique.
    3. Tiling Safe : Allocation mémoire protégée.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "tile_size_spatial": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "temporal_chunk_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "enable_cpu_offload": ("BOOLEAN", {"default": True, "tooltip": "Transférer la RAM au fur et à mesure. Vital pour les vidéos 8K."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_hybrid"
    CATEGORY = "ComfyWan_Architect/Performance"
    
    def decode_hybrid(self, vae, samples, tile_size_spatial, temporal_chunk_size, enable_cpu_offload):
        latents = samples["samples"]
        shape = latents.shape 
        # (B, C, T, H, W) ou (B, C, H, W)
        temporal_dim = shape[2] if len(shape) == 5 else 1
        
        # --- 1. CONFIGURATION VAE ---
        try:
            vae.first_stage_model.enable_tiling(True)
            vae.first_stage_model.tile_sample_min_size = tile_size_spatial
            # Optimisation: 1/8 est standard pour SD, Wan supporte parfois 1/4 mais restons safe
            vae.first_stage_model.tile_latent_min_size = int(tile_size_spatial // 8)
            vae.first_stage_model.tile_overlap_factor = 0.25 
        except: pass 

        device = mm.get_torch_device()
        # Wan aime le BFloat16, c'est plus rapide et consomme moins
        work_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        total_frames = temporal_dim if len(shape) == 5 else shape[0]
        
        # --- 2. NETTOYAGE PRÉVENTIF (PAS DANS LA BOUCLE) ---
        mm.soft_empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()

        final_tensor = None
        current_write_idx = 0
        
        # Création d'un stream dédié pour le transfert mémoire (Pipeline)
        transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        pbar = comfy.utils.ProgressBar(total_frames)
        print(f"💠 [Wan Tesseract] Decoding {total_frames} frames. Async Transfer: {'ON' if transfer_stream else 'OFF'}")

        try:
            for i in range(0, total_frames, temporal_chunk_size):
                end_i = min(i + temporal_chunk_size, total_frames)
                
                # A. Chargement GPU
                # On utilise non_blocking=True pour ne pas bloquer le CPU s'il prépare déjà la suite
                if len(shape) == 5: 
                    chunk_latents = latents[:, :, i:end_i, :, :].to(device, non_blocking=True)
                else: 
                    chunk_latents = latents[i:end_i].to(device, non_blocking=True)
                
                # B. Décodage (Peut prendre du temps)
                try:
                    with torch.autocast(device.type, dtype=work_dtype):
                        if hasattr(vae, "decode_tiled"):
                            chunk_image = vae.decode_tiled(chunk_latents, tile_x=tile_size_spatial, tile_y=tile_size_spatial)
                        else:
                            chunk_image = vae.decode(chunk_latents)
                except torch.cuda.OutOfMemoryError:
                    print(f"♻️ [Wan Guard] OOM. Emergency flush.")
                    mm.soft_empty_cache()
                    chunk_image = vae.decode(chunk_latents)

                # C. Correction Dimensions
                if len(chunk_image.shape) == 5: chunk_image = chunk_image.squeeze(0)
                
                # D. Allocation Unique (Au premier tour)
                if final_tensor is None:
                    out_frames_chunk = chunk_image.shape[0]
                    input_chunk_len = end_i - i
                    ratio = out_frames_chunk / input_chunk_len
                    total_out_frames = int(total_frames * ratio)
                    
                    # PIN_MEMORY=TRUE est CRITIQUE pour la vitesse, mais dangereux si RAM faible.
                    # On tente le coup, c'est ce qui évite le lag.
                    try:
                        final_tensor = torch.empty(
                            (total_out_frames, chunk_image.shape[1], chunk_image.shape[2], chunk_image.shape[3]), 
                            dtype=torch.float32, 
                            device="cpu", 
                            pin_memory=enable_cpu_offload
                        )
                    except:
                        # Fallback si pas assez de RAM pour pinner
                        print("⚠️ [Wan Guard] RAM Low. Disabling Pin Memory.")
                        final_tensor = torch.empty(
                            (total_out_frames, chunk_image.shape[1], chunk_image.shape[2], chunk_image.shape[3]), 
                            dtype=torch.float32, 
                            device="cpu"
                        )

                # E. Transfert Asynchrone (Le Secret de la fluidité)
                out_f = chunk_image.shape[0]
                
                if enable_cpu_offload and transfer_stream:
                    # On attend que le GPU ait fini de calculer ce chunk
                    torch.cuda.current_stream().synchronize()
                    
                    with torch.cuda.stream(transfer_stream):
                        target_slice = final_tensor[current_write_idx : current_write_idx + out_f]
                        # Copie non bloquante vers la RAM Pinned
                        target_slice.copy_(chunk_image, non_blocking=True)
                else:
                    # Fallback Synchrone (Lent mais sûr)
                    final_tensor[current_write_idx : current_write_idx + out_f] = chunk_image.to("cpu")

                current_write_idx += out_f
                pbar.update(end_i - i)
                
                # F. Nettoyage Léger (SANS GC.COLLECT)
                del chunk_image
                del chunk_latents
                # Note: On ne force PAS le garbage collector ici. C'est ça qui fait ramer.
                # On laisse Python gérer sa mémoire, sauf si OOM critique.

            # Synchronisation finale une fois tout fini
            if transfer_stream: transfer_stream.synchronize()
            
        except Exception as e:
            print(f"❌ [Wan Guard Error] {e}")
            # Sauvetage de ce qu'on a pu décoder
            if final_tensor is not None: return (final_tensor[:current_write_idx],)
            raise e

        return (final_tensor[:current_write_idx],)

NODE_CLASS_MAPPINGS = {
    "Wan_TeaCache_Patch": Wan_TeaCache_Patch,
    "Wan_Hybrid_VRAM_Guard": Wan_Hybrid_VRAM_Guard
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_TeaCache_Patch": "Wan Turbo (TeaCache Omega V5)",
    "Wan_Hybrid_VRAM_Guard": "Wan Decode (VRAM Guard Omega V6)"
}
