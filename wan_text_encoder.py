import torch
import comfy.model_management as mm
import hashlib
import gc
from collections import OrderedDict

class Wan_Text_OneShot_Cache:
    """V6 HPC: Pinned Memory Cache."""
    _cache = OrderedDict()
    CACHE_LIMIT = 20
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "aggressive_offload": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode_oneshot"
    CATEGORY = "ComfyWan_Architect/Performance"

    def encode_oneshot(self, clip, text, aggressive_offload):
        unique_key = f"{text}_{id(clip)}"
        prompt_hash = hashlib.md5(unique_key.encode('utf-8')).hexdigest()
        
        if prompt_hash in self._cache:
            print(f">> [Wan Text] Cache Hit (Pinned).")
            self._cache.move_to_end(prompt_hash)
            cpu_data = self._cache[prompt_hash]
            device = mm.get_torch_device()
            
            # Transfert DMA rapide (si pinned)
            gpu_tensor = cpu_data[0].to(device, non_blocking=True)
            gpu_pooled = cpu_data[1].to(device, non_blocking=True) if cpu_data[1] is not None else None
            
            return ([[gpu_tensor, {"pooled_output": gpu_pooled}]],)
        
        print(f">> [Wan Text] Encoding...")
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        # P2 Optimisation: Pinned Memory
        # Note: pin_memory() ne marche que sur des tenseurs CPU
        cpu_cond = cond.to("cpu").pin_memory()
        cpu_pooled = pooled.to("cpu").pin_memory() if pooled is not None else None
        
        self._cache[prompt_hash] = (cpu_cond, cpu_pooled)
        if len(self._cache) > self.CACHE_LIMIT:
            self._cache.popitem(last=False)
        
        current_conditioning = [[cond, {"pooled_output": pooled}]]
        
        if aggressive_offload:
            # Optimisation Soft: On ne purge que si nécessaire
            try:
                if hasattr(clip, "patcher"): clip.patcher.model.to("cpu")
                elif hasattr(clip, "model"): clip.model.to("cpu")
            except: pass
            
            # On évite soft_empty_cache systématique si VRAM ok
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0]
                if free_mem < 2 * (1024**3): # Si < 2GB libre
                    mm.soft_empty_cache()

        return (current_conditioning,)