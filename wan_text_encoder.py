import torch
import comfy.model_management as mm
import hashlib
import gc
from collections import OrderedDict

class Wan_Text_OneShot_Cache:
    """
    OMEGA EDITION: Pinned Memory & DMA Transfer.
    Cache textuel intelligent qui utilise la RAM système verrouillée pour des transferts GPU instantanés.
    """
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
        # Génération d'un hash unique incluant l'ID du modèle CLIP pour éviter les collisions
        unique_key = f"{text}_{id(clip)}"
        prompt_hash = hashlib.md5(unique_key.encode('utf-8')).hexdigest()
        
        device = mm.get_torch_device()

        # 1. Vérification du Cache
        if prompt_hash in self._cache:
            print(f"💾 [Wan Text] Cache Hit via DMA.")
            self._cache.move_to_end(prompt_hash)
            cpu_data = self._cache[prompt_hash]
            
            # Transfert asynchrone (non_blocking=True) possible grâce au pin_memory
            gpu_tensor = cpu_data[0].to(device, non_blocking=True)
            gpu_pooled = cpu_data[1].to(device, non_blocking=True) if cpu_data[1] is not None else None
            
            return ([[gpu_tensor, {"pooled_output": gpu_pooled}]],)
        
        # 2. Encodage
        print(f"🔤 [Wan Text] Encoding new prompt...")
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        # 3. Optimisation Mémoire (Pinned RAM)
        try:
            cpu_cond = cond.to("cpu").pin_memory()
            cpu_pooled = pooled.to("cpu").pin_memory() if pooled is not None else None
        except Exception:
            # Fallback si pin_memory échoue (ex: RAM saturée)
            cpu_cond = cond.to("cpu")
            cpu_pooled = pooled.to("cpu") if pooled is not None else None
        
        # Gestion de la taille du cache
        self._cache[prompt_hash] = (cpu_cond, cpu_pooled)
        if len(self._cache) > self.CACHE_LIMIT:
            self._cache.popitem(last=False)
        
        # 4. Offloading
        if aggressive_offload:
            try:
                if hasattr(clip, "patcher"): clip.patcher.model.to("cpu")
                elif hasattr(clip, "model"): clip.model.to("cpu")
            except: pass
            
            # Nettoyage intelligent : Uniquement si la VRAM est sous pression (< 2GB)
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0]
                if free_mem < 2 * (1024**3): 
                    mm.soft_empty_cache()

        return ([[cond, {"pooled_output": pooled}]],)

NODE_CLASS_MAPPINGS = {
    "Wan_Text_OneShot_Cache": Wan_Text_OneShot_Cache
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Text_OneShot_Cache": "Wan Text Cache (Omega)"
}
