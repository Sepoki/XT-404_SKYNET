import torch
import comfy.model_management as mm
import torch.nn.functional as F

class Wan_Hardware_Accelerator:
    """
    OMEGA EDITION: TF32 Global Activation Control.
    ATTENTION : Activer TF32 change l'état global de PyTorch.
    Gain de vitesse : ~30% sur Ampere+. Perte de précision : Minime (Mantisse 10-bit).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_tf32": ("BOOLEAN", {"default": True, "tooltip": "Active TensorFloat-32. Rapide mais réduit légèrement la précision."}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True, "tooltip": "Optimise les algos de convolution pour les tailles fixes."}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("accelerated_model",)
    FUNCTION = "apply_acceleration"
    CATEGORY = "ComfyWan_Architect/Performance"

    def apply_acceleration(self, model, enable_tf32, cudnn_benchmark):
        # 1. Gestion TF32 (Global State)
        if torch.cuda.is_available():
            current_matmul = torch.backends.cuda.matmul.allow_tf32
            current_cudnn = torch.backends.cudnn.allow_tf32
            
            if enable_tf32:
                if not current_matmul or not current_cudnn:
                    print(f"⚡ [Wan Accel] OMEGA ACTIVATED: TF32 Enabled (Speed Up). Precision reduced slightly.")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                # On ne désactive que si l'utilisateur le demande explicitement
                if current_matmul:
                    print(f"🛡️ [Wan Accel] OMEGA SAFETY: TF32 Disabled. Maximum Precision enforced.")
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        
        # 2. Benchmark CuDNN
        if cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        m = model.clone()
        return (m,)

class Wan_Attention_Slicer:
    """
    OMEGA EDITION: Smart Attention Management.
    Priorité : SDPA (Flash Attention) > Slicing manuel.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "slice_size": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": "0 = Auto (Flash Attention). 1-8 = Force Slicing (Low VRAM)."}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_attention"
    CATEGORY = "ComfyWan_Architect/Performance"
    
    def patch_attention(self, model, slice_size):
        m = model.clone()
        
        # Détection SDPA (Scaled Dot Product Attention)
        has_sdpa = hasattr(F, "scaled_dot_product_attention")
        
        current_options = m.model_options.get("transformer_options", {}).copy()

        if slice_size == 0:
            # MODE AUTO
            if has_sdpa:
                print(f"🚀 [Wan Accel] Using Native PyTorch SDPA (Flash Attention).")
                # Nettoyage des contraintes de slicing pour laisser faire SDPA
                if "attention_slice_size" in current_options:
                    del current_options["attention_slice_size"]
                m.model_options["transformer_options"] = current_options
            else:
                # Fallback pour vieux PyTorch
                print(f"⚠️ [Wan Accel] SDPA not found. Fallback to Slicing (8).")
                current_options["attention_slice_size"] = 8
                m.model_options["transformer_options"] = current_options
        else:
            # MODE FORCÉ (Low VRAM)
            print(f"📉 [Wan Accel] Forcing Attention Slice: {slice_size}. (VRAM Saved / Speed Reduced)")
            current_options["memory_efficient_attention"] = True
            current_options["attention_slice_size"] = slice_size
            m.model_options["transformer_options"] = current_options
            
        return (m,)

NODE_CLASS_MAPPINGS = {
    "Wan_Hardware_Accelerator": Wan_Hardware_Accelerator,
    "Wan_Attention_Slicer": Wan_Attention_Slicer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Hardware_Accelerator": "Wan Hardware Accelerator (Omega)",
    "Wan_Attention_Slicer": "Wan Attention Strategy (Omega)"
}
