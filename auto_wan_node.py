import torch
import torch.nn.functional as F
import comfy.model_management

class AutoWanImageOptimizer:
    """
    Nœud optimisé pour WAN 2.2 Image-to-Video.
    Gère automatiquement :
    - La réduction par 2 (standard).
    - La limite OOM (Max 1024px).
    - Le seuil minimum (Min 512px).
    - Le Modulo 16 (Compatible Vidéo).
    - Le maintien strict du Ratio (Aspect Ratio) quel que soit le format (Portrait, Paysage, Cowboy, etc.).
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "debug_info")
    FUNCTION = "process_image"
    CATEGORY = "Custom/Wan2.2"

    def process_image(self, image):
        # 1. Préparation de l'appareil (GPU/CPU)
        device = comfy.model_management.get_torch_device()
        
        # Format ComfyUI : [Batch, Hauteur, Largeur, Channels] -> Permutation pour PyTorch : [Batch, Channels, Hauteur, Largeur]
        samples = image.movedim(-1, 1).to(device)
        
        batch_size, channels, current_h, current_w = samples.shape
        
        # --- CONSTANTES DE CONFIGURATION ---
        MAX_DIM = 1024  # Protection OOM (Out of Memory)
        MIN_DIM = 512   # Protection Qualité Minimum
        MODULO = 16     # Protection Vidéo (Encodage)

        # --- LOGIQUE DE CALCUL INTELLIGENT ---
        
        target_h = float(current_h)
        target_w = float(current_w)
        max_side = max(target_h, target_w)

        # Étape A : Protection OOM (Priorité Absolue)
        # Si l'image est géante (> 1024), on la force à 1024 max.
        if max_side > MAX_DIM:
            scale_factor = MAX_DIM / max_side
            target_h = target_h * scale_factor
            target_w = target_w * scale_factor
            action_log = "OOM Protection Triggered (Max 1024)"
        
        # Étape B : Comportement Standard (Diviser par 2)
        # Si l'image n'était pas géante, on applique la division par 2 demandée.
        else:
            target_h = target_h / 2.0
            target_w = target_w / 2.0
            action_log = "Standard Division (1/2)"

        # Étape C : Sécurité Minimum
        # Si après réduction, un côté est inférieur à 512, on redimensionne pour atteindre 512 min.
        min_side = min(target_h, target_w)
        if min_side < MIN_DIM:
            scale_fix = MIN_DIM / min_side
            target_h = target_h * scale_fix
            target_w = target_w * scale_fix
            
            # Sous-sécurité : Si remonter à 512 fait dépasser 1024 (ex: image très panoramique), on re-plafonne.
            if max(target_h, target_w) > MAX_DIM:
                scale_cap = MAX_DIM / max(target_h, target_w)
                target_h *= scale_cap
                target_w *= scale_cap
            
            action_log += " + Min Size Fix Applied"

        # Étape D : Application du Modulo 16 (Arrondi intelligent)
        # Indispensable pour que Wan 2.2 ne plante pas.
        final_h = int(round(target_h / MODULO) * MODULO)
        final_w = int(round(target_w / MODULO) * MODULO)

        # --- REDIMENSIONNEMENT HAUTE QUALITÉ ---
        
        # Utilisation de Bicubic + Antialias pour une image "absolument parfaite".
        # C'est mieux que 'Area' pour préserver les détails fins lors de réductions modérées.
        resized_samples = F.interpolate(
            samples, 
            size=(final_h, final_w), 
            mode="bicubic", 
            align_corners=False, 
            antialias=True
        )
        
        # Retour au format CPU et ComfyUI
        output_image = resized_samples.movedim(1, -1).cpu()
        
        # Création du log de debug
        info_string = f"Original: {current_w}x{current_h} | Result: {final_w}x{final_h} | Mode: {action_log}"
        print(f"\033[96m[AutoWanNode] {info_string}\033[0m")

        return (output_image, final_w, final_h, info_string)

# Mappings
NODE_CLASS_MAPPINGS = {
    "AutoWanImageOptimizer": AutoWanImageOptimizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoWanImageOptimizer": "Auto Wan 2.2 Optimizer (Safe Resize)"
}