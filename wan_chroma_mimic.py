import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
import sys

# ==============================================================================
# ARCHITECTURE: XT-404 OMEGA | CHROMA MIMIC SENTINEL
# VERSION: 3.2 (PROGRESSIVE TRACKING & VRAM SAFETY)
# ROLE: Color Matching + Signal Validation + RIFE Correction + Real-Time Feedback
# ==============================================================================

class XT_Mouchard_Mimetic:
    """
    Module de Télémétrie dédié à la Colorimétrie.
    Vérifie la conformité du Gamut et la plage dynamique.
    """
    HEADER = "\033[96m[XT-MIMIC]\033[0m"
    RESET = "\033[0m"
    
    @staticmethod
    def analyze_signal(tag, tensor):
        # Analyse rapide sur un échantillon si le tenseur est énorme
        if tensor.shape[0] > 10:
            sample = tensor[::10] # 1 frame sur 10 pour aller vite
        else:
            sample = tensor

        mean = sample.mean().item()
        min_val = sample.min().item()
        max_val = sample.max().item()
        std_color = sample.std(dim=-1).mean().item()
        
        clipped_high = (sample >= 0.999).sum().item()
        total_px = sample.numel()
        clip_pct = (clipped_high / total_px) * 100.0
        
        status = "\033[92mOK\033[0m"
        warn = ""
        
        if clip_pct > 1.0: 
            status = "\033[93mWARN\033[0m"
            warn = " -> \033[93mHigh Signal Detected (Rolloff Active)\033[0m"
        if clip_pct > 5.0:
            status = "\033[91mCRITICAL\033[0m"
            warn = " -> \033[91mSignal Clipping! Auto-Correction Engaged.\033[0m"

        print(f"{XT_Mouchard_Mimetic.HEADER} 🎨 {tag} | DynRange: [{min_val:.3f}, {max_val:.3f}] | SaturationIndex: {std_color:.3f}")
        print(f"   └── Signal Integrity: {status} (Clip: {clip_pct:.2f}%){warn}")

class Wan_Chroma_Mimic:
    """
    Wan Chroma Mimic - SENTINEL V3.2.
    Intègre un traitement par paquets (Chunks) pour afficher la progression
    et protéger la VRAM sur les rendus longs.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                
                "effect_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                "smart_limit": ("BOOLEAN", {"default": True, "tooltip": "Empêche le transfert de couleur de brûler l'image."}),
                "rife_correction": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Récupère le piqué perdu par l'interpolation."}),
                
                "oled_contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "detail_crispness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("oled_master_video",)
    FUNCTION = "apply_mimic_sentinel"
    CATEGORY = "Wan_Architect/Skynet"

    def rgb_to_lab(self, img):
        matrix = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], device=img.device)
        xyz = torch.matmul(img, matrix.T)
        mask = xyz > 0.008856
        xyz[mask] = torch.pow(xyz[mask], 1/3)
        xyz[~mask] = 7.787 * xyz[~mask] + 16/116
        l = 116 * xyz[..., 1] - 16
        a = 500 * (xyz[..., 0] - xyz[..., 1])
        b = 200 * (xyz[..., 1] - xyz[..., 2])
        return torch.stack([l, a, b], dim=-1)

    def lab_to_rgb(self, lab):
        l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        xyz = torch.stack([x, y, z], dim=-1)
        mask = xyz > 0.206893
        xyz[mask] = torch.pow(xyz[mask], 3)
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787
        matrix = torch.tensor([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875992, 0.041556],
            [0.055648, -0.204043, 1.057311]
        ], device=lab.device)
        rgb = torch.matmul(xyz, matrix.T)
        return rgb 

    def get_moments(self, tensor):
        mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
        std = torch.std(tensor, dim=(1, 2), keepdim=True)
        return mean, std

    def apply_mimic_sentinel(self, images, reference_image, effect_intensity, smart_limit, rife_correction, oled_contrast, detail_crispness):
        
        device = images.device
        batch_size = images.shape[0]
        
        XT_Mouchard_Mimetic.analyze_signal("INPUT CHECK", images)
        print(f"\033[96m[XT-MIMIC] Starting Real-Time Processing on {batch_size} frames...\033[0m")

        # 1. ANALYSE REFERENCE (Fait une seule fois pour tout le lot)
        ref_clean = torch.nan_to_num(reference_image[0:1], nan=0.0).to(device)
        ref_lab = self.rgb_to_lab(ref_clean)
        ref_mean, ref_std = self.get_moments(ref_lab)
        
        # Gamut Sentry Global
        if smart_limit:
            saturation_check = ref_std[..., 1:] 
            if saturation_check.mean() > 40.0:
                ref_std = ref_std * 0.8
                print(f"   🛡️ \033[33m[Gamut Sentry]\033[0m Reference too saturated. Dampening.")

        # 2. BOUCLE DE TRAITEMENT PAR CHUNKS (PROGRESSION)
        # On découpe par paquets de 16 frames pour ne pas saturer la VRAM
        chunk_size = 16 
        processed_chunks = []
        
        # Initialisation de la barre de progression ComfyUI
        pbar = comfy.utils.ProgressBar(batch_size)
        
        for i in range(0, batch_size, chunk_size):
            # Extraction du paquet
            chunk = images[i : i + chunk_size].to(device)
            current_batch_count = chunk.shape[0]
            
            # --- LOGIQUE COULEUR (Sur le paquet) ---
            tgt_lab = self.rgb_to_lab(chunk)
            tgt_mean, tgt_std = self.get_moments(tgt_lab)
            
            normalized = (tgt_lab - tgt_mean) / (tgt_std + 1e-6)
            mimic_lab = normalized * ref_std + ref_mean
            final_lab = torch.lerp(tgt_lab, mimic_lab, effect_intensity)
            final_rgb = self.lab_to_rgb(final_lab)
            
            # --- LOGIQUE SENTINEL (Sur le paquet) ---
            # A. Dampened RIFE Correction
            if rife_correction > 0 or detail_crispness > 0:
                total_sharpen = rife_correction + detail_crispness
                blurred = F.avg_pool2d(final_rgb.movedim(-1, 1), 3, stride=1, padding=1).movedim(1, -1)
                raw_details = final_rgb - blurred
                
                limit_factor = 0.10
                dampened_details = torch.tanh(raw_details / limit_factor) * limit_factor
                
                luma_mask = 1.0 - torch.pow(2.0 * final_rgb.mean(dim=-1, keepdim=True) - 1.0, 4.0)
                final_rgb = final_rgb + (dampened_details * total_sharpen * luma_mask)

            # B. OLED Contrast
            if oled_contrast > 0:
                final_rgb = final_rgb - 0.5
                final_rgb = final_rgb * (1.0 + oled_contrast)
                final_rgb = final_rgb + 0.5

            # C. Smart Limiter
            if smart_limit:
                threshold = 0.95
                mask_high = (final_rgb > threshold).float()
                delta = final_rgb - threshold
                compressed = threshold + (delta / (1.0 + delta * 4.0)) * (1.0 - threshold)
                final_rgb = final_rgb * (1.0 - mask_high) + compressed * mask_high
                final_rgb = torch.clamp(final_rgb, 0.001, 1.0)
            else:
                final_rgb = torch.clamp(final_rgb, 0.0, 1.0)

            final_rgb = torch.nan_to_num(final_rgb, 0.0)
            
            # Stockage et Progression
            processed_chunks.append(final_rgb)
            pbar.update(current_batch_count)
            
            # Affichage console tous les 20% pour ne pas spammer
            percent = int(((i + current_batch_count) / batch_size) * 100)
            if percent % 20 == 0 or percent == 100:
                sys.stdout.write(f"\r\033[90m[XT-MIMIC] Processing: {percent}% ({i + current_batch_count}/{batch_size})\033[0m")
                sys.stdout.flush()

        print("") # Retour à la ligne propre
        
        # Assemblage final
        final_video = torch.cat(processed_chunks, dim=0)
        
        XT_Mouchard_Mimetic.analyze_signal("FINAL VALIDATION", final_video)

        return (final_video,)

# ==============================================================================
# MAPPINGS
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "Wan_Chroma_Mimic": Wan_Chroma_Mimic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Chroma_Mimic": "Wan Chroma Mimic (Progressive)"
}
