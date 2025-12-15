import torch
import numpy as np
from skimage.exposure import match_histograms

class Wan_Chroma_Mimic:
    """
    Wan Chroma Mimic (T-1000 Technology).
    Module de camouflage colorimétrique.
    Adapte l'histogramme de la vidéo générée pour correspondre parfaitement à l'image source.
    Intègre une protection Nano-Shield contre les artefacts noirs (NaNs).
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # La vidéo cible
                "reference_image": ("IMAGE",),  # L'image de référence (Style)
                "blend_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "1.0 = Copie totale des couleurs. 0.5 = Mélange 50/50."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_images",)
    FUNCTION = "apply_chroma_mimic"
    CATEGORY = "Wan_Architect/Skynet"

    def apply_chroma_mimic(self, images, reference_image, blend_factor):
        # 1. Sécurisation de l'image de référence (Nano-Shield)
        # On remplace les valeurs invalides par 0 et on force entre 0 et 1
        ref_tensor = torch.nan_to_num(reference_image[0], nan=0.0, posinf=1.0, neginf=0.0)
        ref_tensor = torch.clamp(ref_tensor, 0.0, 1.0)
        ref_np = ref_tensor.cpu().numpy()

        result_images = []
        total_frames = len(images)
        
        print(f"\n>> [CHROMA MIMIC] Mimicking colors on {total_frames} frames...")

        for i, img in enumerate(images):
            # 2. Sécurisation de l'image source (Vidéo)
            img_clean = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            img_clean = torch.clamp(img_clean, 0.0, 1.0)
            
            target_np = img_clean.cpu().numpy()

            # Application du Match Histogram (Scikit-Image)
            try:
                # channel_axis=-1 indique que le dernier axe est RGB
                matched = match_histograms(target_np, ref_np, channel_axis=-1)
            except Exception as e:
                print(f"!! [Mimic Error] Frame {i} failed, skipping correction: {e}")
                matched = target_np

            # Mélange progressif (Morphing)
            if blend_factor < 1.0:
                matched = (matched * blend_factor) + (target_np * (1.0 - blend_factor))

            # 3. Sécurisation finale du résultat
            # On s'assure qu'après les calculs, on n'a pas créé de nouvelles aberrations
            matched = np.nan_to_num(matched, nan=0.0, posinf=1.0, neginf=0.0)
            matched = np.clip(matched, 0.0, 1.0)

            result_images.append(matched)

        # Reconversion Numpy -> Tensor ComfyUI
        result_tensor = torch.from_numpy(np.array(result_images)).float()
        
        print(f">> [CHROMA MIMIC] Camouflage complete.\n")
        return (result_tensor,)

# Mapping (Déjà géré dans __init__.py, mais présent ici pour référence)
NODE_CLASS_MAPPINGS = {
    "Wan_Chroma_Mimic": Wan_Chroma_Mimic
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Chroma_Mimic": "Wan Chroma Mimic (Color Match)"
}