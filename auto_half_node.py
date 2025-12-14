import torch
import torch.nn.functional as F

class AutoHalfSizeImage:
    """
    Ce nœud prend une image en entrée, calcule ses dimensions,
    les divise par 2 et effectue un redimensionnement (Downscale)
    automatique sur le GPU.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # Entrée : Image originale
            }
        }

    # Sorties : L'image redimensionnée, mais aussi la nouvelle largeur/hauteur si besoin
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process_image"
    CATEGORY = "Custom/Image"

    def process_image(self, image):
        # L'image dans ComfyUI est sous la forme [Batch, Hauteur, Largeur, Channels]
        # On déplace les dimensions pour PyTorch : [Batch, Channels, Hauteur, Largeur]
        samples = image.movedim(-1, 1)
        
        # Récupération des dimensions actuelles
        current_h = samples.shape[2]
        current_w = samples.shape[3]
        
        # Calcule automatique : Division par 2 (division entière)
        new_h = current_h // 2
        new_w = current_w // 2
        
        # Redimensionnement
        # Utilisation de 'area' qui est la méthode mathématiquement la plus propre 
        # pour réduire une image (downscale) sans aliasing, ou 'bicubic' pour le lissage.
        # Ici on utilise 'bicubic' avec antialias pour un résultat esthétique optimal (GPU supporté).
        resized_samples = F.interpolate(
            samples, 
            size=(new_h, new_w), 
            mode="bicubic", 
            align_corners=False, 
            antialias=True
        )
        
        # On remet l'image au format ComfyUI [Batch, Hauteur, Largeur, Channels]
        output_image = resized_samples.movedim(1, -1)
        
        # On retourne l'image et les nouvelles dimensions
        return (output_image, new_w, new_h)

# Mapping pour le chargement
NODE_CLASS_MAPPINGS = {
    "AutoHalfSizeImage": AutoHalfSizeImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoHalfSizeImage": "Auto Image Half Size (1/2)"
}