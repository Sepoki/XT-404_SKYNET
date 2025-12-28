import torch

class Wan_Spectre_Chroma_Filter:
    """
    MODULE D : SPECTRE-CHROMA (Luminance Lock Edition)
    Architecture : Corrige UNIQUEMENT la teinte (A/B).
    GARANTIE : Ne touche JAMAIS à la luminosité (L). Empêche le blanchiment/grisaillement.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stabilize_color"
    CATEGORY = "XT-404/V2_Omega"

    def rgb_to_lab(self, img):
        # Conversion RGB -> LAB haute précision
        matrix = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], device=img.device)
        
        # Clamp pour éviter les NaNs
        img = torch.clamp(img, 1e-6, 1.0 - 1e-6)
        
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
        return torch.clamp(rgb, 0.0, 1.0)

    def stabilize_color(self, images, strength):
        # Gestion 4D/5D
        original_shape = images.shape
        if images.ndim == 5:
            images = images.view(-1, *images.shape[-3:])

        if strength == 0: 
            if len(original_shape) == 5: return (images.view(original_shape),)
            return (images,)
        
        device = images.device
        
        # 1. Conversion LAB
        tgt_lab = self.rgb_to_lab(images.to(device))
        
        # 2. Référence (Frame 0) - On ne garde QUE la moyenne de teinte
        ref_a = tgt_lab[0, ..., 1].mean()
        ref_b = tgt_lab[0, ..., 2].mean()
        
        # 3. Moyennes courantes
        curr_a = tgt_lab[..., 1].mean(dim=(1, 2), keepdim=True)
        curr_b = tgt_lab[..., 2].mean(dim=(1, 2), keepdim=True)
        
        # 4. Calcul du Delta (Dérive)
        delta_a = ref_a - curr_a
        delta_b = ref_b - curr_b
        
        # 5. PROTECTION INTELLIGENTE (Skin & Luma Mask)
        # On détecte la peau (Rouge > Vert > Bleu)
        r, g, b = images[..., 0], images[..., 1], images[..., 2]
        is_skin = (r > g) & (g > b)
        
        # Masque Peau : Si c'est de la peau, on réduit la force de 80%
        # Cela empêche la peau de devenir grise/verte
        skin_protection = 1.0 - (is_skin.float() * 0.8)
        
        # Masque Luminosité : On ne touche pas aux zones très claires (blancs) ou très sombres
        l = tgt_lab[..., 0] / 100.0 # 0..1
        # Courbe en cloche : 1.0 au milieu (gris), 0.0 aux extrêmes (noir/blanc)
        luma_protection = 1.0 - torch.pow(2.0 * l - 1.0, 4.0)
        
        # Masque final combiné
        final_mask = (skin_protection * luma_protection).unsqueeze(-1)
        
        # 6. Application
        # On applique la correction pondérée
        new_a = tgt_lab[..., 1] + (delta_a * strength * final_mask.squeeze(-1))
        new_b = tgt_lab[..., 2] + (delta_b * strength * final_mask.squeeze(-1))
        
        # 7. RECONSTRUCTION AVEC L ORIGINAL (Le Secret)
        # On reprend EXPLICITEMENT le canal L d'origine (tgt_lab[..., 0])
        # Ainsi, la luminosité ne change pas d'un iota.
        final_lab = torch.stack([tgt_lab[..., 0], new_a, new_b], dim=-1)
        final_rgb = self.lab_to_rgb(final_lab)
        
        # Frame 0 Intouchée
        final_rgb[0] = images[0]
        
        # Restauration format
        if len(original_shape) == 5:
            final_rgb = final_rgb.view(original_shape)
            
        return (final_rgb,)
