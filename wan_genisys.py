import torch
import torch.nn.functional as F

class Wan_Cyberdyne_Genisys:
    """
    **Wan 2.2 CYBERDYNE GENISYS [NANO-REPAIR CORE]**
    Version: OMEGA PRIME (No-Cache / Pure Safety)
    Role: Protection active du signal (Anti-NaN + TF32 Clamp Fix).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "system_status": ("BOOLEAN", {"default": True, "label": "NANO-REPAIR ACTIVE"}),
                "fix_range": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Amplitude max du signal (Clamp). Wan aime +/- 6.0 à 8.0."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("T3000_Model",)
    FUNCTION = "deploy_genisys"
    CATEGORY = "Wan_Architect/Skynet"

    def deploy_genisys(self, model, system_status, fix_range):
        if not system_status:
            return (model,)

        m = model.clone()

        def t3000_protocol(model_function, params):
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})

            # --- [LAYER 1] PROTECTION D'ENTRÉE ---
            # Nettoyage préventif des NaNs dans le latent d'entrée
            if torch.isnan(input_x).any() or torch.isinf(input_x).any():
                input_x = torch.nan_to_num(input_x, nan=0.0, posinf=0.0, neginf=0.0)

            # Exécution du modèle (Pas de cache, calcul réel obligatoire)
            out = model_function(input_x, timestep, **c)

            # --- [LAYER 2] PROTECTION DE SORTIE (TF32 FIX) ---
            # Le problème de "brûlure" vient de valeurs extrêmes générées par le TF32.
            # On clamp le résultat pour le garder dans une plage de diffusion valide.
            
            # 1. Check rapide (Optimisation: on ne corrige que si nécessaire)
            # Note: Sur GPU, un check .any() est rapide mais force une synchro partielle. 
            # Pour la sécurité max, on applique le nan_to_num systématiquement ou on clamp.
            # Le Clamp est peu coûteux et résout 99% des artefacts noirs de Wan.
            
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = torch.clamp(out, min=-fix_range, max=fix_range)

            return out

        m.set_model_unet_function_wrapper(t3000_protocol)
        print(f"\033[96m[GENISYS PRIME]\033[0m Nano-Repair Protocols Engaged. Range: +/-{fix_range}")
        return (m,)

NODE_CLASS_MAPPINGS = { "Wan_Cyberdyne_Genisys": Wan_Cyberdyne_Genisys }
NODE_DISPLAY_NAME_MAPPINGS = { "Wan_Cyberdyne_Genisys": "💀 Cyberdyne Genisys [NANO-REPAIR]" }
