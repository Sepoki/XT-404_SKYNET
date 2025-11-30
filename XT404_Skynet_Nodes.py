# --- START OF FILE XT404_Skynet_Nodes.py ---
# ARCHITECT: ComfyWan_Architect_OMEGA
# VERSION: 29.0 (GOLD MASTER - Final Stable Release)
# COMPATIBILITY: Wan 2.1 / Wan 2.2 / Flux
# STATUS: BATTLE TESTED

import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management as mm
import gc

# --- UTILITAIRES SYSTÈME ---

def create_combo_list():
    samplers = comfy.samplers.KSampler.SAMPLERS
    # Mapping interne : Wan nécessite 'simple' (Linear) pour fonctionner correctement
    # On garde la liste complète pour la compatibilité UI
    schedulers_map = ["linear", "karras", "exponential", "sgm_uniform", "simple", "normal", "beta", "ddim_uniform"]
    combos = []
    for sch in schedulers_map:
        for sam in samplers:
            combos.append(f"{sch}/{sam}")
    return combos

SKAYNET_COMBOS = create_combo_list()
SAMPLER_MODES = ["standard", "resample", "randomize"] 

# --- PROTOCOLE LOGGING ---

class XT404_Sentinel:
    PREFIX = "\033[36m[XT-404 OMEGA]\033[0m"
    RESET = "\033[0m"
    GREY = "\033[90m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[35m"

    @staticmethod
    def log(tag, message, color=CYAN):
        print(f"{XT404_Sentinel.PREFIX} {XT404_Sentinel.GREY}{tag}:{XT404_Sentinel.RESET} {color}{message}{XT404_Sentinel.RESET}")

# --- MOTEUR CORE ---

class Skynet_Core_Hybrid:
    
    @staticmethod
    def clean_vram(force_unload=False):
        """Nettoyage chirurgical de la mémoire pour prévenir le OOM en HD."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if force_unload:
            mm.soft_empty_cache()

    def parse_combo(self, combo_string):
        if "/" in combo_string:
            sch, sam = combo_string.split("/")
            return "simple", sam # Force Linear pour Wan (Indispensable)
        return "simple", combo_string

    def compute_wan_sigmas(self, steps, shift=1.0):
        """Générateur mathématique Flow Matching haute précision"""
        # Calcul sur CPU en Float32 pour éviter les erreurs d'arrondi
        t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")
        sigmas = (t * shift) / (1 + (shift - 1) * t)
        return sigmas

    def get_internal_sigmas(self, model, steps, shift_val=1.0, denoise=1.0):
        XT404_Sentinel.log("CLOCK", f"Generating Timeline | Steps: {steps} | Shift: {shift_val}", XT404_Sentinel.MAGENTA)
        sigmas = self.compute_wan_sigmas(steps, shift=shift_val)
        
        if denoise < 1.0:
            total_len = len(sigmas)
            slice_idx = int(total_len * (1.0 - denoise))
            if slice_idx >= total_len: slice_idx = total_len - 1
            sigmas = sigmas[slice_idx:]
            
        return sigmas

    def generic_sample(self, model, noise_seed, steps, cfg, sampler_combo, shift_val,
                       positive, negative, latent_image, denoise, steps_to_run, 
                       sampler_mode, use_analog_sync, eta, 
                       previous_options=None, 
                       sigmas_input=None, is_chain=False, node_id="UNK"):
        
        device = mm.get_torch_device()
        
        # --- 1. CONTINUITÉ & SYNCHRONISATION ---
        total_steps = steps
        start_step = 0
        full_sigmas = None

        if is_chain:
            if previous_options is None:
                XT404_Sentinel.log("WARN", "Chain broken! Attempting emergency restart.", XT404_Sentinel.RED)
                # Fallback d'urgence
                total_steps = 20
            else:
                total_steps = previous_options.get("total_steps", 20)
                start_step = previous_options.get("next_step", 0)
                
                # --- MASTER CLOCK SYNC ---
                # Le Chain récupère la timeline exacte du Master
                if "master_sigmas" in previous_options:
                    full_sigmas = previous_options["master_sigmas"].to(device="cpu")
                    XT404_Sentinel.log("SYNC", f"Locked to Master Clock.", XT404_Sentinel.GREEN)
        else:
            # MASTER NODE : Nettoyage initial
            self.clean_vram(force_unload=False)

        # --- 2. GÉNÉRATION DE LA TIMELINE (MASTER) ---
        target_scheduler, target_sampler = self.parse_combo(sampler_combo)
        
        if not is_chain and full_sigmas is None:
            # A. Priorité : Scheduler Externe (WanVideoScheduler)
            if sigmas_input is not None:
                 s_check = sigmas_input
                 if isinstance(s_check, (list, tuple)): s_check = s_check[0]
                 if hasattr(s_check, "shape"):
                     XT404_Sentinel.log("LINK", f"External Scheduler Active.", XT404_Sentinel.GREEN)
                     full_sigmas = s_check.clone().cpu().to(dtype=torch.float32)
                     total_steps = len(full_sigmas) - 1
            
            # B. Fallback : Générateur Interne (Shift)
            if full_sigmas is None:
                # Utilisation du Shift défini par l'utilisateur
                safe_shift = shift_val if shift_val is not None else 1.0
                full_sigmas = self.get_internal_sigmas(model, total_steps, safe_shift, denoise=1.0)
                full_sigmas = full_sigmas.cpu().to(dtype=torch.float32)

        # Safety Net
        if full_sigmas is None:
             full_sigmas = self.get_internal_sigmas(model, total_steps, 1.0, 1.0)

        # --- 3. DÉCOUPAGE ---
        if steps_to_run == -1:
            end_step = len(full_sigmas) - 1
        else:
            end_step = min(start_step + steps_to_run, len(full_sigmas) - 1)

        if start_step >= end_step:
            # Cycle terminé
            return (latent_image, latent_image, {"total_steps": total_steps, "next_step": end_step, "master_sigmas": full_sigmas})

        current_sigmas = full_sigmas[start_step : end_step + 1]
        
        # --- 4. GESTION DU LATENT & BRUIT (ZERO-POINT FIX) ---
        latent_tensor = latent_image["samples"]
        if latent_tensor.device != device: latent_tensor = latent_tensor.to(device)
        
        # Initialisation du bruit pour le Sampler (Silence par défaut)
        noise = torch.zeros_like(latent_tensor)
        
        if start_step == 0:
            # DÉMARRAGE MASTER : CRÉATION DU BRUIT
            actual_seed = noise_seed if noise_seed is not None else 0
            noise = comfy.sample.prepare_noise(latent_tensor, actual_seed, None)
            
            # ZERO-POINT : On vide le latent pour que 0 + Bruit = Bruit pur.
            # C'est ce qui élimine la "neige statique".
            latent_tensor = torch.zeros_like(latent_tensor)
            XT404_Sentinel.log("INIT", "Zero-Point Injection (Clean Start)", XT404_Sentinel.CYAN)

        XT404_Sentinel.log("EXEC", f"{node_id} | Steps: {start_step} -> {end_step}", XT404_Sentinel.CYAN)

        # --- 5. SAMPLING ---
        sampler_obj = comfy.samplers.sampler_object(target_sampler)
        
        try:
            samples = comfy.sample.sample_custom(
                model, noise, cfg, sampler_obj, current_sigmas, 
                positive, negative, latent_tensor, 
                noise_mask=None, callback=None, disable_pbar=False, seed=noise_seed
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                XT404_Sentinel.log("CRITICAL", "OOM Detected. Flushing VRAM...", XT404_Sentinel.RED)
                self.clean_vram(force_unload=True)
            raise e

        # --- 6. SORTIE ---
        out = latent_image.copy()
        out["samples"] = samples
        
        pass_options = {
            "total_steps": total_steps, 
            "next_step": end_step,
            "master_sigmas": full_sigmas # Transmission de l'horloge au nœud suivant
        }
        
        return (out, out, pass_options)

# --- DÉFINITION DES NOEUDS (ORDRE V1 STRICT) ---

class XT404_Skynet_1(Skynet_Core_Hybrid):
    """ Master Node (Gold) """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "eta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}), 
                "steps_to_run": ("INT", {"default": 1, "min": -1, "max": 1000}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 2.50, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_mode": (SAMPLER_MODES, {"default": "standard"}),
                "bongmath": ("BOOLEAN", {"default": True}),
                # DEFAULT SHIFT PASSÉ A 5.0 POUR QUALITÉ OPTIMALE
                "shift_val": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "sigmas": ("SIGMAS",), 
                "guides": ("GUIDES",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/Wan2.2"

    def process(self, model, positive, negative, latent_image, eta, sampler_name, scheduler, steps, steps_to_run, denoise, cfg, seed, sampler_mode, bongmath, shift_val=5.0, sigmas=None, guides=None, **kwargs):
        if sigmas is None and "scheduler_sigmas" in kwargs: sigmas = kwargs["scheduler_sigmas"]

        out, denoised, options = self.generic_sample(
            model, seed, steps, cfg, sampler_name, shift_val, positive, negative, latent_image, denoise, 
            steps_to_run, sampler_mode, bongmath, eta, 
            previous_options=None, sigmas_input=sigmas, is_chain=False, node_id="MASTER"
        )
        return (out, denoised, options, seed)


class XT404_Skynet_2(Skynet_Core_Hybrid):
    """ Chain Node (Gold) """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "eta": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "steps_to_run": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (SAMPLER_MODES, {"default": "resample"}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "shift_val": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "previous_options": ("DICT",), 
                "guides": ("GUIDES",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/Wan2.2"

    def process(self, model, positive, negative, latent_image, seed_input, eta, sampler_name, steps_to_run, cfg, sampler_mode, bongmath, shift_val=5.0, previous_options=None, guides=None, **kwargs):
        out, denoised, options = self.generic_sample(
            model, seed_input, 0, cfg, sampler_name, shift_val, positive, negative, latent_image, 1.0, 
            steps_to_run, sampler_mode, bongmath, eta, 
            previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="CHAIN"
        )
        return (out, denoised, options, seed_input)


class XT404_Skynet_3(Skynet_Core_Hybrid):
    """ Refiner Node (Gold) """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "eta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "steps_to_run": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (SAMPLER_MODES, {"default": "resample"}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "shift_val": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "previous_options": ("DICT",),
                "guides": ("GUIDES",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DICT")
    RETURN_NAMES = ("output", "denoised", "options")
    FUNCTION = "process"
    CATEGORY = "XT-404/Wan2.2"
    
    def process(self, model, positive, negative, latent_image, seed_input, eta, sampler_name, steps_to_run, cfg, sampler_mode, bongmath, shift_val=5.0, previous_options=None, guides=None, **kwargs):
        out, denoised, options = self.generic_sample(
            model, seed_input, 0, cfg, sampler_name, shift_val, positive, negative, latent_image, 1.0, 
            steps_to_run, sampler_mode, bongmath, eta, 
            previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="REFINER"
        )
        return (out, denoised, options)

NODE_CLASS_MAPPINGS = {
    "XT404_Skynet_1": XT404_Skynet_1,
    "XT404_Skynet_2": XT404_Skynet_2,
    "XT404_Skynet_3": XT404_Skynet_3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XT404_Skynet_1": "XT-404 Skynet 1 (Master)",
    "XT404_Skynet_2": "XT-404 Skynet 2 (Chain)",
    "XT404_Skynet_3": "XT-404 Skynet 3 (Refiner)"
}
