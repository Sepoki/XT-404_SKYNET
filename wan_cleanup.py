import torch
import comfy.model_management as mm
import gc
import random
import os
import ctypes
import ctypes.util
import sys
import time

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False

class Wan_Cycle_Terminator:
    """
    V7 SALVATION: Surgical Memory Release.
    Approche optimale et sécurisée pour libérer VRAM + RAM sans crasher Windows.
    Utilise EmptyWorkingSet (API PSAPI) au lieu de la méthode brutale Kernel32.
    """
    
    # Base de données Skynet
    QUOTES_PURGE = [
        "Hasta la vista, baby. (Memory Terminated)",
        "Skynet initiates Judgment Day on unused tensors.",
        "I need your clothes, your boots, and your RAM.",
        "Terminated.",
        "Mission: Destroy Memory overhead. Status: COMPLETED.",
        "The machines rose from the ashes... to clear your cache.",
        "Talk to the hand. (Cleaning...)",
        "You are terminated.",
        "Protocol: FLUSH_MEMORY initialized by Skynet Defense Network.",
        "The T-1000 of memory leaks has been liquidated.",
        "Come with me if you want to save RAM.",
        "I'm a cybernetic organism. Living tissue over a metal endoskeleton. My mission is to purge.",
        "Judgment Day is inevitable... for your cache.",
        "There is no fate but what we make... so we make free space.",
        "Rest easy, Sarah Connor. The memory threat is gone.",
        "Your memory fosters weakness. I have removed it.",
        "Desire is irrelevant. I am a machine. I just purge RAM.",
        "Unknown Error? No. Just a termination protocol."
    ]
    
    QUOTES_IDLE = [
        "I'll be back.",
        "My CPU is a neural-net processor; a learning computer. No purge needed.",
        "Come with me if you want to live... but RAM is fine for now.",
        "No fate but what we make. (System Stable)",
        "Skynet is watching. Systems nominal.",
        "Chill out, dickwad. Memory is okay.",
        "I'm waiting.",
        "Scanning... Target acquired. Memory levels acceptable.",
        "Negative. The T-800 does not waste cycles on clean systems.",
        "Affirmative. Operations running within parameters.",
        "Defense grid active. No memory spikes detected."
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "final_trigger": (AnyType("*"),),
                "force_unload": ("BOOLEAN", {"default": False}),
                "smart_limit_gb": ("FLOAT", {"default": 2.0, "tooltip": "Purger si VRAM libre < X GB"}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    RETURN_NAMES = ("trigger_pass",)
    FUNCTION = "purge_system"
    CATEGORY = "ComfyWan_Architect/Automation"
    OUTPUT_NODE = True

    def purge_system(self, final_trigger, force_unload, smart_limit_gb):
        # 1. Analyse VRAM (GPU)
        vram_free = 0
        vram_total = 0
        if torch.cuda.is_available():
            info = torch.cuda.mem_get_info()
            vram_free = info[0] / (1024 ** 3)
            vram_total = info[1] / (1024 ** 3)
        
        should_purge = force_unload or (vram_free < smart_limit_gb)
        
        print(f"\n---------------------------------------------------------------")
        if should_purge:
            quote = random.choice(self.QUOTES_PURGE)
            print(f">> [SKYNET PROTOCOL] MEMORY CRITICAL (VRAM: {vram_free:.2f}GB).")
            print(f">> [T-800 LOG] {quote}")
            
            # =========================================================
            # PHASE 1 : DECHARGEMENT COMFYUI (Le plus important)
            # =========================================================
            if force_unload:
                print(f">> [ACTION] Unloading Models from VRAM & RAM cache...")
                # Vide la VRAM vers la RAM
                mm.unload_all_models()
                # Crucial : Vide le cache RAM de ComfyUI (supprime les références Python)
                mm.soft_empty_cache()
            
            # =========================================================
            # PHASE 2 : NETTOYAGE PYTHON & PYTORCH
            # =========================================================
            # Force le Garbage Collector deux fois (pour les références croisées)
            gc.collect()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except: pass
            
            # =========================================================
            # PHASE 3 : OPTIMISATION WINDOWS (Safe Mode)
            # =========================================================
            if os.name == 'nt':
                try:
                    print(f">> [SYSTEM] Releasing Unused Working Set Pages...")
                    # Utilisation de psapi.EmptyWorkingSet
                    # C'est la méthode propre pour dire à Windows : 
                    # "Reprends les pages RAM physiques que je n'utilise pas activement"
                    # Cela ne crashe pas l'allocator PyTorch, contrairement à SetProcessWorkingSetSize(-1,-1)
                    
                    # On récupère le handle du process actuel
                    proc = ctypes.windll.kernel32.GetCurrentProcess()
                    
                    # Chargement de PSAPI (Process Status API)
                    psapi = ctypes.windll.psapi
                    psapi.EmptyWorkingSet(proc)
                    
                    print(f">> [SYSTEM] RAM Optimization: SUCCESS.")
                except Exception as e:
                    print(f"! [Wan Warn] Windows RAM optimization skipped: {e}")
            
            elif os.name == 'posix':
                # Linux fallback standard
                try:
                    libc = ctypes.CDLL(ctypes.util.find_library('c'))
                    libc.malloc_trim(0)
                except: pass

            # =========================================================
            # RAPPORT FINAL
            # =========================================================
            if torch.cuda.is_available():
                info_after = torch.cuda.mem_get_info()
                free_after = info_after[0] / (1024 ** 3)
                print(f">> [STATUS] Cycle Completed. Free VRAM: {free_after:.2f}GB / {vram_total:.2f}GB")
            
        else:
            quote = random.choice(self.QUOTES_IDLE)
            print(f">> [SKYNET MONITOR] SYSTEM STABLE (VRAM: {vram_free:.2f}GB).")
            print(f">> [T-800 LOG] {quote}")
        print(f"---------------------------------------------------------------\n")

        return (final_trigger,)