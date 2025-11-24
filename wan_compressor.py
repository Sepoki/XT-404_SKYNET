import os
import subprocess
import sys
import time
from typing import List, Union

# --- AUTO-INSTALL (Inchangé) ---
print(">> [Wan Architect] Initializing Compression Engine...")
def install_dependencies():
    try:
        import imageio_ffmpeg
    except ImportError:
        print(">> [Wan Dependency] Installing codecs...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        except Exception as e:
            print(f"!! [Wan Critical] Install failed: {e}")
install_dependencies()

class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False

class Wan_Video_Compressor:
    """
    V5 CALIBRATION:
    - Encodage H.265 10-bits (Meilleure compression des dégradés/couleurs).
    - Gestion chirurgicale du grain (Psy-RD) pour éviter l'explosion du poids.
    - Objectif : < 5MB avec qualité maximale.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": (AnyType("*"), {"forceInput": True}), 
                "mode": ([
                    "Web/Discord (Cible < 5MB)", 
                    "Master (Haute Fidelité)", 
                    "Archival (Lossless)"
                ], {"default": "Web/Discord (Cible < 5MB)"}),
                "remove_original": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_grain_texture": ("BOOLEAN", {"default": True, "tooltip": "Active le PSY-RD (Grain) sans faire exploser le poids."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("compressed_video_path",)
    FUNCTION = "compress_video"
    CATEGORY = "ComfyWan_Architect/PostProcessing"
    OUTPUT_NODE = True

    def _get_ffmpeg(self):
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except:
            try:
                subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return "ffmpeg"
            except:
                local = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
                if os.path.exists(local): return local
        raise RuntimeError("FFmpeg manquant.")

    def _recursive_find_video(self, data):
        valid_exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
        found = []
        if isinstance(data, str):
            if data.lower().endswith(valid_exts) and os.path.exists(data):
                found.append(data)
        elif isinstance(data, (list, tuple)):
            for item in data:
                found.extend(self._recursive_find_video(item))
        elif isinstance(data, dict):
            for value in data.values():
                found.extend(self._recursive_find_video(value))
        return found

    def compress_video(self, video_path, mode, remove_original, keep_grain_texture):
        target_files = self._recursive_find_video(video_path)
        if not target_files: return (video_path,)

        ffmpeg_exe = self._get_ffmpeg()
        output_paths = []

        # --- CALIBRATION V5 ---
        # H.265 (HEVC) Obligatoire pour le ratio poids/qualité
        codec = "libx265"
        preset = "veryslow" # Indispensable pour compresser fort sans perdre de qualité
        
        # Paramètres experts x265
        # aq-mode=3 : Bias vers les zones sombres (évite le blocking dans les ombres)
        x265_params = ["aq-mode=3"]

        if "Web/Discord" in mode:
            # Le réglage pour passer sous les 5MB
            crf = "26" 
            suffix = "_optm"
            # On limite le bitrate max pour être sûr de ne pas dépasser (safety net)
            # vbv-maxrate=5000k (5mbps) -> ~3.5 Mo pour 5 sec
            x265_params.append("vbv-maxrate=6000:vbv-bufsize=12000")
            
        elif "Master" in mode:
            crf = "22" # Le standard de qualité haute
            suffix = "_mstr"
        else: # Archival
            crf = "18"
            suffix = "_arch"

        # Gestion Fine du Grain (Sans -tune grain qui est trop lourd)
        if keep_grain_texture:
            # psy-rd=2.0 aide à garder la texture visuelle sans encoder le bruit blanc inutile
            x265_params.append("psy-rd=2.0:psy-rdoq=1.0")
        else:
            x265_params.append("psy-rd=1.0")

        x265_params_str = ":".join(x265_params)

        for input_file in target_files:
            dir_name = os.path.dirname(input_file)
            file_name = os.path.basename(input_file)
            name_no_ext, _ = os.path.splitext(file_name)
            
            if suffix in name_no_ext: 
                output_paths.append(input_file)
                continue

            output_file = os.path.join(dir_name, f"{name_no_ext}{suffix}.mp4")
            print(f">> [Wan Compressor] Mode: {mode} | CRF: {crf} | 10-bit Color")

            cmd = [
                ffmpeg_exe, "-y", "-v", "error",
                "-i", input_file,
                "-c:v", codec,
                "-crf", crf,
                "-preset", preset,
                "-x265-params", x265_params_str,
                # LE SECRET V5 : Encodage 10-bits (plus efficace même sur source 8-bits)
                "-pix_fmt", "yuv420p10le", 
                "-c:a", "copy",
                output_file
            ]

            try:
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                subprocess.run(cmd, check=True, startupinfo=startupinfo)

                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    orig_s = os.path.getsize(input_file) / (1024*1024)
                    new_s = os.path.getsize(output_file) / (1024*1024)
                    reduction = (1 - (new_s / orig_s)) * 100
                    
                    print(f">> [Wan Stats] {orig_s:.2f}MB -> {new_s:.2f}MB (-{reduction:.1f}%)")
                    output_paths.append(output_file)
                    
                    if remove_original:
                        try:
                            os.remove(input_file)
                        except: pass
                else:
                    output_paths.append(input_file)

            except Exception as e:
                print(f"!! [Wan Error] {e}")
                output_paths.append(input_file)

        if len(output_paths) == 1: return (output_paths[0],)
        return (output_paths,)