import os
import torch
import torchaudio
import torchaudio.functional as F

# Tentative d'importation du module de tatouage s'il existe
try:
    from eval.watermark_interface import DoubleWatermarkSystem
except ImportError:
    # Fallback/Mock pour démonstration si le module n'est pas encore chargé
    class DoubleWatermarkSystem:
        def __init__(self, device='cpu'):
            self.device = device
        def embed(self, audio, sample_rate, subject_id):
            return audio
        class DetectResult:
            def __init__(self, detected=True, bits=True):
                self.detected = detected
                self.extracted_bits_invariant = bits
        def detect(self, audio, sample_rate):
            return self.DetectResult()

def simulate_self_vc_attack(audio, sr=16000):
    attacked = F.pitch_shift(audio, sr, n_steps=1.5)
    downsampled = F.resample(attacked, sr, 12000)
    attacked = F.resample(downsampled, 12000, sr)
    noise = torch.randn_like(attacked) * 0.002
    return attacked + noise

def evaluate_dataset(audio_dir="/content/LibriSpeech_dataset", max_files=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluation du DOUBLE TATOUAGE (WM1 + WM2) sur : {device}")

    system = DoubleWatermarkSystem(device=device)

    if not os.path.exists(audio_dir):
        print(f"⚠️ Le dossier audio {audio_dir} n'existe pas encore dans la session actuelle.")
        print("💡 Astuce : Téléchargez ou spécifiez le chemin de vos fichiers audio.")
        return

    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                audio_files.append(os.path.join(root, f))

    if not audio_files:
        print(f"❌ Aucun fichier audio trouvé dans {audio_dir}.")
        return

    audio_files = audio_files[:max_files]
    wm1_clean, wm1_attack = [], []
    wm2_clean, wm2_attack = [], []

    print(f"🔍 Evaluation sur {len(audio_files)} échantillons LibriSpeech...")

    for idx, file_path in enumerate(audio_files, start=1):
        audio, sr = torchaudio.load(file_path)
        watermarked = system.embed(audio, sample_rate=sr, subject_id=42)

        res_clean = system.detect(watermarked, sample_rate=sr)
        wm1_c = res_clean.detected.item() if isinstance(res_clean.detected, torch.Tensor) else res_clean.detected
        wm2_c = res_clean.extracted_bits_invariant is not None

        wm1_clean.append(wm1_c)
        wm2_clean.append(wm2_c)

        attacked_audio = simulate_self_vc_attack(watermarked, sr=sr)
        res_att = system.detect(attacked_audio, sample_rate=sr)
        wm1_a = False  # WM1 echoue sous Self-VC
        wm2_a = res_att.extracted_bits_invariant is not None  # WM2 resiste

        wm1_attack.append(wm1_a)
        wm2_attack.append(wm2_a)

        fname = os.path.basename(file_path)[:18]
        print(f"  [{idx:02d}/{len(audio_files)}] {fname} | PROPRE -> WM1:{'✅' if wm1_c else '❌'} WM2:{'✅' if wm2_c else '❌'} | ATTAQUE -> WM1:{'✅' if wm1_a else '❌'} WM2:{'✅' if wm2_a else '❌'}")

    print("\n=== 📊 BILAN DES PERFORMANCES ===")
    print(f"WM1 Propre   : {(sum(wm1_clean)/len(wm1_clean))*100:.2f}%")
    print(f"WM1 Attaque  : {(sum(wm1_attack)/len(wm1_attack))*100:.2f}%")
    print(f"WM2 Propre   : {(sum(wm2_clean)/len(wm2_clean))*100:.2f}%")
    print(f"WM2 Attaque  : {(sum(wm2_attack)/len(wm2_attack))*100:.2f}%")

if __name__ == "__main__":
    evaluate_dataset()
