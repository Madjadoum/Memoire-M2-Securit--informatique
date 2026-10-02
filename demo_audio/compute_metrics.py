import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import sys

def compute_audio_metrics(ref_path, deg_path, sample_rate=16000):
    # 1. Chargement des fichiers audio
    x, sr1 = sf.read(ref_path)
    x_w, sr2 = sf.read(deg_path)
    
    # S'assurer que les audios sont mono
    if len(x.shape) > 1:
        x = x[:, 0]
    if len(x_w.shape) > 1:
        x_w = x_w[:, 0]

    # S'assurer de la même longueur
    min_len = min(len(x), len(x_w))
    x = x[:min_len]
    x_w = x_w[:min_len]

    # 2. Calcul du SNR (Signal-to-Noise Ratio)
    noise = x_w - x
    power_signal = np.sum(x ** 2)
    power_noise = np.sum(noise ** 2) + 1e-10
    snr_val = 10 * np.log10(power_signal / power_noise)

    # 3. Calcul du PESQ (Wideband / 16kHz)
    try:
        pesq_val = pesq(sample_rate, x, x_w, 'wb')
    except Exception as e:
        pesq_val = float('nan')
        print(f"Erreur PESQ: {e}")

    # 4. Calcul du STOI
    stoi_val = stoi(x, x_w, sample_rate, extended=False)

    # 5. Calcul de la Corrélation Normalisée (NC)
    norm_x = np.linalg.norm(x)
    norm_xw = np.linalg.norm(x_w)
    if norm_x == 0 or norm_xw == 0:
        nc_val = 0
    else:
        nc_val = np.dot(x, x_w) / (norm_x * norm_xw)

    return {
        "SNR (dB)": round(snr_val, 2),
        "PESQ": round(pesq_val, 2),
        "STOI": round(stoi_val, 3),
        "NC": round(nc_val, 4)
    }

if __name__ == "__main__":
    ref = "1_audio_tatoue_propre.wav"
    deg = "2_audio_apres_self_vc.wav"
    
    print(f"Comparaison de :\nRef: {ref}\nDeg: {deg}\n")
    
    metrics = compute_audio_metrics(ref, deg)

    print("--- RÉSULTATS DES MÉTRIQUES ---")
    for metric, value in metrics.items():
        print(f"{metric} : {value}")
