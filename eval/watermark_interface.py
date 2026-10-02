import torch

class DoubleWatermarkSystem:
    """
    Interface du système de double tatouage :
    - WM1 : Filigrane spectral (type AudioSeal)
    - WM2 : Filigrane invariant biométrique
    """
    def __init__(self, device='cpu'):
        self.device = device

    def embed(self, audio, sample_rate=16000, subject_id=42):
        # Simulation/Ancrage de l'insertion des deux filigranes
        # WM1 : Bruit spectral haute fréquence
        wm1_layer = torch.sin(2 * 3.14159 * 4000 * torch.linspace(0, audio.shape[-1]/sample_rate, audio.shape[-1])) * 0.005
        # WM2 : Signature invariante
        wm2_layer = torch.cos(2 * 3.14159 * 100 * torch.linspace(0, audio.shape[-1]/sample_rate, audio.shape[-1])) * 0.002
        
        watermarked = audio + wm1_layer.to(audio.device) + wm2_layer.to(audio.device)
        return watermarked

    class DetectResult:
        def __init__(self, detected=True, bits=None):
            self.detected = torch.tensor(detected)
            self.extracted_bits_invariant = bits if bits is not None else [1, 0, 1, 1, 0, 1, 0, 0]

    def detect(self, audio, sample_rate=16000):
        # Simulation du processus de détection
        return self.DetectResult(detected=True)
