import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights
from PIL import Image
from torch.utils.checkpoint import checkpoint
import numpy as np
import sys
import os

# --- Konfigurasi Model dan Kelas ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PASTIKAN NAMA FILE MODEL INI BENAR
MODEL_PATH = 'EfficientNetV2-L_Citrus.pth' 

CLASS_NAMES = [
    'Citrus_Canker_Diseases_Leaf_Orange',
    'Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange',
    'Healthy_Leaf_Orange',
    'Multiple_Diseases_Leaf_Orange',
    'Young_Healthy_Leaf_Orange'
]
NUM_CLASSES = len(CLASS_NAMES)

# --- Kelas Penolong (Wajib untuk Memuat Model) ---
class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        # Menggunakan use_reentrant=False seperti di notebook Anda
        return checkpoint(self.module, x, use_reentrant=False)

class CitrusPredictor:
    """Kelas untuk memuat dan menjalankan inferensi model EfficientNetV2-L."""
    def __init__(self):
        # Simpan path model sebagai atribut kelas
        self.MODEL_PATH = MODEL_PATH
        self.model = self._load_model()
        self.transform = self._get_transform()
        print(f"Model {self.MODEL_PATH} siap di {DEVICE.type.upper()}.")

    def _get_transform(self):
        """Mendapatkan transformasi preprocessing (sama seperti val_transforms)."""
        weights = EfficientNet_V2_L_Weights.DEFAULT
        # val_transforms Anda hanya menggunakan preprocess bawaan
        return weights.transforms(antialias=True) 

    def _load_model(self):
        """Memuat arsitektur model dan bobot terbaik."""
        # 1. Inisialisasi arsitektur (tanpa bobot pretrained bawaan)
        model = models.efficientnet_v2_l(weights=None)
        
        # 2. Terapkan CheckpointWrapper
        for i, block_sequence in enumerate(model.features):
            if isinstance(block_sequence, nn.Sequential):
                model.features[i] = CheckpointWrapper(block_sequence)

        # 3. Ubah layer klasifikasi
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        
        # 4. Muat bobot
        if not os.path.exists(self.MODEL_PATH):
             raise FileNotFoundError(f"File model tidak ditemukan: {self.MODEL_PATH}. Pastikan nama dan lokasi file benar.")

        state_dict = torch.load(self.MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model

    def predict(self, image_path: str):
        """Memuat gambar, melakukan inferensi, dan mengembalikan hasil."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return "File Error", 0.0, None

        # Pre-processing dan tambahkan batch dimension (1, C, H, W)
        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Inferensi dengan AMP opsional untuk GPU
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
                output = self.model(input_tensor)
        
        # Konversi output ke probabilitas dan prediksi
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index] * 100

        class_details = {name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, probabilities)}
        
        return predicted_class, confidence, class_details