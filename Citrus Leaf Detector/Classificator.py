import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights
from PIL import Image
from torch.utils.checkpoint import checkpoint
import numpy as np

# --- Konfigurasi ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        return checkpoint(self.module, x, use_reentrant=False)

class CitrusPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _get_transform(self):
        weights = EfficientNet_V2_L_Weights.DEFAULT
        return weights.transforms(antialias=True) 

    def _load_model(self):
        model = models.efficientnet_v2_l(weights=None)
        
        # Terapkan CheckpointWrapper
        for i, block_sequence in enumerate(model.features):
            if isinstance(block_sequence, nn.Sequential):
                model.features[i] = CheckpointWrapper(block_sequence)

        # Ubah layer klasifikasi
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
        
        # Muat bobot
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print(f"Model {MODEL_PATH} siap di {DEVICE.type.upper()}")
            return model
        except Exception as e:
            print(f"ERROR: Gagal memuat model dari {MODEL_PATH}. {e}")
            raise

    def predict(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return "File Error", 0.0, None

        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == 'cuda')):
                output = self.model(input_tensor)
        
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index] * 100

        class_details = {name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, probabilities)}
        
        return predicted_class, confidence, class_details