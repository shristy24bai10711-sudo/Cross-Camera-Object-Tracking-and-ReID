"""ResNet-50 appearance embedding extractor for object re-identification."""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

class ReIDEmbedder:
    def __init__(self, model_name: str = "resnet50", embedding_dim: int = 2048,
                 device: str = "cpu"):
        if not hasattr(models, model_name):
            raise ValueError(f"Unsupported torchvision model: {model_name}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.device = torch.device(device)
        weights_enum = getattr(models, f"{model_name.capitalize()}_Weights", None)
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = weights_enum.DEFAULT if weights_enum else None
        base = getattr(models, model_name)(weights=weights)
        self.model = nn.Sequential(*list(base.children())[:-1]).to(self.device).eval()
        actual_dim = base.fc.in_features if hasattr(base, "fc") else embedding_dim
        self.embedding_dim = int(actual_dim)
        if embedding_dim != self.embedding_dim:
            # Keep the configured value honest; the backbone determines the dimension.
            self.embedding_dim = int(actual_dim)
        self.transform = T.Compose([
            T.Resize((256, 128)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Cannot extract a ReID embedding from an empty crop.")
        rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        feat = self.model(tensor).flatten(1)[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / max(norm, 1e-12)
