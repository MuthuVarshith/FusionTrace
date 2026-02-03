
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from timm import create_model
import logging
from .config import IMAGE_MODEL_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define EfficientNetV2 model
class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3, pretrained=False):
        super().__init__()
        self.base_model = create_model('tf_efficientnetv2_l', pretrained=pretrained, num_classes=0)
        num_features = self.base_model.num_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        self.freeze_layers()

    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in list(self.base_model.parameters())[-30:]:
            param.requires_grad = True

    def forward(self, x):
        features = self.base_model.forward_features(x)
        out = self.classifier(features)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = EfficientNetV2()
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded image model from {IMAGE_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load image model from {IMAGE_MODEL_PATH}: {e}")
    raise

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def detect_image_deepfake(image_path: str) -> dict:
    """
    Detect if an image is a deepfake.
    Args:
        image_path (str): Path to the input image.
    Returns:
        dict: Prediction ("Real" or "Fake") and confidence score.
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            prediction = "Fake" if prob >= 0.5 else "Real"
            confidence = prob if prediction == "Fake" else 1 - prob

        return {
            "prediction": prediction,
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return {"prediction": "Error", "confidence": str(e)}
