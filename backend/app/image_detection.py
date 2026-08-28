
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


# ── Image model cache ──────────────────────────────────────────────────────────
# Mirrors the audio model lazy-load pattern in service.py.
# The model is loaded once on first use, not at import time,
# so a missing/corrupt .pth file no longer crashes the whole app on startup.
_image_model_cache: dict = {}


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


def _get_image_model():
    """Return (model, device), loading from disk only on the first call."""
    if not _image_model_cache:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Loading image model into memory cache…")
        m = EfficientNetV2()
        m.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
        m.to(device)
        m.eval()
        _image_model_cache["model"] = m
        _image_model_cache["device"] = device
        logger.info("Image model cached and ready (device=%s).", device)
    return _image_model_cache["model"], _image_model_cache["device"]


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
        model, device = _get_image_model()

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


def generate_gradcam(image_path: str, output_path: str) -> bool:
    """Generate a GradCAM attention heatmap overlay and save it as a JPEG.

    Uses pure PyTorch hooks + NumPy + PIL — no cv2 or extra packages required.
    Hooks the last convolutional block of the EfficientNetV2 backbone, computes
    gradient-weighted feature-map activations, applies a jet colormap, and blends
    the result with the original image.

    Returns True on success, False on any error (the scan always completes regardless).
    """
    try:
        import numpy as np

        model, device = _get_image_model()

        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        img_tensor = transform(orig_img).unsqueeze(0).to(device)

        activations: list = [None]
        gradients: list = [None]

        # Hook the last block of the timm EfficientNetV2 backbone
        target_layer = model.base_model.blocks[-1]

        fwd_handle = target_layer.register_forward_hook(
            lambda m, i, o: activations.__setitem__(0, o)
        )
        bwd_handle = target_layer.register_full_backward_hook(
            lambda m, gi, go: gradients.__setitem__(0, go[0])
        )

        try:
            model.eval()
            output = model(img_tensor)          # forward WITH gradient tracking
            prob = torch.sigmoid(output)
            model.zero_grad()
            prob.backward()                     # compute gradients

            acts = activations[0]               # [1, C, H, W]
            grads = gradients[0]                # [1, C, H, W]

            if acts is None or grads is None or acts.dim() != 4:
                logger.warning("GradCAM: unexpected activation shape — skipping.")
                return False

            weights = grads.mean(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
            cam = (weights * acts).sum(dim=1).squeeze()              # [H, W]
            cam = torch.relu(cam).detach().cpu().numpy().astype(float)

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min < 1e-8:
                logger.warning("GradCAM: flat activation map — skipping.")
                return False
            cam = (cam - cam_min) / (cam_max - cam_min)              # normalise [0,1]

            # Resize to original image dimensions using PIL
            cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR
            )
            t = np.array(cam_pil).astype(np.float32) / 255.0        # [H, W] in [0,1]

            # Jet colormap — pure NumPy, no cv2/matplotlib
            r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
            g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
            b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
            heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

            # Blend: 40% original + 60% heatmap
            orig_np = np.array(orig_img).astype(np.float32)
            overlay = (0.4 * orig_np + 0.6 * heatmap).clip(0, 255).astype(np.uint8)

            Image.fromarray(overlay).save(output_path, quality=88)
            logger.info("GradCAM heatmap saved to %s", output_path)
            return True
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    except Exception as exc:
        logger.warning("GradCAM generation failed: %s", exc)
        return False
