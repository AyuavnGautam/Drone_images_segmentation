import numpy as np
import sys
import os

# ── COMPATIBILITY PATCH (must run before torch.load) ─────────────────
# Older torchvision (<=0.12) defined ConvBNReLU and InvertedResidual
# directly in torchvision.models.mobilenet. Newer versions moved /
# renamed them. This patch restores all missing names so pickle can
# find them when loading a model saved with the old version.
import torchvision.models.mobilenet as _mb
import torchvision.models.mobilenetv2 as _mb2

# 1. ConvBNReLU → Conv2dNormActivation
if not hasattr(_mb, 'ConvBNReLU'):
    try:
        from torchvision.ops.misc import Conv2dNormActivation
        _mb.ConvBNReLU  = Conv2dNormActivation
        _mb2.ConvBNReLU = Conv2dNormActivation
    except ImportError:
        pass

# 2. InvertedResidual — lives in mobilenetv2 in newer torchvision
if not hasattr(_mb, 'InvertedResidual'):
    try:
        from torchvision.models.mobilenetv2 import InvertedResidual
        _mb.InvertedResidual = InvertedResidual
    except ImportError:
        pass

# 3. ConvBNActivation (another alias used in some versions)
if not hasattr(_mb, 'ConvBNActivation'):
    try:
        from torchvision.ops.misc import Conv2dNormActivation
        _mb.ConvBNActivation  = Conv2dNormActivation
        _mb2.ConvBNActivation = Conv2dNormActivation
    except ImportError:
        pass
# ─────────────────────────────────────────────────────────────────────

import torch
import segmentation_models_pytorch as smp
from torchvision import transforms as T
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = 'Unet-Mobilenet.pt'       # saved via torch.save(model, ...)
IMAGE_PATH = '088.jpg/088.jpg'                 # path to any drone image
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES  = 23

# Inference resize — matches t_test = A.Resize(768, 1152) from notebook
IMG_H, IMG_W = 768, 1152

# ImageNet normalization — matches notebook: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── CLASS LABELS (Semantic Drone Dataset, 23 classes) ────────────────
LABEL_NAMES = [
    'unlabeled', 'paved-area', 'dirt', 'grass', 'gravel',
    'water', 'rocks', 'pool', 'vegetation', 'roof',
    'wall', 'window', 'door', 'fence', 'fence-pole',
    'person', 'dog', 'car', 'bicycle', 'tree',
    'bald-tree', 'ar-marker', 'obstacle'
]

# ── CLASS COLORS (RGB) ───────────────────────────────────────────────
COLORS = [
    [0,   0,   0  ],  # unlabeled
    [128, 64,  128],  # paved-area
    [130, 76,  0  ],  # dirt
    [0,   102, 0  ],  # grass
    [112, 103, 87 ],  # gravel
    [28,  42,  168],  # water
    [48,  41,  30 ],  # rocks
    [0,   50,  89 ],  # pool
    [107, 142, 35 ],  # vegetation
    [70,  70,  70 ],  # roof
    [102, 102, 156],  # wall
    [254, 228, 12 ],  # window
    [254, 148, 12 ],  # door
    [190, 153, 153],  # fence
    [153, 153, 153],  # fence-pole
    [255, 22,  96 ],  # person
    [102, 51,  0  ],  # dog
    [9,   143, 150],  # car
    [119, 11,  32 ],  # bicycle
    [51,  51,  0  ],  # tree
    [190, 250, 190],  # bald-tree
    [112, 150, 146],  # ar-marker
    [2,   135, 115],  # obstacle
]

# ── LOAD MODEL ───────────────────────────────────────────────────────
def load_model():
    """
    Notebook saves: torch.save(model, 'Unet-Mobilenet.pt')
    Architecture  : smp.Unet(encoder_name='mobilenet_v2', encoder_weights='imagenet',
                             classes=23, activation=None, encoder_depth=5,
                             decoder_channels=[256, 128, 64, 32, 16])
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found.\n"
            "Place 'Unet-Mobilenet.pt' in the same directory as this script.\n"
            "It is produced by: torch.save(model, 'Unet-Mobilenet.pt') in the notebook."
        )

    # Primary: load full saved model object (matches notebook's torch.save(model, ...))
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        print(f"✅ Model loaded (full object) on {DEVICE}")
        return model
    except Exception as e:
        print(f"  Warning: full-object load failed ({str(e)[:120]})")
        print("  Falling back to architecture + state_dict load...")

    # Fallback: rebuild exact architecture, then load weights
    try:
        model = smp.Unet(
            encoder_name='mobilenet_v2',
            encoder_weights=None,
            in_channels=3,
            classes=N_CLASSES,
            activation=None,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16]
        )
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model.load_state_dict(state.state_dict())
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded (architecture + weights) on {DEVICE}")
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")

# ── PREPROCESS IMAGE ─────────────────────────────────────────────────
def preprocess(image_path):
    """
    Matches notebook inference pipeline:
      1. Read BGR → convert to RGB
      2. Resize to (768, 1152)   [matches t_test = A.Resize(768, 1152)]
      3. ToTensor + Normalize    [matches T.Compose([T.ToTensor(), T.Normalize(mean, std)])]
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

    pil_img   = Image.fromarray(img_resized)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    tensor = transform(pil_img).unsqueeze(0)   # (1, 3, H, W)
    return img_rgb, img_resized, tensor

# ── COLORIZE MASK ────────────────────────────────────────────────────
def colorize_mask(mask):
    """Convert integer class mask (H, W) → RGB image (H, W, 3)."""
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in enumerate(COLORS):
        colored[mask == cls_id] = color
    return colored

# ── AREA ESTIMATION ──────────────────────────────────────────────────
def estimate_areas(pred):
    total = pred.size
    areas = {}
    for cls_id in np.unique(pred):
        count = int((pred == cls_id).sum())
        areas[cls_id] = {'pixels': count, 'pct': count / total * 100}
    return areas

# ── RUN INFERENCE ────────────────────────────────────────────────────
def run(image_path=IMAGE_PATH):
    print(f"\n📷  Running segmentation on: {image_path}")
    print(f"    Resize → ({IMG_H}, {IMG_W})  |  Device: {DEVICE}\n")

    original_rgb, img_resized, tensor = preprocess(image_path)
    model = load_model()

    with torch.no_grad():
        output = model(tensor.to(DEVICE))                        # (1, 23, H, W) logits
        pred   = output.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)

    colored_mask = colorize_mask(pred)
    areas        = estimate_areas(pred)

    # ── Console summary ──────────────────────────────────────────────
    print("\n🏷️   Detected Classes")
    print("─" * 47)
    print(f"  {'ID':<4} {'Class':<18} {'Pixels':>9}  {'Area %':>7}")
    print("─" * 47)
    for cls_id, info in sorted(areas.items(), key=lambda x: -x[1]['pct']):
        print(f"  [{cls_id:2d}] {LABEL_NAMES[cls_id]:<18} {info['pixels']:>9,}  {info['pct']:>6.1f}%")
    print("─" * 47)

    # ── Plot ─────────────────────────────────────────────────────────
    overlay = cv2.addWeighted(img_resized, 0.55, colored_mask, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f'UNet-MobileNetV2 | Semantic Drone Dataset (23 classes)\n{os.path.basename(image_path)}',
        fontsize=13, fontweight='bold'
    )
    axes[0].imshow(img_resized);    axes[0].set_title('Original Image');         axes[0].axis('off')
    axes[1].imshow(colored_mask);   axes[1].set_title('Segmentation Mask');      axes[1].axis('off')
    axes[2].imshow(overlay);        axes[2].set_title('Overlay (55% + 45%)');    axes[2].axis('off')

    detected_ids = sorted(areas.keys())
    patches = [
        mpatches.Patch(
            color=np.array(COLORS[i]) / 255.0,
            label=f"[{i}] {LABEL_NAMES[i]} ({areas[i]['pct']:.1f}%)"
        )
        for i in detected_ids
    ]
    fig.legend(handles=patches, loc='lower center',
               ncol=min(6, len(patches)), fontsize=8, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    out_path = 'segmentation_result.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅  Result saved as '{out_path}'")

# ── ENTRY POINT ──────────────────────────────────────────────────────
if __name__ == '__main__':
    # Usage:
    #   python inference.py                → uses default IMAGE_PATH
    #   python inference.py my_photo.jpg  → custom image path
    img = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    run(img)