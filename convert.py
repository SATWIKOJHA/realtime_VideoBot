# convert.py
import torch
from models import Wav2Lip

def load_w2l_model(path):
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location="cpu")
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s, strict=False)
    return model.eval()

model = load_w2l_model("checkpoints/wav2lip_gan.pth")

# ✅ CORRECT: Wav2Lip expects 6-channel input (masked_face + full_face)
dummy_mel = torch.randn(1, 1, 80, 16)      # (B, 1, H, W)
dummy_img = torch.randn(1, 6, 96, 96)      # (B, 6, H, H) ← 6 channels!

# Optional: Use dynamo=True to avoid legacy warning (PyTorch >= 2.0)
torch.onnx.export(
    model,
    (dummy_mel, dummy_img),
    "wav2lip.onnx",
    input_names=["mel", "img"],
    output_names=["output"],
    dynamic_axes={
        "mel": {0: "batch"},
        "img": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=13,
    # dynamo=True  # Uncomment if you have PyTorch >= 2.0 and want new exporter
)
print("✅ ONNX exported successfully")