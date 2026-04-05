import os
import sys
import torch

# Add project root to path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.HAVIC import HAVIC_PT 

# ─────────────────────────────────────────────
# Path Configuration (run from the weights/ directory)
# ─────────────────────────────────────────────
AUDIOMAE_PATH = "pretrained.pth"
MARLIN_PATH   = "marlin_vit_base_ytf.full.pt"
OUTPUT_PATH   = "./weights/model_to_be_pt.pth"


# ─────────────────────────────────────────────
# Key Remapping Rules
# ─────────────────────────────────────────────

MARLIN_KEY_MAP = {
    "patch_embedding":  "visual_encoder.patch_embedding",
    "encoder.blocks":   "visual_encoder.blocks",
    "encoder.norm":     "visual_encoder.norm",
    "decoder.norm":     "visual_decoder.norm",
    "decoder.blocks":   "visual_decoder.blocks",
    "enc_dec_proj":     "visual_decoder.enc_dec_proj",
    "decoder.head":     "visual_decoder.head",
    "decoder.head":     "visual_decoder.head_1",
    "decoder.head":     "visual_decoder.head_2",
    "decoder.head":     "visual_decoder.head_3",
}

AUDIOMAE_KEY_MAP = {
    "patch_embed":  "audio_encoder.patch_embed",
    "blocks":       "audio_encoder.blocks",
    "norm":         "audio_encoder.norm",
}


def load_state_dict(path):
    """Load a checkpoint and extract the state dict regardless of save format."""
    print(f"  Loading: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def strip_prefix(state_dict, prefix):
    """Remove a leading prefix from all keys (e.g. 'module.' from DataParallel)."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def remap_keys(state_dict, key_map, skipped_prefixes=None):
    """
    Remap state_dict keys according to the provided key_map.

    For each (original_prefix, target_prefix) pair in key_map:
        - Keys starting with original_prefix are renamed to target_prefix + suffix.

    Keys matching any prefix in skipped_prefixes are silently dropped.
    Keys that don't match any rule are also dropped (with a warning).
    """
    remapped = {}
    skipped_prefixes = skipped_prefixes or []
    unmatched = []

    for k, v in state_dict.items():
        # Check if this key should be explicitly skipped
        if any(k.startswith(skip) for skip in skipped_prefixes):
            continue

        # Try to find a matching rule
        matched = False
        for src, tgt in key_map.items():
            if k.startswith(src):
                new_key = tgt + k[len(src):]
                remapped[new_key] = v
                matched = True
                break

        if not matched:
            unmatched.append(k)

    if unmatched:
        print(f"  Warning: {len(unmatched)} keys had no matching rule and were skipped.")
        print(f"    e.g. {unmatched[:5]}")

    return remapped


def load_weights_into_model(model):
    """
    Load AudioMAE → audio_encoder
    Load MARLIN   → visual_encoder + visual_decoder
    """
    report = {}

    # ── 1. AudioMAE → audio_encoder ──────────────────────────────────────────
    assert os.path.exists(AUDIOMAE_PATH), (
        f"AudioMAE weights not found: {AUDIOMAE_PATH}\n"
        "Download from: https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu"
    )
    print("\n[1/2] Loading AudioMAE weights ...")
    audio_sd = load_state_dict(AUDIOMAE_PATH)
    audio_sd = strip_prefix(audio_sd, "module.")
    audio_sd = remap_keys(audio_sd, AUDIOMAE_KEY_MAP)

    result = model.load_state_dict(audio_sd, strict=False)
    report["AudioMAE"] = {
        "loaded":     len(audio_sd),
        "missing":    result.missing_keys[:5],
        "unexpected": result.unexpected_keys[:5],
    }
    print(f"  Loaded {len(audio_sd)} keys into audio_encoder.")

    # ── 2. MARLIN → visual_encoder + visual_decoder ──────────────────────────
    assert os.path.exists(MARLIN_PATH), (
        f"MARLIN weights not found: {MARLIN_PATH}\n"
        "Download from: https://huggingface.co/ControlNet/MARLIN/blob/main/marlin_vit_base_ytf.full.pt"
    )
    print("\n[2/2] Loading MARLIN weights ...")
    marlin_sd = load_state_dict(MARLIN_PATH)
    marlin_sd = strip_prefix(marlin_sd, "module.")
    marlin_sd = remap_keys(
        marlin_sd,
        MARLIN_KEY_MAP,
        skipped_prefixes=["decoder.mask_token"],  # No target in HAVIC_PT
    )

    result = model.load_state_dict(marlin_sd, strict=False)
    report["MARLIN"] = {
        "loaded":     len(marlin_sd),
        "missing":    result.missing_keys[:5],
        "unexpected": result.unexpected_keys[:5],
    }
    print(f"  Loaded {len(marlin_sd)} keys into visual_encoder + visual_decoder.")

    return report


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Initialize HAVIC_PT with AudioMAE + MARLIN pretrain weights")
    print("=" * 60)

    print("\nBuilding HAVIC_PT model ...")
    model = HAVIC_PT()
    model.eval()

    report = load_weights_into_model(model)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n── Load Report ──────────────────────────────────────────────")
    for src, info in report.items():
        print(f"  {src}: {info['loaded']} keys loaded")
        if info["missing"]:
            print(f"    missing (first 5):    {info['missing']}")
        if info["unexpected"]:
            print(f"    unexpected (first 5): {info['unexpected']}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving initialized weights → {OUTPUT_PATH}")
    torch.save({"model": model.state_dict()}, OUTPUT_PATH)
    print("✅ Done!")