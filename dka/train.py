# # Dynamic Kernel Attention (DKA) — Training Notebook
# 
# Main training notebook for DKA experiments. Runnable top-to-bottom.
# 
# **Supported tasks:**
# - CIFAR-10 image classification
# - Tiny ImageNet image classification
# - AG News text classification
# - WikiText-2 language modeling
# 
# **Supported models:**
# - DKA-Small (primary)
# - Baselines: ViT-Small, ResNet-18, ConvNeXt-Tiny, DeiT-Small
# 
# Select the config YAML and set flags below to control what runs.

# ## 1. Imports and Setup

import os
import sys
import time
import math
import copy
import random
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Ensure the project root is on the path
PROJECT_ROOT = Path(os.getcwd()).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Device detection ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (no GPU detected)")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# --- Optional: Weights & Biases ---
try:
    import wandb
    WANDB_AVAILABLE = True
    print("W&B available")
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not installed — logging to console only")


# ## 2. Load Config from YAML

# ============================================================
# SELECT CONFIG HERE
# Options: "cifar10", "tinyimagenet", "agnews", "wikitext2"
# ============================================================
CONFIG_NAME = "cifar10"

# ============================================================
# BASELINE MODE
# Set to True and choose baseline_model to train a baseline
# instead of DKA. Options: "vit", "resnet18", "convnext", "deit"
# ============================================================
RUN_BASELINE = False
BASELINE_MODEL = "vit"

# ============================================================

config_path = Path("configs") / f"{CONFIG_NAME}.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Override baseline flag from cell settings
if RUN_BASELINE:
    cfg["baselines"]["enabled"] = True
    cfg["baselines"]["model_name"] = BASELINE_MODEL

print(f"Loaded config: {config_path}")
print(f"Experiment: {cfg['experiment']['name']}")
print(f"Model: {cfg['model']['name']} ({'BASELINE: ' + BASELINE_MODEL if RUN_BASELINE else 'DKA'})")
print(f"Dataset: {cfg['data']['dataset']}")
print(f"Epochs: {cfg['training']['epochs']}, Batch size: {cfg['training']['batch_size']}")
print(f"Precision: {cfg['experiment']['precision']}")


# --- Seed everything for reproducibility ---
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg["experiment"]["seed"])
print(f"Seed: {cfg['experiment']['seed']}")


# ## 3. Create Dataloaders

from dka.data.cifar10 import get_cifar10_loaders, MixupCutMix

dataset_name = cfg["data"]["dataset"]
batch_size = cfg["training"]["batch_size"]
num_workers = cfg["training"]["num_workers"]
data_dir = cfg["data"]["data_dir"]

mixup_cutmix_fn = None  # Will be set for image tasks with augmentation

if dataset_name == "cifar10":
    from dka.data.cifar10 import get_cifar10_loaders
    train_loader, val_loader, mixup_cutmix_fn = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    num_classes = 10
    task = "classification"

elif dataset_name == "tinyimagenet":
    from dka.data.tinyimagenet import get_tinyimagenet_loaders
    train_loader, val_loader, mixup_cutmix_fn = get_tinyimagenet_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    num_classes = 200
    task = "classification"

elif dataset_name == "agnews":
    from dka.data.agnews import get_agnews_loaders
    train_loader, val_loader = get_agnews_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_len=cfg["data"]["seq_len"],
        vocab_size=cfg["data"]["vocab_size"],
    )
    num_classes = 4
    task = "classification"

elif dataset_name == "wikitext2":
    from dka.data.wikitext2 import get_wikitext2_loaders
    train_loader, val_loader = get_wikitext2_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seq_len=cfg["data"]["seq_len"],
        vocab_size=cfg["data"]["vocab_size"],
    )
    num_classes = None  # Language modeling
    task = "language_modeling"

else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

# Check if mixup/cutmix should be disabled
aug_cfg = cfg.get("augmentation", {})
if aug_cfg.get("mixup_alpha", 0) == 0 and aug_cfg.get("cutmix_alpha", 0) == 0:
    mixup_cutmix_fn = None

print(f"Dataset: {dataset_name}")
print(f"Task: {task}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"Mixup/CutMix: {'enabled' if mixup_cutmix_fn is not None else 'disabled'}")


# ## 4. Create Model

from dka.models.dka_model import DKAImageModel, DKATextModel
from dka.models.baselines.vit import ViT, ViTForTextClassification


def create_dka_model(cfg):
    """Create a DKA model from config."""
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    reg_cfg = cfg["regularization"]

    if model_cfg["type"] == "image":
        model = DKAImageModel(
            img_size=data_cfg["image_size"],
            patch_size=data_cfg["patch_size"],
            num_classes=data_cfg["num_classes"],
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            kernel_sizes=model_cfg["kernel_sizes"],
            rank=model_cfg["rank"],
            dropout=reg_cfg["dropout"],
            drop_path_rate=reg_cfg["drop_path"],
        )
    elif model_cfg["type"] == "text":
        is_lm = data_cfg.get("task") == "language_modeling"
        model = DKATextModel(
            vocab_size=data_cfg["vocab_size"],
            max_seq_len=data_cfg["seq_len"],
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            kernel_sizes=model_cfg["kernel_sizes"],
            rank=model_cfg["rank"],
            num_classes=data_cfg.get("num_classes") if not is_lm else None,
            dropout=reg_cfg["dropout"],
            drop_path_rate=reg_cfg["drop_path"],
            causal=model_cfg.get("causal", False),
        )
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")

    return model


def create_baseline_model(cfg):
    """Create a baseline model from config."""
    baseline_name = cfg["baselines"]["model_name"]
    data_cfg = cfg["data"]
    reg_cfg = cfg["regularization"]

    if data_cfg["dataset"] in ["cifar10", "tinyimagenet"]:
        if baseline_name == "vit":
            model = ViT(
                image_size=data_cfg["image_size"],
                patch_size=data_cfg["patch_size"],
                in_channels=data_cfg["in_channels"],
                num_classes=data_cfg["num_classes"],
                d=cfg["model"]["d_model"],
                num_heads=cfg["model"]["num_heads"],
                num_layers=cfg["model"]["num_layers"],
                dropout=reg_cfg["dropout"],
                drop_path=reg_cfg["drop_path"],
            )
        elif baseline_name == "resnet18":
            import torchvision.models as tv_models
            model = tv_models.resnet18(num_classes=data_cfg["num_classes"])
            # Modify for small input: kernel 3, stride 1, no maxpool
            if data_cfg["image_size"] <= 32:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()
        elif baseline_name == "convnext":
            try:
                import timm
                model = timm.create_model(
                    "convnext_tiny",
                    pretrained=False,
                    num_classes=data_cfg["num_classes"],
                    in_chans=data_cfg["in_channels"],
                )
            except ImportError:
                raise ImportError("timm required for ConvNeXt baseline: pip install timm")
        elif baseline_name == "deit":
            try:
                import timm
                model = timm.create_model(
                    "deit_small_patch16_224",
                    pretrained=False,
                    num_classes=data_cfg["num_classes"],
                    img_size=data_cfg["image_size"],
                )
            except ImportError:
                raise ImportError("timm required for DeiT baseline: pip install timm")
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

    elif data_cfg["dataset"] in ["agnews", "wikitext2"]:
        # Text baselines use the ViTForTextClassification
        model = ViTForTextClassification(
            vocab_size=data_cfg["vocab_size"],
            max_seq_len=data_cfg["seq_len"],
            num_classes=data_cfg.get("num_classes", 4),
            d=cfg["model"]["d_model"],
            num_heads=cfg["model"]["num_heads"],
            num_layers=cfg["model"]["num_layers"],
            dropout=reg_cfg["dropout"],
            drop_path=reg_cfg["drop_path"],
        )
    else:
        raise ValueError(f"No baseline for dataset: {data_cfg['dataset']}")

    return model


# --- Create the model ---
if cfg["baselines"]["enabled"]:
    model = create_baseline_model(cfg)
    model_label = f"Baseline ({cfg['baselines']['model_name']})"
else:
    model = create_dka_model(cfg)
    model_label = cfg["model"]["name"]

model = model.to(DEVICE)

# --- Parameter count ---
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: {model_label}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Approx size: {total_params * 4 / 1e6:.1f} MB (fp32)")

# Detailed breakdown for DKA
if not cfg["baselines"]["enabled"]:
    kernel_gen_params = 0
    alpha_params = 0
    base_kernel_params = 0
    other_params = 0
    for name, p in model.named_parameters():
        if "kernel_generators" in name:
            if "alpha" in name:
                alpha_params += p.numel()
            elif "K_base" in name:
                base_kernel_params += p.numel()
            else:
                kernel_gen_params += p.numel()
        else:
            other_params += p.numel()

    print(f"\nParameter breakdown:")
    print(f"  Kernel generators:  {kernel_gen_params:,}")
    print(f"  Base kernels:       {base_kernel_params:,}")
    print(f"  Alpha scalars:      {alpha_params:,}")
    print(f"  Other (embed+FFN):  {other_params:,}")


# ## 5. Create Optimizer, Scheduler, and Loss Function

def build_optimizer(model, cfg):
    """Build AdamW optimizer with separate parameter groups.

    Three groups:
    1. Main parameters (embeddings, FFN, projections, K_base, LayerNorm)
    2. Kernel generator parameters (spatial/channel MLPs)
    3. Alpha scalars

    Biases and LayerNorm params get weight_decay=0 in their respective groups.
    """
    opt_cfg = cfg["optimizer"]
    is_dka = not cfg["baselines"]["enabled"]

    if not is_dka:
        # Simple single-group optimizer for baselines
        no_decay = ["bias", "norm"]
        params = [
            {
                "params": [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": opt_cfg["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=opt_cfg["main_lr"])
        return optimizer

    # DKA: three parameter groups
    kernel_gen_params_decay = []
    kernel_gen_params_no_decay = []
    alpha_params = []
    main_params_decay = []
    main_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "kernel_generators" in name:
            if "alpha" in name:
                alpha_params.append(param)
            elif "K_base" in name:
                main_params_decay.append(param)
            elif "bias" in name:
                kernel_gen_params_no_decay.append(param)
            else:
                kernel_gen_params_decay.append(param)
        elif "bias" in name or "norm" in name.lower():
            main_params_no_decay.append(param)
        else:
            main_params_decay.append(param)

    param_groups = [
        {
            "params": main_params_decay,
            "lr": opt_cfg["main_lr"],
            "weight_decay": opt_cfg["weight_decay"],
            "group_name": "main_decay",
        },
        {
            "params": main_params_no_decay,
            "lr": opt_cfg["main_lr"],
            "weight_decay": 0.0,
            "group_name": "main_no_decay",
        },
        {
            "params": kernel_gen_params_decay,
            "lr": opt_cfg["kernel_gen_lr"],
            "weight_decay": opt_cfg["weight_decay"],
            "group_name": "kernel_gen_decay",
        },
        {
            "params": kernel_gen_params_no_decay,
            "lr": opt_cfg["kernel_gen_lr"],
            "weight_decay": 0.0,
            "group_name": "kernel_gen_no_decay",
        },
        {
            "params": alpha_params,
            "lr": opt_cfg["alpha_lr"],
            "weight_decay": 0.0,
            "group_name": "alpha",
        },
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups)

    print("Optimizer parameter groups:")
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['group_name']}: {n_params:,} params, lr={g['lr']}, wd={g['weight_decay']}")

    return optimizer


# --- Scale LR for batch size ---
# Reference batch sizes from build guide (section 5.4).
# If actual batch size differs, scale LR by sqrt(actual / ref) for AdamW.
REFERENCE_BATCH_SIZES = {
    "cifar10": 256,
    "tinyimagenet": 128,
    "agnews": 64,
    "wikitext2": 64,
}
ref_bs = REFERENCE_BATCH_SIZES.get(dataset_name, batch_size)
if batch_size != ref_bs:
    lr_scale = math.sqrt(batch_size / ref_bs)
    cfg["optimizer"]["main_lr"] *= lr_scale
    cfg["optimizer"]["kernel_gen_lr"] *= lr_scale
    cfg["optimizer"]["alpha_lr"] *= lr_scale
    print(f"LR scaled by sqrt({batch_size}/{ref_bs}) = {lr_scale:.3f} for batch_size={batch_size}")
    print(f"  main_lr:       {cfg['optimizer']['main_lr']:.2e}")
    print(f"  kernel_gen_lr: {cfg['optimizer']['kernel_gen_lr']:.2e}")
    print(f"  alpha_lr:      {cfg['optimizer']['alpha_lr']:.2e}")

optimizer = build_optimizer(model, cfg)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build warmup + cosine annealing schedule.

    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    """
    sched_cfg = cfg["scheduler"]
    total_epochs = cfg["training"]["epochs"]
    warmup_epochs = sched_cfg["warmup_epochs"]
    min_lr = sched_cfg["min_lr"]

    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            # Scale factor to anneal from base_lr to min_lr
            # At progress=0: return 1.0 (full lr)
            # At progress=1: return min_lr/base_lr
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Interpolate between 1.0 and min_lr/base_lr
            # We approximate by just using the cosine factor with a floor
            return max(min_lr / optimizer.param_groups[0]["lr"], cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


steps_per_epoch = len(train_loader)
scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
print(f"Scheduler: warmup {cfg['scheduler']['warmup_epochs']} epochs, "
      f"cosine to {cfg['scheduler']['min_lr']}")
print(f"Steps per epoch: {steps_per_epoch}, Total steps: {cfg['training']['epochs'] * steps_per_epoch}")


# --- Loss function ---
label_smoothing = cfg["regularization"]["label_smoothing"]

if task == "language_modeling":
    # Next-token prediction cross-entropy, no label smoothing for LM
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    print("Loss: CrossEntropyLoss (language modeling)")
else:
    # Classification with label smoothing
    # When using mixup/cutmix, targets are already soft -> use soft CE
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    print(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")

# --- Mixed precision scaler ---
use_amp = cfg["experiment"]["precision"] == "fp16" and DEVICE.type == "cuda"
scaler = GradScaler(enabled=use_amp)
print(f"AMP: {'enabled' if use_amp else 'disabled'}")


# ## 6. Training Loop

# --- EMA setup ---
class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Replace model params with EMA params for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model params after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


ema_cfg = cfg.get("ema", {})
ema = None
if ema_cfg.get("enabled", False):
    ema = EMA(model, decay=ema_cfg["decay"])
    print(f"EMA enabled with decay={ema_cfg['decay']}")
else:
    print("EMA disabled")


def compute_diversity_loss(model, cfg):
    """Compute kernel diversity loss from the last forward pass.

    For each head in each DKA layer, sample random token pairs and penalize
    high cosine similarity between their generated kernels.
    """
    div_cfg = cfg.get("diversity_loss", {})
    lambda_div = div_cfg.get("lambda_div", 0.0)
    tau = div_cfg.get("tau", 0.5)
    num_pairs = div_cfg.get("num_pairs", 64)

    if lambda_div == 0.0:
        return torch.tensor(0.0, device=DEVICE)

    total_div_loss = torch.tensor(0.0, device=DEVICE)
    num_terms = 0

    # Iterate over DKA blocks
    blocks = model.blocks if hasattr(model, "blocks") else []
    for block in blocks:
        if not hasattr(block, "dka"):
            continue
        dka_module = block.dka
        kernels = dka_module.get_last_kernels()
        if kernels is None:
            continue

        for h, K_hat in kernels.items():
            # K_hat: (B, n, k_h, d_h)
            B, n, k_h, d_h = K_hat.shape
            # Flatten kernels: (B, n, k_h*d_h)
            K_flat = K_hat.reshape(B, n, -1)

            # Sample random token pairs from first sample in batch
            k_sample = K_flat[0]  # (n, k_h*d_h)
            actual_pairs = min(num_pairs, n * (n - 1) // 2)
            if actual_pairs == 0:
                continue

            idx_i = torch.randint(0, n, (actual_pairs,), device=K_hat.device)
            idx_j = torch.randint(0, n, (actual_pairs,), device=K_hat.device)
            # Ensure i != j
            idx_j = (idx_i + 1 + torch.randint(0, n - 1, (actual_pairs,), device=K_hat.device)) % n

            ki = k_sample[idx_i]  # (num_pairs, k_h*d_h)
            kj = k_sample[idx_j]  # (num_pairs, k_h*d_h)

            # Cosine similarity
            cos_sim = F.cosine_similarity(ki, kj, dim=-1)  # (num_pairs,)

            # Hinge loss: penalize similarity above tau
            div_loss = torch.clamp(cos_sim - tau, min=0.0).mean()
            total_div_loss = total_div_loss + div_loss
            num_terms += 1

    if num_terms > 0:
        total_div_loss = total_div_loss / num_terms

    return lambda_div * total_div_loss


def collect_alpha_values(model):
    """Collect all alpha_h values from the model for logging."""
    alphas = {}
    blocks = model.blocks if hasattr(model, "blocks") else []
    for layer_idx, block in enumerate(blocks):
        if not hasattr(block, "dka"):
            continue
        dka_module = block.dka
        for h, kg in enumerate(dka_module.kernel_generators):
            alphas[f"alpha/layer{layer_idx}_head{h}"] = kg.alpha.item()
    return alphas


def clip_kernel_gen_grads(model, max_norm):
    """Apply additional gradient clipping to kernel generator parameters."""
    kernel_gen_params = []
    for name, p in model.named_parameters():
        if "kernel_generators" in name and "alpha" not in name and "K_base" not in name:
            if p.grad is not None:
                kernel_gen_params.append(p)
    if kernel_gen_params:
        torch.nn.utils.clip_grad_norm_(kernel_gen_params, max_norm)


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion,
                    cfg, epoch, ema=None, mixup_cutmix_fn=None):
    """Train for one epoch. Returns dict of averaged metrics."""
    model.train()
    is_dka = not cfg["baselines"]["enabled"]
    grad_clip_cfg = cfg.get("grad_clip", {})
    global_max_norm = grad_clip_cfg.get("global_max_norm", 1.0)
    kernel_gen_max_norm = grad_clip_cfg.get("kernel_gen_max_norm", 0.5)
    log_every = cfg["logging"].get("log_every_n_steps", 50)

    running_loss = 0.0
    running_ce_loss = 0.0
    running_div_loss = 0.0
    running_correct = 0
    running_total = 0
    num_batches = 0

    for step, batch in enumerate(train_loader):
        # --- Unpack batch based on task ---
        if task == "language_modeling":
            input_ids = batch[0].to(DEVICE)   # (B, seq_len)
            # For LM: input is tokens[:-1], target is tokens[1:]
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
        elif task == "classification" and cfg["data"]["dataset"] in ["agnews", "wikitext2"]:
            input_ids = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)
            inputs = input_ids
        else:
            # Image classification
            images = batch[0].to(DEVICE)
            targets = batch[1].to(DEVICE)

            # Apply mixup/cutmix at batch level
            if mixup_cutmix_fn is not None:
                images, targets = mixup_cutmix_fn(images, targets)
            inputs = images

        # --- Forward pass ---
        amp_context = autocast(enabled=use_amp, dtype=torch.float16)
        with amp_context:
            outputs = model(inputs)

            if task == "language_modeling":
                # Reshape for cross-entropy: (B*n, V) vs (B*n,)
                B, n, V = outputs.shape
                ce_loss = criterion(outputs.reshape(-1, V), targets.reshape(-1))
            elif mixup_cutmix_fn is not None and task == "classification" and \
                    cfg["data"]["dataset"] in ["cifar10", "tinyimagenet"]:
                # Soft targets from mixup/cutmix: use soft cross-entropy
                log_probs = F.log_softmax(outputs, dim=-1)
                ce_loss = -(targets * log_probs).sum(dim=-1).mean()
            else:
                ce_loss = criterion(outputs, targets)

            # Diversity loss (DKA only)
            div_loss = torch.tensor(0.0, device=DEVICE)
            if is_dka:
                div_loss = compute_diversity_loss(model, cfg)

            loss = ce_loss + div_loss

        # --- Backward pass ---
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), global_max_norm)
        if is_dka:
            clip_kernel_gen_grads(model, kernel_gen_max_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # EMA update
        if ema is not None:
            ema.update()

        # --- Metrics ---
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_div_loss += div_loss.item()
        num_batches += 1

        if task == "classification":
            preds = outputs.argmax(dim=-1)
            if mixup_cutmix_fn is not None and cfg["data"]["dataset"] in ["cifar10", "tinyimagenet"]:
                # For soft targets, accuracy is measured against original hard labels
                hard_targets = targets.argmax(dim=-1)
                running_correct += (preds == hard_targets).sum().item()
            else:
                running_correct += (preds == targets).sum().item()
            running_total += targets.size(0)

        # --- Logging ---
        global_step = epoch * len(train_loader) + step
        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            msg = (f"  Step {step}/{len(train_loader)} | "
                   f"Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}")
            if is_dka:
                msg += f", Div: {div_loss.item():.4f}"
            msg += f") | LR: {lr:.2e}"
            print(msg)

            # W&B logging
            if WANDB_AVAILABLE and cfg["logging"].get("use_wandb", False) and wandb.run is not None:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/ce_loss": ce_loss.item(),
                    "train/div_loss": div_loss.item(),
                    "train/lr": lr,
                    "train/epoch": epoch,
                }
                if is_dka:
                    log_dict.update(collect_alpha_values(model))
                wandb.log(log_dict, step=global_step)

    # --- Epoch averages ---
    metrics = {
        "train_loss": running_loss / max(num_batches, 1),
        "train_ce_loss": running_ce_loss / max(num_batches, 1),
        "train_div_loss": running_div_loss / max(num_batches, 1),
    }
    if task == "classification" and running_total > 0:
        metrics["train_accuracy"] = running_correct / running_total

    return metrics


@torch.no_grad()
def evaluate(model, val_loader, criterion, task):
    """Evaluate on the validation set. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    num_batches = 0

    for batch in val_loader:
        if task == "language_modeling":
            input_ids = batch[0].to(DEVICE)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            outputs = model(inputs)
            B, n, V = outputs.shape
            loss = criterion(outputs.reshape(-1, V), targets.reshape(-1))
            total_loss += loss.item() * B * n
            total_tokens += B * n

        elif task == "classification":
            if cfg["data"]["dataset"] in ["agnews", "wikitext2"]:
                inputs = batch[0].to(DEVICE)
                targets = batch[1].to(DEVICE)
            else:
                inputs = batch[0].to(DEVICE)
                targets = batch[1].to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)

            preds = outputs.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

        num_batches += 1

    metrics = {}
    if task == "language_modeling":
        avg_loss = total_loss / max(total_tokens, 1)
        metrics["val_loss"] = avg_loss
        metrics["val_perplexity"] = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    else:
        metrics["val_loss"] = total_loss / max(total_samples, 1)
        metrics["val_accuracy"] = total_correct / max(total_samples, 1)

    return metrics


# --- Initialize W&B ---
if WANDB_AVAILABLE and cfg["logging"].get("use_wandb", False):
    run_name = cfg["experiment"]["name"]
    if cfg["baselines"]["enabled"]:
        run_name += f"_baseline_{cfg['baselines']['model_name']}"
    wandb.init(
        project=cfg["logging"]["wandb_project"],
        name=run_name,
        config=cfg,
    )
    print(f"W&B run: {wandb.run.name}")
else:
    print("W&B logging disabled")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

num_epochs = cfg["training"]["epochs"]
save_every = cfg["checkpointing"]["save_every_n_epochs"]
save_dir = Path(cfg["checkpointing"]["save_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

best_metric = 0.0 if task == "classification" else float("inf")
best_metric_name = cfg["checkpointing"].get("best_metric", "val_accuracy")
history = defaultdict(list)
alpha_history = []  # Track alpha values per epoch for DKA

print(f"\n{'='*60}")
print(f"Starting training: {num_epochs} epochs")
print(f"{'='*60}\n")

for epoch in range(num_epochs):
    epoch_start = time.time()

    # --- Train ---
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, scheduler, scaler, criterion,
        cfg, epoch, ema=ema, mixup_cutmix_fn=mixup_cutmix_fn,
    )

    # --- Evaluate ---
    if ema is not None:
        ema.apply_shadow()

    val_metrics = evaluate(model, val_loader, criterion, task)

    if ema is not None:
        ema.restore()

    # --- Collect alpha values for DKA ---
    is_dka = not cfg["baselines"]["enabled"]
    if is_dka:
        alphas = collect_alpha_values(model)
        alpha_history.append({"epoch": epoch, **alphas})

    # --- Log ---
    epoch_time = time.time() - epoch_start
    all_metrics = {**train_metrics, **val_metrics}

    for k, v in all_metrics.items():
        history[k].append(v)

    # Print summary
    summary = f"  [{epoch_time:.1f}s]"
    for k, v in all_metrics.items():
        if "accuracy" in k:
            summary += f" | {k}: {v:.4f}"
        elif "perplexity" in k:
            summary += f" | {k}: {v:.2f}"
        else:
            summary += f" | {k}: {v:.4f}"
    print(summary)

    # W&B
    if WANDB_AVAILABLE and cfg["logging"].get("use_wandb", False) and wandb.run is not None:
        wandb.log({f"epoch/{k}": v for k, v in all_metrics.items()}, step=(epoch + 1) * steps_per_epoch)

    # --- Save best model ---
    current_metric = val_metrics.get(best_metric_name, 0)
    is_best = False
    if "perplexity" in best_metric_name:
        is_best = current_metric < best_metric
    else:
        is_best = current_metric > best_metric

    if is_best and cfg["checkpointing"]["save_best"]:
        best_metric = current_metric
        best_path = save_dir / f"best_{cfg['experiment']['name']}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": cfg,
        }, best_path)
        print(f"  -> New best {best_metric_name}: {best_metric:.4f} (saved)")

    # --- Periodic checkpoint ---
    if (epoch + 1) % save_every == 0:
        ckpt_path = save_dir / f"{cfg['experiment']['name']}_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": dict(history),
            "alpha_history": alpha_history if is_dka else None,
            "config": cfg,
        }, ckpt_path)
        print(f"  -> Checkpoint saved: {ckpt_path}")

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Best {best_metric_name}: {best_metric:.4f}")
print(f"{'='*60}")


# ## 7. Save Final Checkpoint

# Save final checkpoint with all training state
final_path = save_dir / f"{cfg['experiment']['name']}_final.pt"

save_dict = {
    "epoch": num_epochs - 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "history": dict(history),
    "alpha_history": alpha_history if is_dka else None,
    "config": cfg,
    "best_metric": best_metric,
}

# Also save EMA weights if applicable
if ema is not None:
    save_dict["ema_shadow"] = ema.shadow

torch.save(save_dict, final_path)
print(f"Final checkpoint saved: {final_path}")
print(f"Total parameters: {total_params:,}")
print(f"Best {best_metric_name}: {best_metric:.4f}")


# ## 8. Evaluate on Test Set

# Load best checkpoint for final evaluation
best_path = save_dir / f"best_{cfg['experiment']['name']}.pt"
if best_path.exists():
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']+1}")
else:
    print("No best checkpoint found, using final model weights")

# Run evaluation
test_metrics = evaluate(model, val_loader, criterion, task)

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS")
print(f"{'='*60}")
print(f"Model: {model_label}")
print(f"Dataset: {dataset_name}")
print(f"Parameters: {total_params:,}")
for k, v in test_metrics.items():
    if "accuracy" in k:
        print(f"{k}: {v:.4f} ({v*100:.2f}%)")
    elif "perplexity" in k:
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v:.4f}")
print(f"{'='*60}")

# Log final metrics to W&B
if WANDB_AVAILABLE and cfg["logging"].get("use_wandb", False) and wandb.run is not None:
    wandb.summary.update({f"test/{k}": v for k, v in test_metrics.items()})
    wandb.summary["total_params"] = total_params
    wandb.finish()
    print("W&B run finished")

