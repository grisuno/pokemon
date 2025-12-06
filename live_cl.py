#!/usr/bin/env python3
"""
=============================================================================
🧬 OMNI BRAIN v8.3 - UNIFIED IMPLEMENTATION
=============================================================================
Synthesis of the best features from v8.2 variants:
- Robust ablation study framework
- Optimized baseline performance (~83%+)
- Clean architecture with proper fixes
- Professional logging and metrics tracking
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json

# =============================================================================
# 1. CONFIGURATION SYSTEM
# =============================================================================
@dataclass
class Config:
    """Centralized configuration with ablation flags"""
    # Ablation Flags - Baseline mode (all experimental features disabled)
    USE_FAST_SLOW: bool = False
    USE_INTEGRATION_INDEX: bool = False
    USE_DUAL_PATHWAY: bool = False
    USE_MEMORY_BUFFER: bool = False
    
    # Training Hyperparameters
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 5e-4
    
    # Fast Learning Parameters (for future experiments)
    fast_lr: float = 0.005
    fast_decay: float = 0.95
    fast_update_interval: int = 10
    
    # System Configuration
    log_interval: int = 200
    num_workers: int = 1
    use_wandb: bool = False  # Disabled by default for cleaner runs
    project_name: str = "omni-brain-v8.3"
    
    # Optimization
    grad_clip: float = 1.0
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# 2. LOGGING SETUP
# =============================================================================
def setup_logging():
    """Professional logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# =============================================================================
# 3. DETERMINISTIC SETUP
# =============================================================================
def set_seed(seed: int = 42):
    """Ensure reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = 'cpu'
torch.set_num_threads(max(1, os.cpu_count() // 2))


# =============================================================================
# 4. INTEGRATION INDEX COMPUTATION
# =============================================================================
def compute_integration_index(activity: torch.Tensor) -> float:
    """
    Compute neural integration using SVD (Singular Value Decomposition)
    Returns value in [0, 1] representing degree of neural coordination
    """
    if activity.numel() < 100 or activity.size(0) < 5:
        return 0.0
    
    with torch.no_grad():
        # Center the activity
        activity = activity - activity.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.mm(activity.t(), activity) / (activity.size(0) - 1)
        
        try:
            # SVD for numerical stability
            _, s, _ = torch.linalg.svd(cov)
            total_var = s.sum()
            
            if total_var < 1e-8:
                return 0.0
            
            # First singular value ratio as integration measure
            integration_ratio = (s[0] / total_var).item()
            return max(0.0, min(1.0, integration_ratio))
            
        except RuntimeError:
            return 0.0


# =============================================================================
# 5. NEURAL MODULES
# =============================================================================
class FastSlowLinear(nn.Module):
    """
    Dual-system linear layer with fast (Hebbian) and slow (gradient) learning.
    Fast learning is disabled in baseline mode but structure is maintained.
    """
    def __init__(self, in_features: int, out_features: int, config: Config):
        super().__init__()
        self.config = config
        
        # Slow weights (gradient-based)
        self.slow_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.slow_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.slow_weight)
        
        # Fast weights (Hebbian learning)
        self.register_buffer('fast_weight', torch.zeros(out_features, in_features))
        self.register_buffer('fast_bias', torch.zeros(out_features))
        self.register_buffer('update_counter', torch.tensor(0))
        self.register_buffer('fast_weight_norm', torch.tensor(0.0))
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_features)
    
    def reset_fast_weights(self):
        """Reset fast weights (memory purge)"""
        self.fast_weight.zero_()
        self.fast_bias.zero_()
        self.fast_weight_norm.zero_()
    
    def update_fast_weights(self, x: torch.Tensor, slow_out: torch.Tensor):
        """Hebbian learning update"""
        self.update_counter += 1
        
        if self.update_counter % self.config.fast_update_interval != 0:
            return
        
        with torch.no_grad():
            # Temporal decay
            self.fast_weight.mul_(self.config.fast_decay)
            self.fast_bias.mul_(self.config.fast_decay)
            
            # Hebbian update
            hebb_update = torch.mm(slow_out.t(), x) / x.size(0)
            self.fast_weight.add_(hebb_update, alpha=self.config.fast_lr)
            
            # Norm constraint (homeostasis)
            current_norm = self.fast_weight.norm()
            if current_norm > 1.0:
                self.fast_weight.mul_(1.0 / (current_norm + 1e-8))
            
            # Bias update with clamping
            self.fast_bias.add_(slow_out.mean(dim=0), alpha=self.config.fast_lr)
            self.fast_bias.clamp_(-0.2, 0.2)
            
            self.fast_weight_norm = self.fast_weight.norm()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slow_out = F.linear(x, self.slow_weight, self.slow_bias)
        
        if not self.config.USE_FAST_SLOW:
            return self.norm(slow_out)
        
        if self.training:
            self.update_fast_weights(x.detach(), slow_out.detach())
        
        effective_w = self.slow_weight + self.fast_weight
        effective_b = self.slow_bias + self.fast_bias
        return self.norm(F.linear(x, effective_w, effective_b))
    
    def get_fast_norm(self) -> float:
        return self.fast_weight_norm.item()


class DualSystemModule(nn.Module):
    """
    Dual-pathway processing with fast and slow streams.
    Bypassed in baseline mode but ready for activation.
    """
    def __init__(self, dim: int, config: Config):
        super().__init__()
        self.config = config
        
        if config.USE_DUAL_PATHWAY:
            self.fast_path = FastSlowLinear(dim, dim, config)
            self.slow_path = FastSlowLinear(dim, dim, config)
            self.integrator = nn.Linear(dim * 2, dim)
        
        self.register_buffer('memory', torch.zeros(1, dim))
        self.tau = 0.95
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update memory buffer
        if self.config.USE_MEMORY_BUFFER:
            with torch.no_grad():
                self.memory = self.tau * self.memory + (1 - self.tau) * x.mean(dim=0, keepdim=True).detach()
        
        # Bypass if not using dual pathway
        if not self.config.USE_DUAL_PATHWAY:
            return x
        
        # Dual processing
        fast_out = self.fast_path(x)
        slow_in = x + (self.memory if self.config.USE_MEMORY_BUFFER else 0)
        slow_out = self.slow_path(slow_in)
        
        combined = torch.cat([fast_out, slow_out], dim=1)
        return self.integrator(combined)


class IntegrationModule(nn.Module):
    """
    Neural integration module with adaptive gating.
    Bypassed in baseline mode.
    """
    def __init__(self, features: int, config: Config):
        super().__init__()
        self.config = config
        
        self.integration_net = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )
        
        self.integration_threshold = 0.2
        self.register_buffer('running_index', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.USE_INTEGRATION_INDEX:
            return x
        
        if self.training:
            idx = compute_integration_index(x)
            self.running_index = 0.9 * self.running_index + 0.1 * idx
        
        # Adaptive integration
        if self.running_index > self.integration_threshold:
            return self.integration_net(x)
        else:
            return x + 0.1 * self.integration_net(x)


# =============================================================================
# 6. MAIN ARCHITECTURE
# =============================================================================
class OmniBrain(nn.Module):
    """
    Unified Omni Brain architecture with configurable modules.
    Optimized baseline with experimental features ready for activation.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Visual encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Core processing
        self.core = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Experimental modules (bypassed in baseline)
        self.dual = DualSystemModule(256, config)
        self.integration = IntegrationModule(256, config)
        
        # Classifier
        self.classifier = nn.Linear(256, 10)
        
        # Tracking
        self.register_buffer('batch_count', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.core(h)
        h = self.dual(h)
        h = self.integration(h)
        return self.classifier(h)
    
    def reset_all_fast_weights(self):
        """Reset all fast weights in the network"""
        for module in self.modules():
            if hasattr(module, 'reset_fast_weights'):
                module.reset_fast_weights()
    
    def get_fast_norms(self) -> List[float]:
        """Collect fast weight norms for monitoring"""
        return [m.get_fast_norm() for m in self.modules() 
                if hasattr(m, 'get_fast_norm')]
    
    def get_ablation_state(self) -> Dict[str, bool]:
        """Return current ablation configuration"""
        return {
            "fast_slow_active": self.config.USE_FAST_SLOW,
            "dual_pathway_active": self.config.USE_DUAL_PATHWAY,
            "integration_active": self.config.USE_INTEGRATION_INDEX,
            "memory_active": self.config.USE_MEMORY_BUFFER
        }


# =============================================================================
# 7. DATA LOADING
# =============================================================================
def get_data_loaders(config: Config):
    """Prepare CIFAR-10 data loaders with augmentation"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616)),
    ])
    
    train_ds = datasets.CIFAR10('./data', train=True, download=True, 
                               transform=transform_train)
    test_ds = datasets.CIFAR10('./data', train=False, 
                              transform=transform_test)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        pin_memory=True
    )
    
    return train_loader, test_loader


# =============================================================================
# 8. EVALUATION
# =============================================================================
def evaluate(model: nn.Module, loader: DataLoader, 
             device: str) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    losses = []
    integration_indices = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            loss = criterion(logits, y)
            losses.append(loss.item())
            
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)
            
            # Collect integration index if active
            if hasattr(model, 'integration') and model.config.USE_INTEGRATION_INDEX:
                integration_indices.append(model.integration.running_index.item())
    
    return {
        "accuracy": correct / total,
        "avg_loss": np.mean(losses),
        "integration_index": np.mean(integration_indices) if integration_indices else 0.0,
        "fast_norms": model.get_fast_norms()
    }


# =============================================================================
# 9. TRAINING ENGINE
# =============================================================================
def train(config: Config, silent: bool = False) -> Dict[str, List[float]]:
    """
    Main training loop with comprehensive logging
    
    Args:
        config: Configuration object
        silent: If True, reduce logging for ablation studies
    
    Returns:
        Dictionary of training metrics history
    """
    if not silent:
        logger.info("=" * 80)
        logger.info("🧠 OMNI BRAIN v8.3 - Unified Implementation")
        logger.info("=" * 80)
        logger.info(f"Configuration: {config.to_dict()}")
    
    # Prepare data and model
    train_loader, test_loader = get_data_loaders(config)
    model = OmniBrain(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if not silent:
        logger.info(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")
        logger.info(f"Ablation State: {model.get_ablation_state()}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.scheduler_T0, 
        T_mult=config.scheduler_Tmult,
        eta_min=1e-5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    metrics_history = {
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "integration_index": [],
        "fast_norm": [],
        "lr": []
    }
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        
        # Periodic fast weight reset
        if config.USE_FAST_SLOW and epoch % 10 == 0:
            model.reset_all_fast_weights()
        
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            
            # L2 regularization (light)
            l2_reg = sum(torch.norm(p) for p in model.parameters()) * 1e-5
            loss = loss + l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Batch logging
            if not silent and batch_idx % config.log_interval == 0:
                avg_loss = total_loss / total_samples
                fast_norms = model.get_fast_norms()
                avg_fast = np.mean(fast_norms) if fast_norms else 0.0
                
                logger.info(
                    f"Epoch {epoch+1:02d}/{config.epochs} | "
                    f"Batch {batch_idx:04d}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"FastNorm: {avg_fast:.3f}"
                )
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # Evaluation
        eval_metrics = evaluate(model, test_loader, device)
        
        # Epoch logging
        if not silent or epoch == config.epochs - 1:
            logger.info(
                f"\n📊 Epoch {epoch+1:02d}/{config.epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {total_loss/total_samples:.4f} | "
                f"Test Acc: {eval_metrics['accuracy']:.2%} | "
                f"Test Loss: {eval_metrics['avg_loss']:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
        
        # Store metrics
        metrics_history["train_loss"].append(total_loss / total_samples)
        metrics_history["test_acc"].append(eval_metrics['accuracy'])
        metrics_history["test_loss"].append(eval_metrics['avg_loss'])
        metrics_history["integration_index"].append(eval_metrics['integration_index'])
        metrics_history["lr"].append(scheduler.get_last_lr()[0])
        
        avg_fast = np.mean(eval_metrics['fast_norms']) if eval_metrics['fast_norms'] else 0.0
        metrics_history["fast_norm"].append(avg_fast)
    
    # Save model
    if not silent:
        save_path = f"omni_brain_v83_{config.project_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'metrics': metrics_history
        }, save_path)
        logger.info(f"\n✅ Model saved to {save_path}")
    
    return metrics_history


# =============================================================================
# 10. ABLATION STUDY
# =============================================================================
def run_ablation_study(quick_test: bool = True) -> Dict[str, float]:
    """
    Comprehensive ablation study across different configurations
    
    Args:
        quick_test: If True, run 5 epochs per config; else full 50 epochs
    
    Returns:
        Dictionary mapping configuration names to final test accuracies
    """
    logger.info("\n" + "=" * 80)
    logger.info("🧪 ABLATION STUDY - OMNI BRAIN v8.3")
    logger.info("=" * 80)
    
    test_epochs = 5 if quick_test else 50
    
    scenarios = [
        {
            "name": "baseline",
            "desc": "Pure baseline (no experimental features)",
            "config": {
                "USE_FAST_SLOW": False,
                "USE_DUAL_PATHWAY": False,
                "USE_INTEGRATION_INDEX": False,
                "USE_MEMORY_BUFFER": False
            }
        },
        {
            "name": "fast_slow",
            "desc": "Fast-slow learning system",
            "config": {
                "USE_FAST_SLOW": True,
                "USE_DUAL_PATHWAY": False,
                "USE_INTEGRATION_INDEX": False,
                "USE_MEMORY_BUFFER": False
            }
        },
        {
            "name": "dual_pathway",
            "desc": "Dual processing pathways",
            "config": {
                "USE_FAST_SLOW": False,
                "USE_DUAL_PATHWAY": True,
                "USE_INTEGRATION_INDEX": False,
                "USE_MEMORY_BUFFER": True
            }
        },
        {
            "name": "integration",
            "desc": "Integration index modulation",
            "config": {
                "USE_FAST_SLOW": False,
                "USE_DUAL_PATHWAY": False,
                "USE_INTEGRATION_INDEX": True,
                "USE_MEMORY_BUFFER": False
            }
        },
        {
            "name": "full_system",
            "desc": "All experimental features active",
            "config": {
                "USE_FAST_SLOW": True,
                "USE_DUAL_PATHWAY": True,
                "USE_INTEGRATION_INDEX": True,
                "USE_MEMORY_BUFFER": True
            }
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Testing: {scenario['name'].upper()}")
        logger.info(f"Description: {scenario['desc']}")
        logger.info(f"{'─' * 80}")
        
        # Create config
        config = Config()
        config.epochs = test_epochs
        config.project_name = f"ablation_{scenario['name']}"
        
        # Apply scenario settings
        for key, value in scenario['config'].items():
            setattr(config, key, value)
        
        # Reset seed for fair comparison
        set_seed(42)
        
        # Train
        metrics = train(config, silent=True)
        final_acc = metrics['test_acc'][-1]
        results[scenario['name']] = final_acc
        
        logger.info(f"✓ Final accuracy: {final_acc:.2%}")
    
    # Summary report
    logger.info("\n" + "=" * 80)
    logger.info("📊 ABLATION STUDY RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Configuration':<20} | {'Test Accuracy':<15}")
    logger.info("─" * 80)
    
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{name:<20} | {acc:.2%}")
    
    logger.info("=" * 80)
    
    # Save results
    with open("ablation_results.json", "w") as f:
        json.dump({
            "test_type": "quick" if quick_test else "full",
            "epochs_per_config": test_epochs,
            "results": {k: f"{v:.4f}" for k, v in results.items()}
        }, f, indent=2)
    
    logger.info("\n✅ Results saved to ablation_results.json")
    
    return results


# =============================================================================
# 11. MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        # Run ablation study
        quick = "--quick" in sys.argv
        run_ablation_study(quick_test=quick)
    else:
        # Run standard baseline training
        config = Config()
        train(config)
        
        logger.info("\n" + "─" * 80)
        logger.info("💡 TIP: Run with 'python script.py ablation' to test all configurations")
        logger.info("    Add '--quick' flag for fast 5-epoch tests")
        logger.info("─" * 80)