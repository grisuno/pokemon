"""
================================================================================
NeuroLogos v5.1 - Ablación Científica Rigurosa (Vision-Language)
================================================================================
Matriz de ablación: 11 experimentos | Validación cruzada | Análisis estadístico
Componentes: GRID (G) | SYMBIOTIC (S) | ADVERSARIAL (A)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
from scipy import stats
from collections import defaultdict
import json
import time
from pathlib import Path

def seed_everything(seed: int):
    """Control total de reproducibilidad"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_effect_size(group1: List[float], group2: List[float]) -> float:
    """Cohen's d con corrección de sesgo"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / (pooled_std + 1e-8)

# ============================================================================
# CONFIGURACIÓN CIENTÍFICA
# ============================================================================
@dataclass
class NeuroLogosConfig:
    """Configuración ablacionable para NeuroLogos"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Componentes ablacionables
    use_grid: bool = False
    use_symbiotic: bool = False
    use_adversarial: bool = False
    
    # Hiperparámetros entrenamiento
    batch_size: int = 64
    epochs: int = 30
    lr: float = 0.001
    weight_decay: float = 1e-5
    
    # Parámetros adversariales
    adv_epsilon: float = 0.1
    adv_alpha: float = 0.01
    adv_steps: int = 5
    adv_start_epoch: int = 5
    
    # Regularización
    lambda_density: float = 0.01  # Penaliza colapso topológico
    
    # Estabilidad numérica
    clip_value: float = 1.0
    
    # Validación cruzada
    cv_folds: int = 3
    
    def to_dict(self):
        return asdict(self)
    
    def component_signature(self) -> str:
        components = []
        if self.use_grid: components.append("G")
        if self.use_symbiotic: components.append("S")
        if self.use_adversarial: components.append("A")
        return "".join(components) if components else "BASE"

        
# ============================================================================
# TU ARQUITECTURA ORIGINAL (Sin modificaciones)
# ============================================================================
class TopoBrainCore(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, output_dim=512, 
                 grid_size=4, use_grid=True, use_symbiotic=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.n_nodes = grid_size * grid_size
        self.use_grid = use_grid
        self.use_symbiotic = use_symbiotic
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.n_nodes)
        
        if self.use_grid:
            self.register_buffer('grid_coords', self._init_grid())
            self.plasticity = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        
        if self.use_symbiotic:
            self.symbiotic = nn.Linear(self.n_nodes, self.n_nodes, bias=False)
            nn.init.orthogonal_(self.symbiotic.weight)
        
        self.fc3 = nn.Linear(self.n_nodes, output_dim)
        self.last_density = 0.0
        
    def _init_grid(self):
        coords = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                coords.append([i / self.grid_size, j / self.grid_size])
        return torch.tensor(coords, dtype=torch.float32)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        
        if self.use_grid:
            distances = torch.cdist(self.grid_coords, self.grid_coords)
            topology = torch.exp(-distances * 2.0)
            plastic_weights = topology * torch.sigmoid(self.plasticity)
            h2_plastic = h2 @ plastic_weights
            with torch.no_grad():
                active_nodes = (h2.abs() > 0.1).float().mean()
                self.last_density = active_nodes.item()
        else:
            h2_plastic = h2
            self.last_density = 1.0
        
        if self.use_symbiotic:
            h2_refined = self.symbiotic(h2_plastic)
        else:
            h2_refined = h2_plastic
        
        output = self.fc3(F.relu(h2_refined))
        return output
    
    def get_metrics(self):
        metrics = {'density': self.last_density}
        if self.use_grid:
            metrics['plasticity_norm'] = self.plasticity.norm().item()
        return metrics

class PGDAttack:
    def __init__(self, epsilon=0.1, alpha=0.01, steps=5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def attack(self, model_fn, x, y, criterion):
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        
        for _ in range(self.steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            
            outputs = model_fn(x_adv)
            loss = criterion(outputs, y)
            loss.backward()
            
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
                x_adv = torch.clamp(x_adv, -1, 1)
            
            x_adv.requires_grad = True
        
        return x_adv.detach()

class MiniUnconscious(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.topo_bridge = nn.Linear(256*4, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        features = self.stem(x)
        encoded = self.topo_bridge(features)
        return self.norm(encoded)

class TopoUnconscious(nn.Module):
    def __init__(self, output_dim=512, use_grid=True, use_symbiotic=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()
        )
        self.topobrain = TopoBrainCore(
            input_dim=256*4, hidden_dim=64, output_dim=output_dim,
            grid_size=4, use_grid=use_grid, use_symbiotic=use_symbiotic
        )
        
    def forward(self, x):
        features = self.stem(x)
        return self.topobrain(features)
    
    def get_metrics(self):
        return self.topobrain.get_metrics()

class ConsciousCore(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        q = x.unsqueeze(1)
        attended, _ = self.attention(q, q, q)
        return self.norm(attended.squeeze(1))

class BioDecoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, thought, captions=None, max_len=20):
        batch_size = thought.size(0)
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            lstm_out, _ = self.lstm(embeddings, self._get_init_state(thought))
            gate = self.liquid_gate(lstm_out)
            lstm_out = lstm_out * gate
            return self.out(lstm_out)
        else:
            generated = []
            input_word = torch.full((batch_size, 1), 1, dtype=torch.long, device=thought.device)
            hidden = self._get_init_state(thought)
            
            for _ in range(max_len):
                emb = self.embedding(input_word)
                out, hidden = self.lstm(emb, hidden)
                gate = self.liquid_gate(out)
                out = out * gate
                logits = self.out(out).squeeze(1) / 0.8
                probs = F.softmax(logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                generated.append(next_word)
                input_word = next_word
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, thought):
        h0 = thought.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

# ============================================================================
# MODELO PRINCIPAL CON CONFIGURACIÓN ABLACIONABLE
# ============================================================================
class NeuroLogos(nn.Module):
    def __init__(self, vocab_size=1000, config: NeuroLogosConfig = None):
        super().__init__()
        self.config = config or NeuroLogosConfig()
        
        # Visual encoder según ablación
        if not self.config.use_grid and not self.config.use_symbiotic:
            self.eye = MiniUnconscious(output_dim=512)
        else:
            self.eye = TopoUnconscious(
                output_dim=512,
                use_grid=self.config.use_grid,
                use_symbiotic=self.config.use_symbiotic
            )
        
        # Core components
        self.cortex = ConsciousCore(dim=512)
        self.broca = BioDecoder(vocab_size, embed_dim=128, hidden_dim=512)
        
        # Adversarial
        self.pgd = None
        if self.config.use_adversarial:
            self.pgd = PGDAttack(
                epsilon=self.config.adv_epsilon,
                alpha=self.config.adv_alpha,
                steps=self.config.adv_steps
            )
    
    def forward(self, image, captions=None):
        visual = self.eye(image)
        thought = self.cortex(visual)
        return self.broca(thought, captions)
    
    def get_metrics(self):
        return self.eye.get_metrics() if hasattr(self.eye, 'get_metrics') else {}

# ============================================================================
# DATASET
# ============================================================================
class CIFARCaptions:
    def __init__(self):
        self.dataset = datasets.CIFAR10('./data', train=True, download=True,
                                       transform=transforms.ToTensor())
        self.templates = {
            0: ["a red airplane in the sky", "a silver aircraft with wings"],
            1: ["a shiny yellow car", "a red automobile on asphalt"],
            2: ["a small bird with feathers", "a flying sparrow in daylight"],
            3: ["a black domestic cat", "a furry feline sitting still"],
            4: ["a wild deer in forest", "a brown animal with antlers"],
            5: ["a loyal brown dog", "a playful canine running"],
            6: ["a green frog on lily pad", "a moist amphibian near pond"],
            7: ["a strong brown horse", "a galloping equine in field"],
            8: ["a blue cargo ship", "a white vessel on ocean"],
            9: ["a large delivery truck", "a heavy-duty cargo vehicle"]
        }
        
        self.vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for descs in self.templates.values():
            for desc in descs:
                self.vocab.extend(desc.split())
        self.vocab = list(dict.fromkeys(self.vocab))
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        desc = np.random.choice(self.templates[label])
        tokens = ["<BOS>"] + desc.split() + ["<EOS>"]
        token_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in tokens]
        token_ids = token_ids[:20] + [self.word2id["<PAD>"]] * (20 - len(token_ids))
        return image, torch.tensor(token_ids, dtype=torch.long), label

# ============================================================================
# SISTEMA DE ABLACIÓN CIENTÍFICA
# ============================================================================
class AblationMatrix:
    """Matriz de ablación para 3 componentes (G, S, A)"""
    COMPONENTS = {'G': 'use_grid', 'S': 'use_symbiotic', 'A': 'use_adversarial'}
    
    @staticmethod
    def level1_isolated():
        return [
            ("00_Baseline", {}),
            ("01_Grid_Only", {"use_grid": True}),
            ("02_Symbiotic_Only", {"use_symbiotic": True}),
            ("03_Adversarial_Only", {"use_adversarial": True}),
        ]
    
    @staticmethod
    def level2_pairs():
        return [
            ("04_Pair_G+S", {"use_grid": True, "use_symbiotic": True}),
            ("05_Pair_G+A", {"use_grid": True, "use_adversarial": True}),
            ("06_Pair_S+A", {"use_symbiotic": True, "use_adversarial": True}),
        ]
    
    @staticmethod
    def level3_full():
        return [("07_Full_G+S+A", {"use_grid": True, "use_symbiotic": True, "use_adversarial": True})]
    
    @staticmethod
    def level4_inverse():
        return [
            ("08_Without_G", {"use_symbiotic": True, "use_adversarial": True}),
            ("09_Without_S", {"use_grid": True, "use_adversarial": True}),
            ("10_Without_A", {"use_grid": True, "use_symbiotic": True}),
        ]
    
    @classmethod
    def get_complete_matrix(cls):
        return cls.level1_isolated() + cls.level2_pairs() + cls.level3_full() + cls.level4_inverse()

class ScientificAnalyzer:
    @staticmethod
    def compute_statistics(cv_results: List[Dict]) -> Dict:
        test_accs = [r['final_test_acc'] for r in cv_results]
        train_accs = [r['final_train_acc'] for r in cv_results]
        densities = [r['final_density'] for r in cv_results]
        
        return {
            'test_mean': np.mean(test_accs),
            'test_std': np.std(test_accs, ddof=1),
            'test_sem': stats.sem(test_accs),
            'test_ci95': stats.t.interval(0.95, len(test_accs)-1,
                                         loc=np.mean(test_accs),
                                         scale=stats.sem(test_accs)),
            'train_mean': np.mean(train_accs),
            'gap_mean': np.mean(train_accs) - np.mean(test_accs),
            'density_mean': np.mean(densities),
        }
    
    @staticmethod
    def ttest_vs_baseline(exp_scores: List[float], baseline_scores: List[float]):
        t_stat, p_val = stats.ttest_ind(exp_scores, baseline_scores)
        cohens_d = compute_effect_size(exp_scores, baseline_scores)
        return t_stat, p_val, cohens_d
    
    @staticmethod
    def detect_synergy(pair_score, comp_a_score, comp_b_score, baseline_score):
        expected = comp_a_score + comp_b_score - baseline_score
        synergy = pair_score - expected
        
        if synergy > 2.0:
            interaction = "COOPERACIÓN"
        elif synergy < -2.0:
            interaction = "ANTAGONISMO"
        else:
            interaction = "ADITIVO"
        return synergy, interaction
    
    @staticmethod
    def rank_criticality(full_score, ablation_results: Dict):
        rankings = []
        for name, result in ablation_results.items():
            without_score = result['statistics']['test_mean']
            criticality = full_score - without_score
            
            if criticality > 5:
                verdict = "🔥 ESENCIAL"
            elif criticality > 2:
                verdict = "⚠️ IMPORTANTE"
            elif criticality > -2:
                verdict = "🟡 OPCIONAL"
            else:
                verdict = "🗑️ PERJUDICIAL"
            
            rankings.append((name, criticality, verdict, without_score))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

# ============================================================================
# ENTRENAMIENTO CON VALIDACIÓN CRUZADA
# ============================================================================
def train_epoch_cv(model, loader, optimizer, config: NeuroLogosConfig, epoch: int, vocab):
    model.train()
    metrics = defaultdict(float)
    total_batches = len(loader)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    
    for images, captions, _ in loader:
        images = images.to(config.device) * 2 - 1
        captions = captions.to(config.device)
        
        optimizer.zero_grad()
        
        # Forward limpio
        logits = model(images, captions)
        loss_ce = criterion(logits.reshape(-1, len(vocab)), captions[:, 1:].reshape(-1))
        
        # Adversarial (si activo y después de adv_start_epoch)
        use_adv = config.use_adversarial and epoch >= config.adv_start_epoch
        if use_adv:
            def model_fn(x_adv):
                return model(x_adv, captions)
            
            def crit_fn(out, tgt):
                return criterion(out.reshape(-1, len(vocab)), tgt[:, 1:].reshape(-1))
            
            images_adv = model.pgd.attack(model_fn, images, captions, crit_fn)
            logits_adv = model(images_adv, captions)
            loss_adv = criterion(logits_adv.reshape(-1, len(vocab)), captions[:, 1:].reshape(-1))
            loss = 0.7 * loss_ce + 0.3 * loss_adv
        else:
            loss = loss_ce
        
        # Penalización de densidad
        if hasattr(model, 'get_metrics'):
            density = model.get_metrics().get('density', 1.0)
            if density < 0.1:
                loss += config.lambda_density * (0.1 - density)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
        optimizer.step()
        
        # Métricas
        pred = logits.argmax(dim=-1)
        mask = captions[:, 1:] != vocab["<PAD>"]
        acc = ((pred == captions[:, 1:]) & mask).sum().item() / mask.sum().item()
        metrics['loss'] += loss.item()
        metrics['acc'] += acc
        metrics['density'] += model.get_metrics().get('density', 1.0)
        metrics['count'] += 1
    
    return {k: v / metrics['count'] for k, v in metrics.items() if k != 'count'}

@torch.no_grad()
def evaluate_cv(model, loader, config: NeuroLogosConfig, vocab):
    model.eval()
    correct = total = 0
    
    for images, captions, _ in loader:
        images = images.to(config.device) * 2 - 1
        captions = captions.to(config.device)
        logits = model(images, captions)
        
        pred = logits.argmax(dim=-1)
        mask = captions[:, 1:] != vocab["<PAD>"]
        correct += ((pred == captions[:, 1:]) & mask).sum().item()
        total += mask.sum().item()
    
    return 100.0 * correct / total

def train_with_cv(config: NeuroLogosConfig, dataset, vocab):
    # Etiquetas para stratificación
    labels = [dataset[i][2] for i in range(len(dataset))]
    
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"        Fold {fold_idx + 1}/{config.cv_folds}...", end=" ")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        
        # Modelo nuevo por fold
        model = NeuroLogos(vocab_size=dataset.vocab_size, config=config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # Entrenamiento
        best_val_acc = 0.0
        history = []
        
        for epoch in range(config.epochs):
            train_metrics = train_epoch_cv(model, train_loader, optimizer, config, epoch, vocab)
            val_acc = evaluate_cv(model, val_loader, config, vocab)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['acc'],
                'val_acc': val_acc,
                'density': train_metrics['density']
            })
        
        # Resultados del fold
        final_test_acc = evaluate_cv(model, val_loader, config, vocab)
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'final_test_acc': final_test_acc,
            'final_train_acc': history[-1]['train_acc'],
            'final_density': history[-1]['density'],
            'history': history
        })
        
        print(f"Done (Val: {final_test_acc:.2f}%)")
    
    return fold_results

# ============================================================================
# EJECUTOR PRINCIPAL
# ============================================================================
def run_scientific_ablation():
    seed_everything(42)
    results_dir = Path("neuro_logos_ablation")
    results_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧠 NeuroLogos v5.1 - Ablación Científica Rigurosa")
    print("="*80)
    print(f"📊 11 experimentos | 3 folds CV | Stats: t-test, Cohen's d, CI95")
    print(f"⚙️ Componentes: GRID, SYMBIOTIC, ADVERSARIAL")
    print("="*80 + "\n")
    
    # Dataset
    dataset = CIFARCaptions()
    vocab = dataset.word2id
    
    # Matriz de experimentos
    ablation_matrix = AblationMatrix.get_complete_matrix()
    all_results = {}
    
    # =========================================================================
    # FASE 1: EJECUTAR EXPERIMENTOS
    # =========================================================================
    for exp_idx, (exp_name, overrides) in enumerate(ablation_matrix):
        print(f"[{exp_idx+1:02d}/11] {exp_name}")
        
        # Configurar experimento
        config = NeuroLogosConfig()
        for key, value in overrides.items():
            setattr(config, key, value)
        
        # Entrenar con CV
        start_time = time.time()
        cv_results = train_with_cv(config, dataset, vocab)
        elapsed = time.time() - start_time
        
        # Estadísticas
        stats_dict = ScientificAnalyzer.compute_statistics(cv_results)
        
        # Contar parámetros
        temp_model = NeuroLogos(vocab_size=dataset.vocab_size, config=config)
        n_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        
        all_results[exp_name] = {
            'config': config.to_dict(),
            'cv_results': cv_results,
            'statistics': stats_dict,
            'n_params': n_params,
            'elapsed_time': elapsed,
            'component_signature': config.component_signature()
        }
        
        # Resumen rápido
        print(f"      ✅ Test: {stats_dict['test_mean']:.2f}%±{stats_dict['test_std']:.2f}% "
              f"| Params: {n_params:,} | Time: {elapsed:.1f}s\n")
    
    # Guardar resultados
    with open(results_dir / "raw_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # =========================================================================
    # FASE 2: ANÁLISIS ESTADÍSTICO
    # =========================================================================
    print("\n" + "="*80)
    print("📊 ANÁLISIS ESTADÍSTICO - 4 NIVELES")
    print("="*80)
    
    baseline = all_results['00_Baseline']
    baseline_scores = [r['final_test_acc'] for r in baseline['cv_results']]
    
    # NIVEL 1: Contribución Individual
    print("\n🔬 NIVEL 1: Contribución Individual")
    print("-"*80)
    print(f"{'Componente':<20} {'Test Acc':<15} {'Δ vs Base':<12} {'p-value':<10} {'Cohen d':<10}")
    print("-"*80)
    
    individual_results = {}
    for exp_name in ['01_Grid_Only', '02_Symbiotic_Only', '03_Adversarial_Only']:
        result = all_results[exp_name]
        exp_scores = [r['final_test_acc'] for r in result['cv_results']]
        stats_dict = result['statistics']
        
        t_stat, p_val, cohens_d = ScientificAnalyzer.ttest_vs_baseline(exp_scores, baseline_scores)
        delta = stats_dict['test_mean'] - baseline['statistics']['test_mean']
        
        component = exp_name.split('_')[1]
        individual_results[component] = stats_dict['test_mean']
        
        print(f"{component:<20} {stats_dict['test_mean']:>6.2f}%±{stats_dict['test_std']:<5.2f} "
              f"{delta:>+10.2f}% {p_val:>9.4f} {cohens_d:>9.3f}")
    
    # NIVEL 2: Sinergias
    print("\n🤝 NIVEL 2: Detección de Sinergias")
    print("-"*80)
    print(f"{'Par':<15} {'Real':<10} {'Esperado':<10} {'Sinergia':<10} {'Tipo':<15}")
    print("-"*80)
    
    pair_map = {
        '04_Pair_G+S': ('Grid', 'Symbiotic'),
        '05_Pair_G+A': ('Grid', 'Adversarial'),
        '06_Pair_S+A': ('Symbiotic', 'Adversarial')
    }
    
    for exp_name, (comp_a, comp_b) in pair_map.items():
        result = all_results[exp_name]
        pair_score = result['statistics']['test_mean']
        pgd_a = individual_results.get(comp_a, baseline['statistics']['test_mean'])
        pgd_b = individual_results.get(comp_b, baseline['statistics']['test_mean'])
        
        synergy, interaction = ScientificAnalyzer.detect_synergy(
            pair_score, pgd_a, pgd_b, baseline['statistics']['test_mean']
        )
        
        expected = pgd_a + pgd_b - baseline['statistics']['test_mean']
        emoji = "🟢" if interaction == "COOPERACIÓN" else "🔴" if interaction == "ANTAGONISMO" else "🟡"
        print(f"{comp_a[:4]}+{comp_b[:4]:<10} {pair_score:>6.2f}% {expected:>8.2f}% "
              f"{synergy:>+8.2f}% {emoji} {interaction:<12}")
    
    # NIVEL 3: Modelo Completo
    full_model = all_results['07_Full_G+S+A']
    full_score = full_model['statistics']['test_mean']
    
    # NIVEL 4: Ablación Inversa
    print("\n⚡ NIVEL 4: Análisis de Criticidad")
    print("-"*80)
    print(f"{'Removido':<15} {'Score (2/3)':<12} {'Criticidad':<12} {'Veredicto':<15}")
    print("-"*80)
    
    inverse_map = {
        '08_Without_G': 'Grid',
        '09_Without_S': 'Symbiotic',
        '10_Without_A': 'Adversarial'
    }
    
    rankings = ScientificAnalyzer.rank_criticality(full_score, {
        name: all_results[exp_name] for name, _ in inverse_map.items()
    })
    
    for name, criticality, verdict, without_score in rankings:
        print(f"{inverse_map[name]:<15} {without_score:>8.2f}% {criticality:>+10.2f}% {verdict:<15}")
    
    # RESUMEN EJECUTIVO
    print("\n" + "="*80)
    print("📋 RESUMEN EJECUTIVO")
    print("="*80)
    
    # Top 5 configuraciones
    sorted_results = sorted(all_results.items(),
                           key=lambda x: x[1]['statistics']['test_mean'],
                           reverse=True)
    
    print("\n🏆 TOP 5 CONFIGURACIONES:")
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        stats_dict = result['statistics']
        print(f"   {i}. {name:<25} "
              f"Test: {stats_dict['test_mean']:>6.2f}%±{stats_dict['test_std']:.2f}% "
              f"| Gap: {stats_dict['gap_mean']:>5.2f}% "
              f"| Params: {result['n_params']:,}")
    
    # Hallazgos críticos
    print("\n⚠️ HALLAZGOS CRÍTICOS:")
    
    # Colapso de densidad
    collapsed = [(n, r['statistics']) for n, r in all_results.items()
                 if r['statistics']['density_mean'] < 0.05]
    if collapsed:
        print(f"   • {len(collapsed)} configuraciones colapsaron:")
        for name, stats_dict in collapsed:
            print(f"     - {name}: Densidad = {stats_dict['density_mean']:.3f}")
    
    # Componente más crítico
    most_critical = rankings[0]
    print(f"   • Componente más crítico: {most_critical[0]} "
          f"(Δ: {most_critical[1]:+.2f}%)")
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    best_config = sorted_results[0]
    print(f"   1. Usar: {best_config[0]}")
    print(f"   2. Evitar: {rankings[-1][0]} "
          f"(mejor sin él: {rankings[-1][3]:.2f}% vs {full_score:.2f}%)")
    print(f"   3. Monitorear densidad cada época")
    
    # Guardar reporte
    report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'best_configuration': sorted_results[0][0],
        'best_score': sorted_results[0][1]['statistics']['test_mean'],
        'criticality_rankings': rankings,
        'all_results': {k: v['statistics'] for k, v in all_results.items()}
    }
    
    with open(results_dir / "executive_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📁 Reporte guardado en: {results_dir}/")
    print("✅ Ablación científica completada")
    
    return all_results

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    seed_everything(42)
    results = run_scientific_ablation()