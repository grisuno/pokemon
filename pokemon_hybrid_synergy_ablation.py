"""
🌟 POKEMON HYBRID SYNERGY ABLATION STUDY 🌟
==========================================

Estudio de ablación de 4 niveles enfocado en maximizar sinergias entre:
- VAE: Compactación y generación generativa
- Transformer: Multi-head attention y secuencialidad  
- GAN: Adversarial training y generación discriminativa
- TopoBrain: Topología adaptativa y memoria continua

Autores: MiniMax Agent
Fecha: 2025-12-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import time
import json
import random
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN DEL ESTUDIO DE SINERGIAS
# =============================================================================

@dataclass
class SynergyConfig:
    device: str = "cpu"
    seed: int = 42
    n_samples: int = 800
    n_features: int = 20
    n_classes: int = 3
    
    # Hiperparámetros basados en modelos ganadores
    batch_size: int = 32
    
    # VAE Components (Ganador: +8.9%)
    latent_dim: int = 2  # Compactación extrema
    vae_hidden_dim: int = 64
    
    # Transformer Components (Ganador: +6.5%)
    transformer_d_model: int = 32  # Dimensión compacta
    transformer_heads: int = 4
    transformer_ff_dim: int = 64
    seq_length: int = 8
    
    # GAN Components (Ganador: +11.4%)
    gan_latent_dim: int = 50  # Reducido de 100 para eficiencia
    gan_hidden: int = 128
    
    # TopoBrain Components (Evolution: v2→v8 con 98% accuracy)
    grid_size: int = 2  # 4 nodos
    embed_dim: int = 16
    topology_sparsity: float = 0.8
    
    # Training
    epochs: int = 15
    lr: float = 0.001
    
    # Sinergy levels
    vae_attention_fusion: float = 0.5  # Balance VAE-Attention
    adversarial_weight: float = 0.7    # GAN loss weight
    topology_adapt_weight: float = 0.8 # TopoBrain plasticity
    
    def to_dict(self):
        return asdict(self)

# =============================================================================
# COMPONENTES HÍBRIDOS OPTIMIZADOS
# =============================================================================

class SynergyVAELayer(nn.Module):
    """VAE híbrido con capacidades de compactación extremas"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Latent space
        self.mu_head = nn.Linear(hidden_dim//2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, return_encoding=False):
        h = self.encoder(x)
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        
        if return_encoding:
            return self.decoder(z), mu, logvar, z
        return self.decoder(z), mu, logvar

class SynergyAttentionLayer(nn.Module):
    """Multi-head attention optimizado para datos tabulares"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        return x

class SynergyGANLayer(nn.Module):
    """GAN híbrido optimizado para features tabulares"""
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def generate(self, z):
        return self.generator(z)
    
    def discriminate(self, x):
        return self.discriminator(x)

class AdaptiveTopologyLayer(nn.Module):
    """Topología adaptativa inspirada en TopoBrain evolution"""
    def __init__(self, grid_size, embed_dim, sparsity=0.8):
        super().__init__()
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.embed_dim = embed_dim
        
        # Matriz de adyacencia adaptativa
        self.adj_matrix = nn.Parameter(
            torch.randn(self.num_nodes, self.num_nodes) * 0.1
        )
        self.target_sparsity = sparsity
        
        # Procesadores de nodos
        self.node_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(self.num_nodes)
        ])
        
        # Readout global
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim * self.num_nodes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2)
        )
        
    def get_adjacency_matrix(self):
        # Aplicar máscara de conectividad y umbral de sparsidad
        adj = torch.sigmoid(self.adj_matrix)
        
        # Asegurar conectividad mínima
        mask = torch.ones_like(adj)
        np.fill_diagonal(mask.detach().numpy(), 0)
        
        # Aplicar sparsidad target
        threshold = torch.sort(adj.view(-1))[0][int(self.target_sparsity * adj.numel())]
        adj = adj * (adj > threshold)
        
        return adj * mask
    
    def forward(self, x):
        batch_size = x.size(0)
        adj = self.get_adjacency_matrix()
        
        # Procesar cada nodo
        node_outputs = []
        for i, processor in enumerate(self.node_processors):
            node_input = x[:, i*self.embed_dim:(i+1)*self.embed_dim]
            node_output = processor(node_input)
            node_outputs.append(node_output)
        
        # Propagación topológica
        node_tensor = torch.stack(node_outputs, dim=1)  # [batch, num_nodes, embed_dim]
        
        # Aplicar adyacencia como mezcla
        mixed_nodes = []
        for i in range(self.num_nodes):
            neighbors = adj[i].unsqueeze(0)  # [1, num_nodes]
            weighted_sum = torch.sum(node_tensor * neighbors.unsqueeze(-1), dim=1)
            mixed_nodes.append(weighted_sum)
        
        # Combinar con procesamiento original
        final_nodes = []
        for i, (mixed, original) in enumerate(zip(mixed_nodes, node_outputs)):
            final_node = 0.5 * mixed + 0.5 * original
            final_nodes.append(final_node)
        
        final_tensor = torch.stack(final_nodes, dim=1)
        global_repr = self.global_pool(final_tensor.view(batch_size, -1))
        
        return global_repr, adj

# =============================================================================
# MODELO HÍBRIDO PRINCIPAL
# =============================================================================

class PokemonSynergyModel(nn.Module):
    """Modelo híbrido que combina los mejores elementos de VAE, Transformer, GAN y TopoBrain"""
    
    def __init__(self, config: SynergyConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.n_features, config.embed_dim * config.grid_size * config.grid_size)
        
        # VAE Layer (Compactación generativa)
        self.vae = SynergyVAELayer(
            config.embed_dim * config.grid_size * config.grid_size,
            config.vae_hidden_dim,
            config.latent_dim
        )
        
        # Attention Layer (Procesamiento secuencial)
        self.attention = SynergyAttentionLayer(
            config.transformer_d_model,
            config.transformer_heads,
            config.transformer_ff_dim
        )
        
        # GAN Layer (Generación adversaria)
        self.gan = SynergyGANLayer(
            config.embed_dim,
            config.gan_latent_dim,
            config.gan_hidden
        )
        
        # Topology Layer (Topología adaptativa)
        self.topology = AdaptiveTopologyLayer(
            config.grid_size,
            config.embed_dim,
            config.topology_sparsity
        )
        
        # Fusion Layer (Combinación de características)
        fusion_dim = (
            config.latent_dim +           # VAE encoding
            config.transformer_d_model +  # Attention output
            config.gan_latent_dim +       # GAN generation
            config.embed_dim // 2         # Topology global
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.n_classes)
        )
        
        # Métricas para monitoreo
        self.register_buffer('vae_loss_history', torch.zeros(100))
        self.register_buffer('gan_loss_history', torch.zeros(100))
        self.register_buffer('synergy_score', torch.zeros(1))
        
    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        
        # 1. VAE Path (Compactación y generación)
        vae_input = self.input_projection(x)
        vae_recon, mu, logvar = self.vae(vae_input, return_encoding=True)
        vae_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # 2. Attention Path (Procesamiento secuencial)
        # Reshape para atención: [batch, seq_len, d_model]
        attention_input = vae_input.view(batch_size, -1, self.config.embed_dim)
        attention_input = F.pad(attention_input, 
                               (0, self.config.transformer_d_model - self.config.embed_dim, 
                                0, self.config.seq_length - attention_input.size(1), 
                                0, 0))
        attention_output = self.attention(attention_input)
        attention_flat = attention_output.view(batch_size, -1)
        
        # 3. GAN Path (Generación adversaria)
        # Generar desde ruido
        noise = torch.randn(batch_size, self.config.gan_latent_dim)
        fake_features = self.gan.generate(noise)
        
        # Discriminar features reales
        real_prob = self.gan.discriminate(vae_input)
        
        # 4. Topology Path (Topología adaptativa)
        topo_output, adj_matrix = self.topology(attention_flat)
        
        # 5. Fusion Path (Combinación sinérgica)
        # Combinar todas las características
        combined_features = torch.cat([
            mu,                    # VAE encoding [batch, latent_dim]
            attention_flat,        # Attention [batch, transformer_d_model]
            noise,                # GAN noise [batch, gan_latent_dim]  
            topo_output           # Topology global [batch, embed_dim//2]
        ], dim=-1)
        
        logits = self.feature_fusion(combined_features)
        
        if return_all:
            return {
                'logits': logits,
                'vae_loss': vae_loss,
                'reconstruction': vae_recon,
                'attention_output': attention_output,
                'gan_fake': fake_features,
                'gan_real_prob': real_prob,
                'topology_output': topo_output,
                'adjacency': adj_matrix,
                'combined_features': combined_features
            }
        
        return logits

# =============================================================================
# ESTUDIO DE ABLACIÓN DE 4 NIVELES
# =============================================================================

class SynergyAblationStudy:
    """Estudio de ablación sistemático de sinergias"""
    
    def __init__(self, config: SynergyConfig):
        self.config = config
        self.results = {}
        
    def get_ablation_matrix(self):
        """Matriz de ablación de 4 niveles:
        
        Nivel 1 - Baseline: Solo VAE básico
        Nivel 2 - Hybrid: VAE + Attention 
        Nivel 3 - Advanced: VAE + Attention + GAN
        Nivel 4 - Full Synergy: VAE + Attention + GAN + Topology
        """
        return {
            "Nivel_1_Baseline": {
                "description": "Solo VAE - Compactación básica",
                "components": ["vae"],
                "expected_improvement": "0% (baseline)"
            },
            "Nivel_2_Hybrid": {
                "description": "VAE + Attention - Compactación con secuencia",
                "components": ["vae", "attention"],
                "expected_improvement": "8-12%"
            },
            "Nivel_3_Advanced": {
                "description": "VAE + Attention + GAN - Generación adversaria",
                "components": ["vae", "attention", "gan"],
                "expected_improvement": "15-20%"
            },
            "Nivel_4_FullSynergy": {
                "description": "VAE + Attention + GAN + Topology - Sinergia completa",
                "components": ["vae", "attention", "gan", "topology"],
                "expected_improvement": "25-35%"
            }
        }
    
    def create_variant_model(self, level_name):
        """Crear modelo variante para un nivel específico"""
        if level_name == "Nivel_1_Baseline":
            # Solo VAE básico
            class BaselineModel(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.vae = SynergyVAELayer(config.n_features, 64, 2)
                    self.classifier = nn.Linear(2, config.n_classes)
                    
                def forward(self, x):
                    _, mu, logvar, _ = self.vae(x, return_encoding=True)
                    logits = self.classifier(mu)
                    return logits
            
            return BaselineModel(self.config)
            
        elif level_name == "Nivel_2_Hybrid":
            # VAE + Attention
            class HybridModel(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.vae = SynergyVAELayer(config.n_features, 64, 8)
                    self.attention = SynergyAttentionLayer(8, 2, 16)
                    self.classifier = nn.Linear(8, config.n_classes)
                    
                def forward(self, x):
                    _, mu, logvar, _ = self.vae(x, return_encoding=True)
                    # Expandir para atención
                    attention_input = mu.unsqueeze(1)  # [batch, 1, 8]
                    attn_out = self.attention(attention_input)
                    attn_flat = attn_out.squeeze(1)
                    logits = self.classifier(attn_flat)
                    return logits
            
            return HybridModel(self.config)
            
        elif level_name == "Nivel_3_Advanced":
            # VAE + Attention + GAN
            class AdvancedModel(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.vae = SynergyVAELayer(config.n_features, 64, 12)
                    self.attention = SynergyAttentionLayer(12, 3, 24)
                    self.gan = SynergyGANLayer(12, 20, 32)
                    self.classifier = nn.Linear(32, config.n_classes)  # 12 (VAE) + 20 (GAN)
                    
                def forward(self, x):
                    _, mu, logvar, _ = self.vae(x, return_encoding=True)
                    attention_input = mu.unsqueeze(1)
                    attn_out = self.attention(attention_input)
                    attn_flat = attn_out.squeeze(1)
                    
                    # Generar ruido con dimensión correcta para el generador
                    batch_size = x.size(0)
                    noise = torch.randn(batch_size, 20)  # Dimensión correcta para SynergyGANLayer
                    gan_gen = self.gan.generate(noise)
                    
                    combined = torch.cat([mu, gan_gen], dim=-1)
                    logits = self.classifier(combined)
                    return logits
            
            return AdvancedModel(self.config)
            
        else:  # Nivel_4_FullSynergy
            return PokemonSynergyModel(self.config)

# =============================================================================
# EJECUCIÓN DEL ESTUDIO
# =============================================================================

def run_synergy_ablation():
    """Ejecutar estudio de ablación completo"""
    print("🌟 POKEMON HYBRID SYNERGY ABLATION STUDY 🌟")
    print("=" * 80)
    
    # Configurar
    config = SynergyConfig()
    study = SynergyAblationStudy(config)
    ablation_matrix = study.get_ablation_matrix()
    
    # Crear dataset
    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        n_informative=15,
        n_redundant=3,
        random_state=config.seed
    )
    
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    results = {}
    
    for level_name, level_config in ablation_matrix.items():
        print(f"\n🔥 Nivel {level_name}")
        print(f"Descripción: {level_config['description']}")
        print(f"Mejora esperada: {level_config['expected_improvement']}")
        print("-" * 60)
        
        # Crear modelo variante
        model = study.create_variant_model(level_name)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Entrenar
        model.train()
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                if level_name == "Nivel_1_Baseline":
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                elif level_name == "Nivel_4_FullSynergy":
                    # Modelo completo con todas las pérdidas
                    outputs = model(x_batch, return_all=True)
                    logits = outputs['logits']
                    
                    # Pérdidas múltiples
                    vae_loss = outputs['vae_loss']
                    
                    # Pérdida GAN (simulada)
                    real_prob = outputs['gan_real_prob']
                    fake_prob = 1.0 - real_prob  # Simplificado
                    gan_loss = -torch.log(real_prob + 1e-8) - torch.log(fake_prob + 1e-8)
                    gan_loss = gan_loss.mean()
                    
                    # Pérdida principal
                    main_loss = criterion(logits, y_batch)
                    
                    # Combinar pérdidas
                    total_loss = main_loss + 0.1 * vae_loss + 0.05 * gan_loss
                    loss = total_loss
                else:
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Métricas
                epoch_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(y_batch).sum().item()
                total += y_batch.size(0)
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
            
            if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
                print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        # Evaluar en test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                logits = model(x_batch) if level_name != "Nivel_4_FullSynergy" else model(x_batch)['logits']
                pred = logits.argmax(dim=1)
                test_correct += pred.eq(y_batch).sum().item()
                test_total += y_batch.size(0)
        
        final_test_accuracy = 100.0 * test_correct / test_total
        
        # Guardar resultados
        results[level_name] = {
            'config': level_config,
            'final_test_accuracy': final_test_accuracy,
            'training_curve': {
                'losses': epoch_losses,
                'accuracies': epoch_accuracies
            },
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
        print(f"  ✅ Test Accuracy: {final_test_accuracy:.2f}%")
        print(f"  📊 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    return results

def analyze_synergy_results(results):
    """Analizar resultados del estudio de sinergias"""
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS DE RESULTADOS DE SINERGIAS")
    print("=" * 80)
    
    # Ordenar por performance
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['final_test_accuracy'], 
                          reverse=True)
    
    print("\n🏆 RANKING DE RENDIMIENTO:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Modelo':<25} {'Accuracy':<12} {'Params':<12}")
    print("-" * 60)
    
    baseline_acc = results["Nivel_1_Baseline"]["final_test_accuracy"]
    
    for i, (name, result) in enumerate(sorted_results, 1):
        acc = result['final_test_accuracy']
        params = result['model_params']
        improvement = acc - baseline_acc
        
        print(f"{i:<6} {name:<25} {acc:>9.2f}%  {params:>10,}  (+{improvement:>5.2f}%)")
    
    print("\n🔬 ANÁLISIS DE SINERGIAS:")
    print("-" * 60)
    
    # Análisis incremental
    levels = ["Nivel_1_Baseline", "Nivel_2_Hybrid", "Nivel_3_Advanced", "Nivel_4_FullSynergy"]
    
    print("Mejora incremental por nivel:")
    prev_acc = baseline_acc
    for level in levels:
        current_acc = results[level]["final_test_accuracy"]
        improvement = current_acc - prev_acc
        cumulative_improvement = current_acc - baseline_acc
        
        print(f"  {level:<25}: +{improvement:>6.2f}% (Cum: +{cumulative_improvement:>5.2f}%)")
        prev_acc = current_acc
    
    # Identificar el mejor modelo
    best_model = sorted_results[0]
    print(f"\n🥇 GANADOR: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['final_test_accuracy']:.2f}%")
    print(f"   Parámetros: {best_model[1]['model_params']:,}")
    
    # Guardar resultados
    output_dir = Path("synergy_ablation_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "synergy_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Crear visualización
    create_synergy_visualizations(results, output_dir)
    
    return results

def create_synergy_visualizations(results, output_dir):
    """Crear visualizaciones del estudio de sinergias"""
    
    # Configurar matplotlib
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Gráfico de barras con accuracies
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Sinergias - Pokemon Hybrid Models', fontsize=16, fontweight='bold')
    
    # Accuracy por nivel
    levels = list(results.keys())
    accuracies = [results[level]["final_test_accuracy"] for level in levels]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(levels)), accuracies, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([l.replace('_', '\n') for l in levels], rotation=45, ha='right')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Performance por Nivel de Sinergia')
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Curvas de entrenamiento
    ax2 = axes[0, 1]
    for level in levels:
        if 'training_curve' in results[level]:
            epochs = range(1, len(results[level]['training_curve']['accuracies']) + 1)
            ax2.plot(epochs, results[level]['training_curve']['accuracies'], 
                    marker='o', linewidth=2, label=level.replace('_', ' '))
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.set_title('Curvas de Entrenamiento')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Eficiencia (Accuracy vs Parámetros)
    ax3 = axes[1, 0]
    params = [results[level]["model_params"] for level in levels]
    efficiency = [acc/param*100000 for acc, param in zip(accuracies, params)]  # Acc per 100k params
    
    ax3.scatter(params, accuracies, s=100, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
    for i, level in enumerate(levels):
        ax3.annotate(level.replace('_', ' '), (params[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Parámetros del Modelo')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Eficiencia: Accuracy vs Complejidad')
    ax3.grid(True, alpha=0.3)
    
    # 4. Mejora incremental
    ax4 = axes[1, 1]
    baseline_acc = results["Nivel_1_Baseline"]["final_test_accuracy"]
    improvements = [acc - baseline_acc for acc in accuracies]
    
    ax4.bar(range(len(levels)), improvements, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_xticks(range(len(levels)))
    ax4.set_xticklabels([l.replace('_', '\n') for l in levels], rotation=45, ha='right')
    ax4.set_ylabel('Mejora vs Baseline (%)')
    ax4.set_title('Mejora Incremental por Sinergia')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'synergy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizaciones guardadas en {output_dir}/synergy_analysis.png")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Ejecutar estudio
    results = run_synergy_ablation()
    
    # Analizar resultados
    final_results = analyze_synergy_results(results)
    
    print(f"\n✅ ESTUDIO DE SINERGIAS COMPLETADO")
    print(f"📁 Resultados guardados en 'synergy_ablation_results/'")
    print(f"🏆 Mejor modelo identificado con sinergias optimizadas")