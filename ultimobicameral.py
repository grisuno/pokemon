# =============================================================================
# NeuroLogos Bicameral FISIOLÓGICO v3.5
# + Métricas lingüísticas (BLEU, Accuracy)
# + Sistema médico calibrado por niveles
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import Counter
import torchvision.models as models
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MÉTRICAS LINGÜÍSTICAS
# =============================================================================
class LanguageMetrics:
    """Métricas de calidad de generación"""
    
    @staticmethod
    def sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
        """BLEU simplificado a nivel de oración"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # N-gram precision
        precisions = []
        for n in range(1, 5):
            ref_ngrams = LanguageMetrics._get_ngrams(ref_tokens, n)
            hyp_ngrams = LanguageMetrics._get_ngrams(hyp_tokens, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            precisions.append(matches / total if total > 0 else 0.0)
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens))))
        
        # Geometric mean
        if all(p > 0 for p in precisions):
            score = bp * np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))
        else:
            score = 0.0
        
        return score
    
    @staticmethod
    def _get_ngrams(tokens, n):
        """Extraer n-gramas de una lista de tokens"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    @staticmethod
    def token_accuracy(reference, hypothesis):
        """Porcentaje de tokens correctos en posición"""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        min_len = min(len(ref_tokens), len(hyp_tokens))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if ref_tokens[i] == hyp_tokens[i])
        return matches / max(len(ref_tokens), len(hyp_tokens))
    
    @staticmethod
    def word_overlap(reference, hypothesis):
        """Jaccard similarity entre palabras"""
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        
        if len(ref_set | hyp_set) == 0:
            return 0.0
        
        return len(ref_set & hyp_set) / len(ref_set | hyp_set)

# =============================================================================
# SISTEMA MÉDICO CALIBRADO
# =============================================================================
class MedicalSystem:
    """Sistema de intervención médica por niveles"""
    
    def __init__(self):
        self.intervention_history = []
        self.last_intervention_epoch = -5
    
    def diagnose_severity(self, health_score, liquid_norm, gate_mean, callosal_flow):
        """Diagnosticar gravedad del problema con análisis mejorado"""
        issues = []
        severity = 0
        
        # Issue 1: Liquid explosivo (AJUSTADO para ser más estricto)
        if liquid_norm > 20.0:
            issues.append("liquid_critical")
            severity += 3
        elif liquid_norm > 10.0:
            issues.append("liquid_very_high")
            severity += 2
        elif liquid_norm > 5.0:
            issues.append("liquid_high")
            severity += 2
        elif liquid_norm > 3.0:
            issues.append("liquid_elevated")
            severity += 1
        
        # Issue 2: Gate saturado (MÁS ESTRICTO)
        if gate_mean > 0.98 or gate_mean < 0.02:
            issues.append("gate_completely_saturated")
            severity += 3
        elif gate_mean > 0.90 or gate_mean < 0.10:
            issues.append("gate_saturated")
            severity += 2
        elif gate_mean > 0.75 or gate_mean < 0.25:
            issues.append("gate_biased")
            severity += 1
        
        # Issue 3: Callosum débil
        if callosal_flow < 0.15:
            issues.append("callosum_blocked")
            severity += 2
        elif callosal_flow < 0.35:
            issues.append("callosum_weak")
            severity += 1
        
        # Issue 4: Salud crítica
        if health_score <= 1:
            severity += 3
        elif health_score == 2:
            severity += 2
        
        return issues, severity
    
    def apply_intervention(self, model, issues, severity, epoch):
        """Aplicar intervención médica calibrada con más agresividad en gate"""
        if epoch - self.last_intervention_epoch < 2:  # Reducido de 3 a 2
            return False
        
        if severity == 0:
            return False
        
        # Determinar nivel de medicina
        if severity <= 2:
            med_level = "🟡 Nivel 1 (Suave)"
        elif severity <= 5:
            med_level = "🟠 Nivel 2 (Moderado)"
        else:
            med_level = "🔴 Nivel 3 (Agresivo)"
        
        print(f"\n{'='*80}")
        print(f"🏥 INTERVENCIÓN MÉDICA - {med_level} - Severidad: {severity}/12")
        print(f"{'='*80}")
        
        interventions_applied = []
        
        with torch.no_grad():
            right_node = model.right_hemisphere.spatial_liquid
            
            # NIVEL 1: Intervención suave (severity 1-2)
            if severity >= 1:
                if "liquid_elevated" in issues:
                    print("💊 Nivel 1: Estabilizando liquid (decay 95%)")
                    right_node.W_fast *= 0.95
                    interventions_applied.append("liquid_decay_soft")
                
                if "gate_biased" in issues:
                    print("💊 Nivel 1: Rebalanceando gate (ruido pequeño)")
                    for param in model.left_hemisphere.liquid_gate.parameters():
                        param.data += torch.randn_like(param) * 0.002
                    interventions_applied.append("gate_noise_soft")
            
            # NIVEL 2: Intervención moderada (severity 3-5)
            if severity >= 3:
                if "liquid_high" in issues or "liquid_very_high" in issues:
                    print("💊 Nivel 2: Reduciendo liquid (40%)")
                    right_node.W_fast *= 0.6
                    interventions_applied.append("liquid_reduce_moderate")
                
                if "gate_saturated" in issues or "gate_completely_saturated" in issues:
                    print("💊 Nivel 2: RESET COMPLETO del gate")
                    # Resetear toda la arquitectura del gate
                    for layer in model.left_hemisphere.liquid_gate:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.5)
                            if layer.bias is not None:
                                layer.bias.data.zero_()
                    # Asegurar que el último bias sea 0 y pesos pequeños
                    model.left_hemisphere.liquid_gate[-1].bias.data.fill_(0.0)
                    model.left_hemisphere.liquid_gate[-1].weight.data.mul_(0.05)
                    interventions_applied.append("gate_full_reset")
                
                if "callosum_weak" in issues:
                    print("💊 Nivel 2: Aumentando residual scale")
                    model.corpus_callosum.residual_scale.data = torch.tensor(0.85)
                    interventions_applied.append("callosum_boost")
                
                print("💊 Nivel 2: Reduciendo fatiga (50%)")
                right_node.fatigue *= 0.5
                interventions_applied.append("fatigue_reduction")
                
                print("💊 Nivel 2: Ajustando sensibilidad")
                right_node.sensitivity = torch.tensor(0.5)
                interventions_applied.append("sensitivity_reset")
            
            # NIVEL 3: Intervención agresiva (severity 6+)
            if severity >= 6:
                if "liquid_critical" in issues or "liquid_very_high" in issues:
                    print("🚨 Nivel 3: RESETEO TOTAL de liquid")
                    right_node.W_fast = 0.00005 * torch.randn_like(right_node.W_fast)
                    right_node.norm_ema = torch.tensor(0.3)
                    interventions_applied.append("liquid_full_reset")
                
                if "gate_completely_saturated" in issues:
                    print("🚨 Nivel 3: REINICIALIZACIÓN COMPLETA del hemisferio izquierdo")
                    # Reinicializar LSTM
                    for name, param in model.left_hemisphere.lstm.named_parameters():
                        if 'weight' in name:
                            nn.init.orthogonal_(param, gain=0.8)
                        elif 'bias' in name:
                            param.data.zero_()
                    # Reinicializar gate completamente
                    for layer in model.left_hemisphere.liquid_gate:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.3)
                            if layer.bias is not None:
                                layer.bias.data.zero_()
                    model.left_hemisphere.liquid_gate[-1].weight.data.mul_(0.01)
                    interventions_applied.append("left_hemisphere_reinit")
                
                if "callosum_blocked" in issues:
                    print("🚨 Nivel 3: Reinicializando corpus callosum")
                    for layer in model.corpus_callosum.transfer:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.8)
                            if layer.bias is not None:
                                layer.bias.data.zero_()
                    model.corpus_callosum.residual_scale.data = torch.tensor(0.9)
                    interventions_applied.append("callosum_reinit")
                
                print("🚨 Nivel 3: RESTAURACIÓN COMPLETA de homeostasis")
                right_node.homeostasis = torch.tensor(1.0)
                right_node.metabolism = torch.tensor(0.7)
                right_node.fatigue = torch.tensor(0.0)
                right_node.sensitivity = torch.tensor(0.6)
                interventions_applied.append("homeostasis_restore_full")
        
        print(f"\n✓ Intervenciones aplicadas: {len(interventions_applied)}")
        for intervention in interventions_applied:
            print(f"  - {intervention}")
        print(f"{'='*80}\n")
        
        self.intervention_history.append({
            'epoch': epoch,
            'severity': severity,
            'level': med_level,
            'issues': issues,
            'interventions': interventions_applied
        })
        self.last_intervention_epoch = epoch
        
        return True

# =============================================================================
# CONSTANTES
# =============================================================================
IMG_SIZE = 224
MAX_CAPTION_LEN = 30
VOCAB_SIZE = 5000
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
NUM_WORKERS = 4

# =============================================================================
# LIQUID NEURON (igual que antes)
# =============================================================================
class StableLiquidNeuron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_slow = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.W_slow.weight, gain=0.8)
        self.register_buffer('W_fast', 0.0001 * torch.randn(out_dim, in_dim))
        self.register_buffer('metabolism', torch.tensor(0.6))
        self.register_buffer('fatigue', torch.tensor(0.0))
        self.register_buffer('sensitivity', torch.tensor(0.5))
        self.register_buffer('homeostasis', torch.tensor(1.0))
        self.ln = nn.LayerNorm(out_dim)
        self.base_lr = 0.001
        self.register_buffer('norm_ema', torch.tensor(0.5))
        self.register_buffer('norm_target', torch.tensor(1.0))
        
    def forward(self, x):
        slow_out = self.W_slow(x)
        fast_out = F.linear(x, self.W_fast)
        gate = 0.05 + 0.15 * float(self.sensitivity) * float(self.homeostasis)
        out = self.ln(slow_out + gate * fast_out)
        return out, slow_out.detach(), x.detach()
    
    def hebbian_update(self, post, pre, plasticity=0.1):
        with torch.no_grad():
            hebb = torch.mm(post.T, pre) / max(1, pre.size(0))
            hebb = torch.clamp(hebb, -0.3, 0.3)
            current_norm = self.W_fast.norm()
            self.norm_ema = 0.95 * self.norm_ema + 0.05 * current_norm
            norm_ratio = self.norm_ema / self.norm_target
            
            if norm_ratio > 3.0:
                adaptive_lr = self.base_lr * 0.01
                self.homeostasis *= 0.8
            elif norm_ratio > 1.5:
                adaptive_lr = self.base_lr * 0.3
                self.homeostasis *= 0.95
            elif norm_ratio < 0.5:
                adaptive_lr = self.base_lr * 1.2
                self.homeostasis = torch.clamp(self.homeostasis * 1.02, 0.5, 1.0)
            else:
                adaptive_lr = self.base_lr
                self.homeostasis = torch.clamp(self.homeostasis * 1.01, 0.8, 1.0)
            
            update = adaptive_lr * plasticity * float(self.homeostasis) * torch.tanh(hebb)
            self.W_fast += update
            decay = 0.999 if norm_ratio < 1.0 else 0.99 if norm_ratio < 2.0 else 0.98
            self.W_fast *= decay
            self.W_fast.clamp_(-0.5, 0.5)
            
            if current_norm > 5.0:
                self.W_fast *= (self.norm_target / current_norm)
    
    def update_physiology_advanced(self, loss_value):
        with torch.no_grad():
            loss_signal = max(0.0, min(1.0, 1.0 - loss_value / 4.0))
            homeostasis_signal = float(self.homeostasis)
            target_metab = 0.5 + 0.3 * loss_signal + 0.1 * homeostasis_signal
            self.metabolism = 0.9 * self.metabolism + 0.1 * target_metab
            self.metabolism = self.metabolism.clamp(0.3, 0.9)
            
            norm_ratio = self.norm_ema / self.norm_target
            fatigue_increment = 0.002 if norm_ratio < 2.0 else 0.01
            self.fatigue *= 0.99
            self.fatigue += fatigue_increment
            self.fatigue = self.fatigue.clamp(0, 0.5)
            
            if float(self.homeostasis) < 0.7:
                self.sensitivity *= 0.95
            else:
                target_sens = 0.5 + 0.2 * (1.0 - float(self.fatigue))
                self.sensitivity = 0.95 * self.sensitivity + 0.05 * target_sens
            self.sensitivity = self.sensitivity.clamp(0.3, 0.7)

# =============================================================================
# ARQUITECTURA (igual que antes)
# =============================================================================
class RightHemisphere(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in list(resnet.parameters())[:-20]:
            param.requires_grad = False
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.spatial_liquid = StableLiquidNeuron(2048, output_dim)
        
    def forward(self, image):
        features = self.visual_encoder(image)
        features = features.flatten(1)
        out, post, pre = self.spatial_liquid(features)
        return out, post, pre

class LeftHemisphere(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.3)
        self.liquid_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.liquid_gate[-1].bias.data.fill_(0.0)
        self.liquid_gate[-1].weight.data.mul_(0.1)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        
    def forward(self, visual_context, captions=None, max_len=30):
        batch_size = visual_context.size(0)
        device = visual_context.device
        
        if captions is not None:
            embeddings = self.embedding(captions[:, :-1])
            seq_len = embeddings.size(1)
            visual_expanded = visual_context.unsqueeze(1).expand(-1, seq_len, -1)
            lstm_input = torch.cat([embeddings, visual_expanded], dim=2)
            lstm_out, _ = self.lstm(lstm_input, self._get_init_state(visual_context))
            gate_logits = self.liquid_gate(lstm_out)
            gate = torch.sigmoid(gate_logits)
            modulated = lstm_out * (0.5 + 0.5 * gate)
            logits = self.output_projection(modulated)
            return logits, gate
        else:
            generated = []
            hidden = self._get_init_state(visual_context)
            input_token = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            
            for step in range(max_len):
                emb = self.embedding(input_token)
                visual_expanded = visual_context.unsqueeze(1)
                lstm_input = torch.cat([emb, visual_expanded], dim=2)
                out, hidden = self.lstm(lstm_input, hidden)
                gate_logits = self.liquid_gate(out)
                gate = torch.sigmoid(gate_logits)
                out = out * (0.5 + 0.5 * gate)
                logits = self.output_projection(out.squeeze(1))
                probs = F.softmax(logits / 0.9, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token)
                input_token = next_token
                if (next_token == 2).all():
                    break
            return torch.cat(generated, dim=1)
    
    def _get_init_state(self, visual_context):
        h0 = visual_context.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        return (h0, c0)

class CorpusCallosum(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.transfer = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.7))
        self.flow_attention = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(),
            nn.Linear(dim // 4, dim), nn.Sigmoid()
        )
        
    def forward(self, right_features):
        attention = self.flow_attention(right_features)
        attended = right_features * attention
        transformed = self.transfer(attended)
        return transformed + self.residual_scale * right_features

class NeuroLogosBicameralStable(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.right_hemisphere = RightHemisphere(output_dim=512)
        self.left_hemisphere = LeftHemisphere(vocab_size, embed_dim=256, hidden_dim=512)
        self.corpus_callosum = CorpusCallosum(dim=512)
        
    def forward(self, image, captions=None):
        visual_features, right_post, right_pre = self.right_hemisphere(image)
        visual_context = self.corpus_callosum(visual_features)
        
        if captions is not None:
            logits, gate = self.left_hemisphere(visual_context, captions)
            return logits, visual_features, visual_context, gate, right_post, right_pre
        else:
            output = self.left_hemisphere(visual_context, captions)
            return output

# =============================================================================
# DIAGNOSTICO MEJORADO CON MÉTRICAS LINGÜÍSTICAS
# =============================================================================
class EnhancedDiagnostics:
    def __init__(self):
        self.history = {
            'loss': [], 'right_metabolism': [], 'right_fatigue': [],
            'right_liquid_norm': [], 'right_homeostasis': [],
            'callosal_flow': [], 'left_gate_mean': [], 'left_gate_std': [],
            'synergy_score': [], 'health_score': [],
            'bleu_score': [], 'token_accuracy': [], 'word_overlap': []
        }
        self.language_metrics = LanguageMetrics()
    
    def measure_callosal_flow(self, right_features, left_context):
        with torch.no_grad():
            right_norm = F.normalize(right_features, dim=-1)
            left_norm = F.normalize(left_context, dim=-1)
            correlation = (right_norm * left_norm).sum(dim=-1).mean()
            flow_std = left_context.std(dim=-1).mean()
            flow = correlation.item() * min(1.0, flow_std.item() / 0.5)
            return flow
    
    def calculate_synergy(self, right_node, callosal_flow, left_gate_mean, left_gate_std):
        right_health = float(right_node.metabolism) * float(right_node.homeostasis) * (1.0 - float(right_node.fatigue) * 0.5)
        callosal_health = callosal_flow
        gate_balance = 1.0 - abs(left_gate_mean - 0.5) * 2.0
        gate_diversity = min(1.0, left_gate_std * 5.0)
        left_health = 0.7 * gate_balance + 0.3 * gate_diversity
        synergy = (0.35 * right_health + 0.30 * callosal_health + 0.35 * left_health)
        return synergy
    
    def calculate_health(self, right_node, callosal_flow, left_gate_mean, left_gate_std, liquid_norm):
        health = 0
        if liquid_norm < 2.0: health += 1
        if float(right_node.homeostasis) > 0.7: health += 1
        if callosal_flow > 0.4: health += 1
        if 0.4 < left_gate_mean < 0.6 and left_gate_std > 0.05: health += 1
        if float(right_node.fatigue) < 0.3 and float(right_node.metabolism) > 0.55: health += 1
        return health
    
    def update(self, **metrics):
        for key, value in metrics.items():
            if key in self.history and value is not None:
                self.history[key].append(value)
    
    def get_recent_avg(self, key, n=50):
        if key in self.history and len(self.history[key]) > 0:
            return np.mean(self.history[key][-n:])
        return 0.0
    
    def report(self, epoch):
        if len(self.history['loss']) == 0:
            return
        
        print(f"\n{'='*80}")
        print(f"📊 REPORTE COMPLETO - Época {epoch}")
        print(f"{'='*80}")
        
        # Métricas de loss
        loss = self.get_recent_avg('loss')
        print(f"\n📉 ENTRENAMIENTO:")
        print(f"  Loss: {loss:.4f}")
        
        # Métricas lingüísticas
        bleu = self.get_recent_avg('bleu_score')
        acc = self.get_recent_avg('token_accuracy')
        overlap = self.get_recent_avg('word_overlap')
        
        print(f"\n📝 CALIDAD LINGÜÍSTICA:")
        print(f"  BLEU-4:     {bleu:.4f}", end=" ")
        print("🟢" if bleu > 0.15 else "🟡" if bleu > 0.08 else "🔴")
        
        print(f"  Accuracy:   {acc:.4f}", end=" ")
        print("🟢" if acc > 0.30 else "🟡" if acc > 0.15 else "🔴")
        
        print(f"  W-Overlap:  {overlap:.4f}", end=" ")
        print("🟢" if overlap > 0.35 else "🟡" if overlap > 0.20 else "🔴")
        
        # Fisiología
        print(f"\n🧬 FISIOLOGÍA:")
        metab = self.get_recent_avg('right_metabolism')
        fatigue = self.get_recent_avg('right_fatigue')
        liquid = self.get_recent_avg('right_liquid_norm')
        homeo = self.get_recent_avg('right_homeostasis')
        
        print(f"  Liquid Norm:  {liquid:.3f}", end=" ")
        status = "🟢" if liquid < 2.0 else "🟡" if liquid < 4.0 else "🔴"
        print(status)
        
        print(f"  Homeostasis:  {homeo:.3f}", end=" ")
        print("🟢" if homeo > 0.8 else "🟡" if homeo > 0.6 else "🔴")
        
        print(f"  Metabolismo:  {metab:.3f}")
        print(f"  Fatiga:       {fatigue:.3f}")
        
        # Comunicación
        flow = self.get_recent_avg('callosal_flow')
        gate_mean = self.get_recent_avg('left_gate_mean')
        gate_std = self.get_recent_avg('left_gate_std')
        
        print(f"\n🔗 COMUNICACIÓN:")
        print(f"  Callosum:   {flow:.3f}", end=" ")
        print("🟢" if flow > 0.5 else "🟡" if flow > 0.3 else "🔴")
        
        print(f"  Gate Mean:  {gate_mean:.3f}", end=" ")
        print("🟢" if 0.4 < gate_mean < 0.6 else "🟡")
        
        print(f"  Gate Std:   {gate_std:.3f}")
        
        # Salud global
        synergy = self.get_recent_avg('synergy_score')
        health = self.get_recent_avg('health_score')
        
        print(f"\n🏛️  SISTEMA BICAMERAL:")
        print(f"  Sinergia: {synergy:.3f} | Salud: {int(health)}/5", end=" ")
        if health >= 4:
            print("🟢 ÓPTIMO")
        elif health >= 3:
            print("🟡 FUNCIONAL")
        else:
            print("🔴 CRÍTICO")
        
        print(f"{'='*80}\n")

# =============================================================================
# DATASET (mismo que antes)
# =============================================================================
class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_len=30):
        self.images_dir = images_dir
        self.transform = transform
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, caption = parts
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        self.data.append((img_path, caption))
        print(f"Loaded {len(self.data)} image-caption pairs")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        tokens = ['<BOS>'] + caption.lower().split() + ['<EOS>']
        token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        
        return image, torch.tensor(token_ids, dtype=torch.long), caption

def build_vocab_flickr(captions_file, vocab_size=5000):
    print("Building vocabulary...")
    counter = Counter()
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                words = parts[1].lower().split()
                counter.update(words)
    
    most_common = counter.most_common(vocab_size - 4)
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 4
    id2word = {i: w for w, i in vocab.items()}
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, id2word

def setup_flickr8k(data_dir='./data'):
    flickr_dir = os.path.join(data_dir, 'flickr8k')
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✓ Flickr8k ya existe\n")
        return flickr_dir
    print("Dataset no encontrado")
    return flickr_dir

# =============================================================================
# TRAINING CON MÉTRICAS Y MEDICINA
# =============================================================================
def train_with_metrics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"NeuroLogos v3.5 | Métricas + Medicina Calibrada | Device: {device}")
    print(f"{'='*80}\n")
    
    flickr_dir = setup_flickr8k()
    images_dir = os.path.join(flickr_dir, 'Images')
    captions_file = os.path.join(flickr_dir, 'captions.txt')
    
    vocab, id2word = build_vocab_flickr(captions_file, VOCAB_SIZE)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Flickr8kDataset(images_dir, captions_file, vocab, transform, MAX_CAPTION_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    model = NeuroLogosBicameralStable(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    diagnostics = EnhancedDiagnostics()
    medical_system = MedicalSystem()
    
    print(f"🧠 Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Métricas: BLEU-4 + Token Acc + Word Overlap")
    print(f"💊 Sistema Médico: 3 niveles de intervención\n")
    
    os.makedirs('./checkpoints', exist_ok=True)
    
    for epoch in range(30):
        # Plasticidad progresiva
        if epoch < 3:
            plasticity = 0.05
        elif epoch < 10:
            plasticity = 0.10
        elif epoch < 20:
            plasticity = max(0.02, 0.10 * (1 - (epoch-10)/20))
        else:
            plasticity = 0.01
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Diagnóstico pre-época
        right_node = model.right_hemisphere.spatial_liquid
        liquid = diagnostics.get_recent_avg('right_liquid_norm')
        flow = diagnostics.get_recent_avg('callosal_flow')
        gate_mean = diagnostics.get_recent_avg('left_gate_mean')
        gate_std = diagnostics.get_recent_avg('left_gate_std')
        health_score = diagnostics.calculate_health(right_node, flow, gate_mean, gate_std, liquid)
        
        # INTERVENCIÓN MÉDICA si es necesario
        issues, severity = medical_system.diagnose_severity(health_score, liquid, gate_mean, flow)
        
        # Mostrar en progress bar
        medicine_level = "🟢 Nivel 0" if severity == 0 else f"🟡 Nivel 1" if severity <= 2 else f"🟠 Nivel 2" if severity <= 5 else "🔴 Nivel 3"
        
        if severity > 0:
            medical_system.apply_intervention(model, issues, severity, epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Health: {health_score}/5 | Med: {medicine_level}]")
        
        for batch_idx, (images, captions, raw_captions) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits, visual_features, visual_context, gate, right_post, right_pre = model(images, captions)
            
            loss = F.cross_entropy(
                logits.reshape(-1, len(vocab)),
                captions[:, 1:].reshape(-1),
                ignore_index=vocab['<PAD>']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Hebbiano
            model.right_hemisphere.spatial_liquid.hebbian_update(right_post, right_pre, plasticity)
            model.right_hemisphere.spatial_liquid.update_physiology_advanced(loss.item())
            
            # Diagnóstico cada 20 batches
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    liquid_norm = model.right_hemisphere.spatial_liquid.W_fast.norm().item()
                    callosal_flow = diagnostics.measure_callosal_flow(visual_features, visual_context)
                    gate_mean_val = gate.mean().item()
                    gate_std_val = gate.std().item()
                    
                    synergy = diagnostics.calculate_synergy(right_node, callosal_flow, gate_mean_val, gate_std_val)
                    
                    diagnostics.update(
                        loss=loss.item(),
                        right_metabolism=float(right_node.metabolism),
                        right_fatigue=float(right_node.fatigue),
                        right_liquid_norm=liquid_norm,
                        right_homeostasis=float(right_node.homeostasis),
                        callosal_flow=callosal_flow,
                        left_gate_mean=gate_mean_val,
                        left_gate_std=gate_std_val,
                        synergy_score=synergy,
                        health_score=health_score
                    )
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'liquid': f'{liquid_norm:.2f}',
                'gate': f'{gate_mean_val:.2f}'
            })
        
        scheduler.step()
        
        # Evaluación lingüística
        if epoch % 2 == 0:
            model.eval()
            print("\n📸 EVALUACIÓN LINGÜÍSTICA...\n")
            
            bleu_scores = []
            acc_scores = []
            overlap_scores = []
            
            with torch.no_grad():
                for sample_idx in range(min(10, len(dataset))):
                    sample_img, sample_cap, raw_caption = dataset[sample_idx * (len(dataset) // 10)]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    
                    generated = model(sample_img, captions=None)
                    
                    gen_words = [id2word.get(int(t.item()), '<UNK>') for t in generated[0]]
                    gen_sentence = " ".join(w for w in gen_words if w not in ['<BOS>', '<EOS>', '<PAD>'])
                    
                    # Métricas
                    bleu = diagnostics.language_metrics.sentence_bleu(raw_caption, gen_sentence)
                    acc = diagnostics.language_metrics.token_accuracy(raw_caption, gen_sentence)
                    overlap = diagnostics.language_metrics.word_overlap(raw_caption, gen_sentence)
                    
                    bleu_scores.append(bleu)
                    acc_scores.append(acc)
                    overlap_scores.append(overlap)
                    
                    if sample_idx < 3:
                        print(f"Muestra {sample_idx + 1}:")
                        print(f"  GT:   {raw_caption}")
                        print(f"  Gen:  {gen_sentence}")
                        print(f"  BLEU: {bleu:.3f} | Acc: {acc:.3f} | Overlap: {overlap:.3f}\n")
            
            # Actualizar métricas
            diagnostics.update(
                bleu_score=np.mean(bleu_scores),
                token_accuracy=np.mean(acc_scores),
                word_overlap=np.mean(overlap_scores)
            )
            
            model.train()
        
        # Reporte de época
        diagnostics.report(epoch)
        
        avg_loss = total_loss / num_batches
        print(f"Época {epoch:02d} | Loss: {avg_loss:.4f} | Health: {health_score}/5")
        print(f"BLEU: {diagnostics.get_recent_avg('bleu_score'):.3f} | "
              f"Acc: {diagnostics.get_recent_avg('token_accuracy'):.3f}\n")
        
        # Checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'id2word': id2word,
                'diagnostics': diagnostics.history,
                'interventions': medical_system.intervention_history
            }, f'./checkpoints/stable_epoch_{epoch:02d}.pth')
    
    print("✅ Entrenamiento completado!")
    diagnostics.report(29)
    
    print(f"\n{'='*80}")
    print("📋 HISTORIAL DE INTERVENCIONES MÉDICAS")
    print(f"{'='*80}")
    for intervention in medical_system.intervention_history:
        print(f"Época {intervention['epoch']:02d} | Severidad: {intervention['severity']}/10")
        print(f"  Issues: {', '.join(intervention['issues'])}")
        print(f"  Aplicadas: {len(intervention['interventions'])} intervenciones\n")

if __name__ == "__main__":
    train_with_metrics()