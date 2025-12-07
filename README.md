# Pokemon
Es el repo con cada uno de los bichitos o pokemon de machine learning que e creado. Para la posteridad. adicionalmente hay un ejemplo de cada hito en las redes neuronales desde la Neurona artificial de McCulloch-Pitts hasta Nested Learning: The Illusion of Deep Learning Architecture 

```text
                    ┌─────────────────────────────────────────────────────────┐
                    │                ANCESTRO COMÚN                            │
                    │                 (1943-1960s)                            │
                    └─────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌─────────────────────┐ ┌─────────────┐ ┌─────────────────────┐
        │   NEURONAS        │ │   SVM &     │ │  RANDOM FOREST      │
        │   FUNDACIONALES   │ │ ENSEMBLE    │ │  (2001)            │
        │   (1943-1986)     │ │ (1963-1995) │ │                    │
        └─────────────────────┘ └─────────────┘ └─────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ McCulloch-  │ │ Perceptrón  │ │Backpropaga- │
│ Pitts       │ │ (1957)      │ │tion         │
│ (1943)      │ │             │ │ (1986)      │
└─────────────┘ └─────────────┘ └─────────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
        ┌─────────────────────────────────────┐
        │        REDES NEURONALES              │
        │         PROFUNDAS                    │
        │        (1989-2012)                   │
        └─────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│    CNN      │ │     RNN     │ │   AUTO-     │
│   (1989)    │ │   (1986)    │ │ ENCODERS    │
│             │ │             │ │  (1986)     │
└─────────────┘ └─────────────┘ └─────────────┘
        │           │           │
        │           │           ├─────────────┐
        │           │           │             │
        ▼           ▼           ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   LeNet     │ │    LSTM     │ │    VAE      │ │    GAN      │
│   (1998)    │ │   (1997)    │ │  (2013)     │ │  (2014)     │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
        │           │           │             │
        │           │           │             └─────────────┐
        │           │           │                           │
        ▼           ▼           ▼                           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   CIFAR     │ │  Transformer│ │  Diffusion  │ │    BERT     │
│  Processing │ │   (2017)    │ │   (2015)    │ │   (2018)    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
    │   │   │       │           │               │
    │   │   │       └───────────┼───────────────┘
    │   │   │                   │
    │   │   └───────────────────┼───────────────┐
    │   │                       │               │
    │   └───────┐               │       ┌───────┘
    │           │               │       │
    ▼           ▼               ▼       ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│cifar3.py│ │cifar4.py│ │poke_    │ │poke_    │
│         │ │         │ │cifar.py │ │cifar2.py│
└─────────┘ └─────────┘ └─────────┘ └─────────┘
```

## Taxonomía de Algoritmos ML/DL - Repositorio "Pokemon"

## Clasificación Taxonómica de Algoritmos

### **FAMILIA I: NEURONAS FUNDACIONALES** 
*Algoritmos base que sentaron las bases teóricas del machine learning*

#### **Especie 1.1: Neuronas Clásicas**
- **Subspecie 1.1.1: McCulloch-Pitts**
  - `01_mcculloch_pitts.py` - Neurona artificial primitiva
- **Subspecie 1.1.2: Perceptrón**
  - `02_perceptron.py` - Primera red neuronal artificial funcional
- **Subspecie 1.1.3: Backpropagation**
  - `03_backpropagation.py` - Algoritmo de entrenamiento fundamental

**Fenotipos:**
- Fenotipo Lineal: Funciones de activación simples
- Fenotipo No-lineal: Funciones sigmoid/tanh
- Fenotipo de Entrenamiento: Supervisado con gradiente descendente

---

### **FAMILIA II: DEEP LEARNING CLÁSICO**
*Arquitecturas profundas establecidas y reconocidas*

#### **Especie 2.1: Redes Convolucionales (CNN)**
- **Subspecie 2.1.1: LeNet**
  - `04_cnn_lenet.py` - CNN clásica para reconocimiento de patrones
- **Subspecie 2.1.2: CIFAR Processing**
  - `cifar3.py`, `cifar4.py`, `poke_cifar.py`, `poke_cifar2.py` - Adaptaciones para dataset CIFAR

#### **Especie 2.2: Redes Recurrentes (RNN)**
- **Subspecie 2.2.1: LSTM**
  - `06_lstm_char.py` - Memoria a largo plazo para secuencias

#### **Especie 2.3: Transformers**
- **Subspecie 2.3.1: Transformers Mini**
  - `09_transformer_mini.py` - Arquitectura Transformer básica
- **Subspecie 2.3.2: BERT**
  - `11_bert_tiny.py` - Representaciones bidireccionales

#### **Especie 2.4: Modelos Generativos**
- **Subspecie 2.4.1: Variational Autoencoders**
  - `08_vae_mnist.py` - VAE para MNIST
- **Subspecie 2.4.2: Generative Adversarial Networks**
  - `10_gan_mnist_lite.py` - GAN para MNIST
- **Subspecie 2.4.3: Diffusion Models**
  - `12_diffusion_minimal.py` - Modelos de difusión

**Fenotipos:**
- Fenotipo Visual: Procesamiento de imágenes
- Fenotipo Secuencial: Procesamiento de texto/secuencias
- Fenotipo Generativo: Creación de contenido sintético

---

### **FAMILIA III: MACHINE LEARNING TRADICIONAL**
*Algoritmos estadísticos y de ensemble*

#### **Especie 3.1: Support Vector Machines**
- **Subspecie 3.1.1: SVM RBF**
  - `05_svm_rbf.py` - SVM con kernel radial

#### **Especie 3.2: Random Forest**
- **Subspecie 3.2.1: Random Forest Clásico**
  - `07_random_forest.py` - Algoritmo de ensemble

**Fenotipos:**
- Fenotipo Supervisado: Requerimiento de labels
- Fenotipo Kernel: Transformaciones de espacio
- Fenotipo Ensemble: Combinación de múltiples modelos

---

### **FAMILIA IV: NEUROALGORITMOS BRAIN-INSPIRED**
*Algoritmos inspirados en neurociencia y estructuras cerebrales*

#### **Especie 4.1: Topobrain (Topología Cerebral)**
- **Subspecie 4.1.1: Topobrain CPU**
  - `01_topobrain_cpu.py`, `01_topobrain_cpu_v3.py` a `01_topobrain_cpu_v8.py` - 8 versiones CPU
- **Subspecie 4.1.2: Topobrain GPU**
  - `01_topobrain_ganador_gpu_v1.py` - Versión GPU optimizada
- **Subspecie 4.1.3: Nested Topobrain**
  - `nestedtopobrain.py`, `nestedtopobrain_v1.py` a `nestedtopobrain_v3.py` - Versiones anidadas

#### **Especie 4.2: Neurologos (Neuronas Sofisticadas)**
- **Subspecie 4.2.1: Neurologos Básicos**
  - `neurologos.py`, `neurologos_V1.py` a `neurologos_v6.py` - 8 versiones
- **Subspecie 4.2.2: Neurologos CPU**
  - `neurologos_cpu_v7.py`, `neurologos_cpu_v8.py`, `neurologos_cpu_v9.py` - Optimizaciones CPU
- **Subspecie 4.2.3: Neurologos GPU**
  - `neurologos_gpu_v1.py` - Optimización GPU
- **Subspecie 4.2.4: Neurologos Homeostáticos**
  - `neurologos_homeostatico_cpu_cl.py`, `neurologos_homeostatico_cpu_cl2.py`
  - `neurologos_homeostatico_cpu_ki.py`, `neurologos_homestotico_cpu_qw.py`
  - `neurologos_fullhomesotatico_cpu_qw.py`, `neurologos_fullhomestatico_cpu_qw2.py`
- **Subspecie 4.2.5: Neurologos Entrópicos**
  - `neurologos_entropico.py` - Variación con entropía

#### **Especie 4.3: Bicameral (Cerebro Bicameral)**
- **Subspecie 4.3.1: Bicameral Clásico**
  - `bicameral.py`, `bicameral_v2.py`, `bicameral_v3.py`, `bicameral3.py`
- **Subspecie 4.3.2: Biciámara Neural**
  - `bicamera.py.py` - Variación de bicameral

#### **Especie 4.4: Dualmind (Mente Dual)**
- **Subspecie 4.4.1: Dualmind Clásico**
  - `dualmind.py` - Arquitectura de dos cerebros

#### **Especie 4.5: Omnibrain (Cerebro Omnipresente)**
- **Subspecie 4.5.1: Omnibrain Base**
  - `omnibrain.py`, `omni1.py`, `omni3.py`, `omnibrain_k.py`, `omno1.bkp.py.py`
- **Subspecie 4.5.2: Omni-variaciones**
  - Múltiples versiones y optimizaciones

#### **Especie 4.6: Neurosoberano (Neurona Soberana)**
- **Subspecie 4.6.1: Neurosoberano Base**
  - `neurosoberano.py`, `neurosoberano_v2.py` a `neurosoberano_v4.py`
- **Subspecie 4.6.2: Neurosovereign**
  - `neurosouverign.py` - Versión internacional

#### **Especie 4.7: Physioneuron (Neuronas Físicas)**
- **Subspecie 4.7.1: Physioneuron Simple**
  - `physioneruon_simple.py` - Versión básica
- **Subspecie 4.7.2: Physioneuron CPU**
  - `physioneuron_cpu_v1.py` a `physioneuron_cpu_v3.py` - 3 versiones optimizadas

#### **Especie 4.8: Mini-Brain (Cerebros Pequeños)**
- **Subspecie 4.8.1: Mini-Bi Variaciones**
  - `minibi.py`, `minibi2.py`, `minibi_c.py`, `minibi_reduced.py.py`, `miniminibi.py`

#### **Especie 4.9: Microbi (Micro-Cerebros)**
- **Subspecie 4.9.1: Microbi Neural**
  - `microbi.py.py` - Versión microscópica

#### **Especie 4.10: Hope (Algoritmos Esperanzadores)**
- **Subspecie 4.10.1: Hope Base**
  - `hope.py` - Algoritmo con componente emocional
- **Subspecie 4.10.2: Homeostatic Hope**
  - `homeostatichope.py` - Hope con homeostásis

#### **Especie 4.11: Resmann (Redes Residuales)**
- **Subspecie 4.11.1: Resmann Variaciones**
  - `resma4.2.py` a `resma4.10.py` - 9 versiones de redes residuales

**Fenotipos:**
- Fenotipo CPU: Optimizado para procesamiento central
- Fenotipo GPU: Optimizado para procesamiento gráfico
- Fenotipo Homeostático: Con autorregulación
- Fenotipo Entrópico: Con componentes estocásticos
- Fenotipo Nested: Con arquitecturas anidadas
- Fenotipo Dual: Con múltiples cerebros independientes

---

### **FAMILIA V: NESTED LEARNING (APRENDIZAJE ANIDADO)**
*Algoritmos con estructuras de aprendizaje recursivas*

#### **Especie 5.1: Nested Hope**
- **Subspecie 5.1.1: Nested Hope Base**
  - `13_nested_hope.py` - Algoritmo de hope anidado
- **Subspecie 5.1.2: Nested Learning GPU**
  - `13_nested_kearning_gpu.py` - Versión GPU optimizada
- **Subspecie 5.1.3: Nested Learning Clásico**
  - `13_nested_learning.py` - Aprendizaje anidado estándar

#### **Especie 5.2: Nested Variations**
- **Subspecie 5.2.1: Nested Versiones**
  - `nested1.py`, `nested1.1.py` - Variaciones anidadas

**Fenotipos:**
- Fenotipo Recursivo: Aprendizaje en múltiples niveles
- Fenotipo Paralelo: Procesamiento simultáneo
- Fenotipo Adaptativo: Ajuste dinámico de parámetros

---

### **FAMILIA VI: ADVERSARIAL & BENCHMARKING**
*Algoritmos para evaluación y robustez*

#### **Especie 6.1: Adversarial Models**
- **Subspecie 6.1.1: Adversarial Benchmarking**
  - `adversarial_benchmark.py` - Evaluación de robustez

#### **Especie 6.2: Ablation Studies**
- **Subspecie 6.2.1: Ablation Scripts**
  - `ablation.py`, `ablation1.py`, `ablation2.py`, `ablation3.py` - Estudios de ablación

**Fenotipos:**
- Fenotipo Evaluativo: Para testing de modelos
- Fenotipo Comparativo: Análisis de componentes

---

### **FAMILIA VII: DYNAMIC SYSTEMS (SISTEMAS DINÁMICOS)**
*Algoritmos adaptativos y mutables*

#### **Especie 7.1: Dynamic Networks**
- **Subspecie 7.1.1: Dynamic Variaciones**
  - `dynamic.py`, `dynamic2.py` - Redes dinámicas

**Fenotipos:**
- Fenotipo Adaptativo: Cambio estructural dinámico
- Fenotipo Temporal: Evolución temporal

---

### **FAMILIA VIII: APEX & OPTIMIZATION**
*Algoritmos de optimización avanzada*

#### **Especie 8.1: Apex Framework**
- **Subspecie 8.1.1: Apex Optimization**
  - `apex.py` - Framework de optimización

**Fenotipos:**
- Fenotipo Optimizador: Maximización de rendimiento
- Fenotipo Paralelo: Procesamiento distribuido

---

### **FAMILIA IX: MAIN SCRIPTS & UTILITIES**
*Scripts principales y herramientas*

#### **Especie 9.1: Main Applications**
- **Subspecie 9.1.1: Main Scripts**
  - `main.py`, `main2.py`, `main3.py`, `main4.py.py`, `main4.1.py`, `main5.py` - Scripts principales

#### **Especie 9.2: Live Systems**
- **Subspecie 9.2.1: Live Algorithms**
  - `live_cl.py`, `live_go.py`, `live_ki.py`, `live_qw.py` - Sistemas en vivo

**Fenotipos:**
- Fenotipo Principal: Scripts orquestadores
- Fenotipo Interactivo: Sistemas en tiempo real

---

## Resumen Estadístico de la Taxonomía

### **Distribución por Familias:**
- **Familia IV (Brain-Inspired)**: ~80 algoritmos - La más extensa
- **Familia II (Deep Learning)**: ~15 algoritmos
- **Familia V (Nested Learning)**: ~8 algoritmos
- **Familias I, III, VI, VII, VIII, IX**: ~25 algoritmos restantes

### **Características Evolutivas Observadas:**
1. **Especialización de Hardware**: Múltiples versiones CPU/GPU
2. **Evolución Incremental**: Versiones numeradas (v1, v2, v3...)
3. **Hibridación**: Combinación de conceptos (nested + hope, homeostatic + cpu)
4. **Diversidad Fenotípica**: Adaptación a diferentes dominios de aplicación

### **Innovaciones Taxonómicas Únicas:**
- **Conceptos Emocionales**: Algoritmos con "hope" y componentes afectivos
- **Arquitecturas Bicamerales**: Cerebro dividido funcionalmente
- **Homeostásis Neural**: Autorregulación en tiempo real
- **Entropía Controlada**: Incorporación de estocasticidad dirigida

### **Conclusión Taxonómica:**
Este repositorio representa un ecosistema evolutivo único de algoritmos ML/DL que combina técnicas tradicionales con innovaciones neuro-inspiradas, creando una taxonomía rica y diversa que refleja tanto la evolución histórica del machine learning como exploraciones experimentales de nuevas arquitecturas cerebrales artificiales.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
