# Dataset de Segmentação - Cachorros e Gatos

## 📊 Estatísticas do Dataset

### Dados Anotados:
- **Train**: 172 imagens (🐱 84 gatos + 🐶 88 cachorros)
- **Val**: 24 imagens (🐱 12 gatos + 🐶 12 cachorros)
- **Test**: 51 imagens (🐱 25 gatos + 🐶 26 cachorros)

**Total Anotado**: 247 imagens

### Dados Não Anotados:
- **Unannotated**: 12500 imagens (sem máscaras)

## 📁 Estrutura do Dataset

```
dataset_final/
├── train/
│   ├── images/           # Imagens de treinamento
│   ├── masks_colored/    # Máscaras coloridas (🔴 gato, 🟢 cachorro)
│   ├── masks_class/      # Máscaras de classe (formato padrão)
│   └── masks_npy/        # Arrays NumPy otimizados
├── val/
│   ├── images/           # Imagens de validação
│   ├── masks_colored/    # Máscaras coloridas
│   ├── masks_class/      # Máscaras de classe
│   └── masks_npy/        # Arrays NumPy
├── test/
│   ├── images/           # Imagens de teste
│   ├── masks_colored/    # Máscaras coloridas
│   ├── masks_class/      # Máscaras de classe
│   └── masks_npy/        # Arrays NumPy
└── unannotated/          # Imagens sem anotações
```

## 🎯 Classes

0. **_background_** - Fundo
1. **Gato** - Classe dos gatos
2. **Cachorro** - Classe dos cachorros

## 🎨 Formatos de Máscaras

### 1. Máscaras Coloridas (`masks_colored/`)
- **Formato**: PNG colorido
- **Cores**: 
  - 🖤 Fundo: (0, 0, 0)
  - 🔴 Gato: (255, 0, 0)  
  - 🟢 Cachorro: (0, 255, 0)
- **Uso**: Visualização e depuração

### 2. Máscaras de Classe (`masks_class/`)
- **Formato**: PNG em escala de cinza
- **Valores**: Índices de classe específicos do formato original
- **Uso**: Compatibilidade com frameworks específicos

### 3. Arrays NumPy (`masks_npy/`)
- **Formato**: .npy (NumPy array)
- **Valores**: 0 (fundo), 1 (gato), 2 (cachorro)
- **Uso**: Treinamento de modelos (formato mais eficiente)

## 🚀 Como Usar

### Para Treinamento:
```python
import numpy as np
import cv2

# Carregar imagem
image = cv2.imread('dataset_final/train/images/cat.0.jpg')

# Carregar máscara (recomendado: NPY)
mask = np.load('dataset_final/train/masks_npy/cat.0.npy')

# Ou carregar máscara colorida para visualização
mask_colored = cv2.imread('dataset_final/train/masks_colored/cat.0_mask.png')
```

### Para Visualização:
Use as máscaras coloridas em `masks_colored/` para visualizar facilmente as anotações.
