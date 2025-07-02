# Dataset de Segmentação - Cachorros e Gatos

## 📊 Sobre o Projeto

Este projeto contém um dataset organizado para segmentação semântica de cachorros e gatos, processado a partir de anotações do LabelMe.

## 📁 Estrutura do Projeto

```
├── dataset_final/          # Dataset final organizado
│   ├── train/              # Dados de treinamento (172 imagens)
│   ├── val/                # Dados de validação (24 imagens)
│   ├── test/               # Dados de teste (51 imagens)
│   ├── unannotated/        # Imagens sem anotação (12,500 imagens)
│   ├── README.md           # Documentação detalhada do dataset
│   └── class_names.txt     # Lista de classes
├── utils/                  # Scripts utilitários
├── requirements.txt        # Dependências Python
└── README.md              # Este arquivo
```

## 🎯 Classes

- **0**: `_background_` - Fundo da imagem
- **1**: `Gato` - Segmentação de gatos
- **2**: `Cachorro` - Segmentação de cachorros

## 📊 Estatísticas do Dataset

### Dados Anotados (247 imagens total):
- **Train**: 172 imagens (84 gatos + 88 cachorros)
- **Validation**: 24 imagens (12 gatos + 12 cachorros)  
- **Test**: 51 imagens (25 gatos + 26 cachorros)


## 🎨 Formatos de Máscaras

Cada split contém 4 tipos de máscaras:

1. **`images/`** - Imagens originais (JPG)
2. **`masks_colored/`** - Máscaras coloridas para visualização
   - 🖤 Fundo: (0, 0, 0)
   - 🔴 Gato: (255, 0, 0)  
   - 🟢 Cachorro: (0, 255, 0)
3. **`masks_class/`** - Máscaras em formato padrão (PNG)
4. **`masks_npy/`** - Arrays NumPy otimizados para treino
   - 0: Fundo, 1: Gato, 2: Cachorro

## 🚀 Como Usar

### Carregar dados para treinamento:

```python
import numpy as np
import cv2

# Carregar imagem
image = cv2.imread('dataset_final/train/images/cat.0.jpg')

# Carregar máscara (formato NumPy recomendado)
mask = np.load('dataset_final/train/masks_npy/cat.0.npy')

# Para visualização, usar máscara colorida
mask_colored = cv2.imread('dataset_final/train/masks_colored/cat.0_mask.png')
```

### Estrutura para DataLoader (PyTorch):

```python
from torch.utils.data import Dataset
import torch

class SegmentationDataset(Dataset):
    def __init__(self, split='train'):
        self.images_dir = f'dataset_final/{split}/images'
        self.masks_dir = f'dataset_final/{split}/masks_npy'
        # ... implementar __getitem__ e __len__
```

## 🛠️ Dependências

```bash
pip install -r requirements.txt
```

## 📝 Scripts Utilitários

Os scripts de processamento estão em `utils/`:

- `convert_labelme_to_masks.py` - Converter JSONs do LabelMe
- `organize_final_dataset.py` - Organizar estrutura final
- `visualize_colored_masks.py` - Visualizar máscaras
- `analyze_output.py` - Analisar qualidade dos dados

## 🎯 Próximos Passos

1. **Treinar modelo de segmentação** (U-Net, DeepLab, etc.)
2. **Anotar mais imagens** da pasta `unannotated/`
3. **Validar qualidade** das anotações existentes
4. **Augmentação de dados** para melhorar performance

## 📄 Licença

[Adicionar informações de licença conforme necessário]
