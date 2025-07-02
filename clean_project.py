import os
import shutil
import glob

def clean_project():
    """
    Remove arquivos e pastas desnecessários do projeto
    """
    print("🧹 Limpando projeto...")
    print("="*50)
    
    # Pastas para remover (se existirem)
    folders_to_remove = [
        'train',
        'test1', 
        'output',
        'masks',
        'temp',
        'tmp',
        '__pycache__',
        '.ipynb_checkpoints',
        'logs',
        'runs',
        'tb_logs'
    ]
    
    # Arquivos temporários para remover
    files_to_remove = [
        '*.tmp',
        'temp_*',
        'preview_*', 
        '*.log',
        '*.out',
        '*_samples.png',
        '*_visualization.png',
        'colored_masks_samples.png',
        'mask_samples.png',
        'debug_*.py',
        'test_*.py',
        'temp_*.py',
        'playground.py'
    ]
    
    removed_count = 0
    
    # Remover pastas
    print("📁 Removendo pastas desnecessárias...")
    for folder in folders_to_remove:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   ✅ Removido: {folder}/")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Erro ao remover {folder}: {e}")
        else:
            print(f"   ⏭️  {folder}/ (não existe)")
    
    # Remover arquivos temporários
    print("\n📄 Removendo arquivos temporários...")
    for pattern in files_to_remove:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"   ✅ Removido: {file}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Erro ao remover {file}: {e}")
    
    # Verificar arquivos Python desnecessários
    print("\n🐍 Verificando scripts Python...")
    
    # Scripts que podem ser movidos para pasta utils ou removidos
    utility_scripts = [
        'analyze_output.py',
        'convert_labelme_to_masks.py', 
        'organize_dataset.py',
        'organize_final_dataset.py',
        'verify_masks.py',
        'visualize_colored_masks.py'
    ]
    
    # Criar pasta utils se não existir
    if any(os.path.exists(script) for script in utility_scripts):
        os.makedirs('utils', exist_ok=True)
        print("   📂 Criada pasta utils/")
        
        for script in utility_scripts:
            if os.path.exists(script):
                try:
                    shutil.move(script, f'utils/{script}')
                    print(f"   📦 Movido: {script} → utils/")
                    removed_count += 1
                except Exception as e:
                    print(f"   ❌ Erro ao mover {script}: {e}")
    
    return removed_count

def show_current_structure():
    """
    Mostra a estrutura atual do projeto
    """
    print("\n" + "="*50)
    print("📋 Estrutura atual do projeto:")
    print("="*50)
    
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return
            
        items = []
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return
            
        # Separar pastas e arquivos
        folders = [item for item in items if os.path.isdir(os.path.join(path, item)) and not item.startswith('.')]
        files = [item for item in items if os.path.isfile(os.path.join(path, item)) and not item.startswith('.')]
        
        # Mostrar pastas primeiro
        for i, folder in enumerate(folders):
            is_last_folder = (i == len(folders) - 1) and len(files) == 0
            folder_prefix = "└── " if is_last_folder else "├── "
            print(f"{prefix}{folder_prefix}{folder}/")
            
            extension = "    " if is_last_folder else "│   "
            show_tree(os.path.join(path, folder), prefix + extension, max_depth, current_depth + 1)
        
        # Mostrar arquivos
        for i, file in enumerate(files):
            is_last = i == len(files) - 1
            file_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{file_prefix}{file}")
    
    show_tree(".")

def create_project_readme():
    """
    Atualiza o README do projeto
    """
    readme_content = """# Dataset de Segmentação - Cachorros e Gatos

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

### Dados Não Anotados:
- **12,500 imagens** prontas para anotação futura

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
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ README.md atualizado")

if __name__ == "__main__":
    removed_count = clean_project()
    show_current_structure()
    create_project_readme()
    
    print(f"\n🎉 Limpeza concluída!")
    print(f"📊 {removed_count} itens processados")
    print(f"📁 Projeto organizado e pronto para desenvolvimento!")
    print(f"📖 Leia o README.md para mais informações")
