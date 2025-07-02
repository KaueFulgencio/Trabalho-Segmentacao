import os
import shutil
import random
from sklearn.model_selection import train_test_split

def organize_complete_dataset():
    """
    Organiza todos os arquivos das pastas train, test1 e output em uma estrutura final
    """
    print("📂 Organizando dataset completo...")
    print("="*60)
    
    # Criar estrutura final
    final_structure = {
        'dataset_final/train/images': [],
        'dataset_final/train/masks_colored': [],
        'dataset_final/train/masks_class': [],
        'dataset_final/train/masks_npy': [],
        'dataset_final/val/images': [],
        'dataset_final/val/masks_colored': [],
        'dataset_final/val/masks_class': [],
        'dataset_final/val/masks_npy': [],
        'dataset_final/test/images': [],
        'dataset_final/test/masks_colored': [],
        'dataset_final/test/masks_class': [],
        'dataset_final/test/masks_npy': []
    }
    
    # Criar diretórios
    for dir_path in final_structure.keys():
        os.makedirs(dir_path, exist_ok=True)
    
    # 1. Identificar todos os arquivos válidos que têm imagem + anotações
    print("🔍 Identificando arquivos válidos...")
    
    valid_files = []
    
    # Verificar arquivos na pasta train/ que têm JSON
    train_dir = "train"
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if file.endswith('.json'):
                base_name = file.replace('.json', '')
                
                # Verificar se existem todos os arquivos necessários
                img_path = os.path.join(train_dir, f"{base_name}.jpg")
                mask_colored = f"masks/{base_name}_mask.png"
                mask_class = f"output/SegmentationClass/{base_name}.png"
                mask_npy = f"output/SegmentationClassNpy/{base_name}.npy"
                
                if all(os.path.exists(path) for path in [img_path, mask_colored, mask_class, mask_npy]):
                    valid_files.append({
                        'base_name': base_name,
                        'image': img_path,
                        'mask_colored': mask_colored,
                        'mask_class': mask_class,
                        'mask_npy': mask_npy,
                        'class': 'cat' if base_name.startswith('cat') else 'dog'
                    })
    
    print(f"✅ Encontrados {len(valid_files)} arquivos válidos completos")
    
    # 2. Separar por classe para divisão balanceada
    cat_files = [f for f in valid_files if f['class'] == 'cat']
    dog_files = [f for f in valid_files if f['class'] == 'dog']
    
    print(f"   🐱 Gatos: {len(cat_files)}")
    print(f"   🐶 Cachorros: {len(dog_files)}")
    
    # 3. Dividir cada classe (70% train, 20% val, 10% test)
    def split_files(files, test_size=0.3, val_size=0.67):
        if len(files) < 3:
            return files, [], []
        
        # train vs (val+test)
        train, temp = train_test_split(files, test_size=test_size, random_state=42)
        
        if len(temp) < 2:
            return train, temp, []
        
        # val vs test
        val, test = train_test_split(temp, test_size=val_size, random_state=42)
        return train, val, test
    
    cat_train, cat_val, cat_test = split_files(cat_files)
    dog_train, dog_val, dog_test = split_files(dog_files)
    
    splits = {
        'train': cat_train + dog_train,
        'val': cat_val + dog_val, 
        'test': cat_test + dog_test
    }
    
    print("\n📊 Divisão final:")
    for split_name, files in splits.items():
        cats = len([f for f in files if f['class'] == 'cat'])
        dogs = len([f for f in files if f['class'] == 'dog'])
        print(f"   {split_name}: {len(files)} total (🐱{cats} + 🐶{dogs})")
    
    # 4. Copiar arquivos para estrutura final
    print("\n📁 Copiando arquivos...")
    
    for split_name, files in splits.items():
        print(f"\n   Processando {split_name}...")
        
        for file_info in files:
            base_name = file_info['base_name']
            
            # Copiar imagem
            dest_img = f"dataset_final/{split_name}/images/{base_name}.jpg"
            shutil.copy2(file_info['image'], dest_img)
            
            # Copiar máscara colorida
            dest_colored = f"dataset_final/{split_name}/masks_colored/{base_name}_mask.png"
            shutil.copy2(file_info['mask_colored'], dest_colored)
            
            # Copiar máscara de classe
            dest_class = f"dataset_final/{split_name}/masks_class/{base_name}.png"
            shutil.copy2(file_info['mask_class'], dest_class)
            
            # Copiar arquivo npy
            dest_npy = f"dataset_final/{split_name}/masks_npy/{base_name}.npy"
            shutil.copy2(file_info['mask_npy'], dest_npy)
    
    return splits

def handle_test1_folder():
    """
    Processa arquivos da pasta test1 que não têm anotações
    """
    print("\n" + "="*60)
    print("📁 Processando pasta test1...")
    
    test1_dir = "test1"
    if not os.path.exists(test1_dir):
        print("❌ Pasta test1 não encontrada")
        return
    
    # Criar pasta para imagens sem anotação
    os.makedirs("dataset_final/unannotated", exist_ok=True)
    
    test1_files = [f for f in os.listdir(test1_dir) if f.endswith('.jpg')]
    print(f"📊 Encontrados {len(test1_files)} arquivos em test1/")
    
    # Copiar arquivos
    for file in test1_files:
        src = os.path.join(test1_dir, file)
        dst = os.path.join("dataset_final/unannotated", file)
        shutil.copy2(src, dst)
    
    print(f"✅ {len(test1_files)} arquivos copiados para dataset_final/unannotated/")

def create_dataset_info():
    """
    Cria documentação do dataset final
    """
    print("\n" + "="*60)
    print("📄 Criando documentação...")
    
    # Contar arquivos em cada split
    splits_info = {}
    for split in ['train', 'val', 'test']:
        images_dir = f"dataset_final/{split}/images"
        if os.path.exists(images_dir):
            count = len(os.listdir(images_dir))
            cat_count = len([f for f in os.listdir(images_dir) if f.startswith('cat')])
            dog_count = count - cat_count
            splits_info[split] = {'total': count, 'cats': cat_count, 'dogs': dog_count}
    
    # Contar arquivos não anotados
    unannotated_count = 0
    if os.path.exists("dataset_final/unannotated"):
        unannotated_count = len(os.listdir("dataset_final/unannotated"))
    
    info_content = f"""# Dataset de Segmentação - Cachorros e Gatos

## 📊 Estatísticas do Dataset

### Dados Anotados:
"""
    
    total_annotated = 0
    for split, info in splits_info.items():
        info_content += f"- **{split.title()}**: {info['total']} imagens (🐱 {info['cats']} gatos + 🐶 {info['dogs']} cachorros)\n"
        total_annotated += info['total']
    
    info_content += f"""
**Total Anotado**: {total_annotated} imagens

### Dados Não Anotados:
- **Unannotated**: {unannotated_count} imagens (sem máscaras)

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
"""
    
    with open("dataset_final/README.md", "w", encoding="utf-8") as f:
        f.write(info_content)
    
    # Criar arquivo class_names.txt
    with open("dataset_final/class_names.txt", "w", encoding="utf-8") as f:
        f.write("_background_\nGato\nCachorro\n")
    
    print("✅ Documentação criada:")
    print("   📄 dataset_final/README.md")
    print("   📝 dataset_final/class_names.txt")

def verify_final_dataset():
    """
    Verifica o dataset final criado
    """
    print("\n" + "="*60)
    print("🔍 Verificando dataset final...")
    
    for split in ['train', 'val', 'test']:
        base_path = f"dataset_final/{split}"
        if os.path.exists(base_path):
            img_count = len(os.listdir(f"{base_path}/images"))
            colored_count = len(os.listdir(f"{base_path}/masks_colored"))
            class_count = len(os.listdir(f"{base_path}/masks_class"))
            npy_count = len(os.listdir(f"{base_path}/masks_npy"))
            
            print(f"\n{split.upper()}:")
            print(f"   📷 Imagens: {img_count}")
            print(f"   🎨 Máscaras coloridas: {colored_count}")
            print(f"   🎭 Máscaras de classe: {class_count}")
            print(f"   💾 Arrays NPY: {npy_count}")
            
            if img_count == colored_count == class_count == npy_count:
                print(f"   ✅ {split}: Todos os arquivos correspondem")
            else:
                print(f"   ⚠️  {split}: Discrepância nos arquivos!")

if __name__ == "__main__":
    # Executar organização completa
    splits = organize_complete_dataset()
    handle_test1_folder()
    create_dataset_info()
    verify_final_dataset()
    
    print("\n" + "="*60)
    print("🎉 DATASET FINAL ORGANIZADO COM SUCESSO!")
    print("📁 Verifique a pasta: dataset_final/")
    print("📖 Leia a documentação: dataset_final/README.md")
