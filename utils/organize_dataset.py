import os
import shutil
import random
from sklearn.model_selection import train_test_split

def organize_dataset():
    """
    Organiza o dataset dividindo em train/val/test (70/20/10)
    """
    print("📂 Organizando dataset...")
    
    # Listar todos os arquivos que têm tanto imagem quanto máscara
    train_dir = "train"
    masks_dir = "masks"
    
    # Encontrar pares imagem-máscara válidos
    valid_pairs = []
    
    for mask_file in os.listdir(masks_dir):
        if mask_file.endswith('_mask.png'):
            base_name = mask_file.replace('_mask.png', '')
            img_file = base_name + '.jpg'
            
            if os.path.exists(os.path.join(train_dir, img_file)):
                valid_pairs.append(base_name)
    
    print(f"✅ Encontrados {len(valid_pairs)} pares válidos de imagem-máscara")
    
    # Separar por classe para divisão balanceada
    cat_pairs = [p for p in valid_pairs if p.startswith('cat')]
    dog_pairs = [p for p in valid_pairs if p.startswith('dog')]
    
    print(f"   🐱 Gatos: {len(cat_pairs)}")
    print(f"   🐶 Cachorros: {len(dog_pairs)}")
    
    def split_class_data(pairs, test_size=0.3, val_size=0.66):
        """Divide uma classe em train/val/test"""
        if len(pairs) < 3:
            return pairs, [], []
        
        # Primeiro split: train vs (val+test)
        train, temp = train_test_split(pairs, test_size=test_size, random_state=42)
        
        # Segundo split: val vs test 
        if len(temp) < 2:
            return train, temp, []
        
        val, test = train_test_split(temp, test_size=val_size, random_state=42)
        return train, val, test
    
    # Dividir cada classe
    cat_train, cat_val, cat_test = split_class_data(cat_pairs)
    dog_train, dog_val, dog_test = split_class_data(dog_pairs)
    
    # Combinar classes
    splits = {
        'train': cat_train + dog_train,
        'val': cat_val + dog_val,
        'test': cat_test + dog_test
    }
    
    print(f"\n📊 Divisão do dataset:")
    for split_name, pairs in splits.items():
        cats = len([p for p in pairs if p.startswith('cat')])
        dogs = len([p for p in pairs if p.startswith('dog')])
        print(f"   {split_name}: {len(pairs)} total (🐱{cats} + 🐶{dogs})")
    
    # Copiar arquivos para estrutura dataset
    for split_name, pairs in splits.items():
        img_dest = f"dataset/{split_name}/images"
        mask_dest = f"dataset/{split_name}/masks"
        
        print(f"\n📁 Copiando arquivos para {split_name}...")
        
        for base_name in pairs:
            # Copiar imagem
            src_img = os.path.join(train_dir, f"{base_name}.jpg")
            dst_img = os.path.join(img_dest, f"{base_name}.jpg")
            shutil.copy2(src_img, dst_img)
            
            # Copiar máscara
            src_mask = os.path.join(masks_dir, f"{base_name}_mask.png")
            dst_mask = os.path.join(mask_dest, f"{base_name}_mask.png")
            shutil.copy2(src_mask, dst_mask)
    
    print("\n✅ Dataset organizado com sucesso!")
    
    # Verificar resultado final
    print("\n📋 Verificação final:")
    for split in ['train', 'val', 'test']:
        img_count = len(os.listdir(f"dataset/{split}/images"))
        mask_count = len(os.listdir(f"dataset/{split}/masks"))
        print(f"   {split}: {img_count} imagens, {mask_count} máscaras")

def create_dataset_info():
    """
    Cria arquivo com informações do dataset
    """
    info = """# Dataset de Segmentação - Cachorros e Gatos

## Estrutura
```
dataset/
├── train/
│   ├── images/     # Imagens de treinamento
│   └── masks/      # Máscaras correspondentes
├── val/
│   ├── images/     # Imagens de validação  
│   └── masks/      # Máscaras correspondentes
└── test/
    ├── images/     # Imagens de teste
    └── masks/      # Máscaras correspondentes
```

## Classes
- **0**: Fundo (preto)
- **255**: Objeto (branco) - Cachorro ou Gato

## Formato das Máscaras
- Formato: PNG em escala de cinza
- Valores: 0 (fundo) e 255 (objeto)
- Mesmo tamanho das imagens originais

## Como usar
1. Carregar imagem do dataset/*/images/
2. Carregar máscara correspondente do dataset/*/masks/
3. Normalizar valores das máscaras (0-1 ou 0-255 conforme necessário)
"""
    
    with open("dataset/README.md", "w", encoding="utf-8") as f:
        f.write(info)
    
    print("📄 Arquivo dataset/README.md criado")

if __name__ == "__main__":
    organize_dataset()
    create_dataset_info()
    print("\n🎉 Pronto! Seu dataset está organizado e pronto para treino!")
