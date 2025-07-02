import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def visualize_masks_sample():
    """
    Visualiza algumas amostras das máscaras geradas
    """
    train_dir = "train"
    masks_dir = "masks"
    
    # Pegar algumas amostras aleatórias
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_mask.png')]
    samples = random.sample(mask_files, min(6, len(mask_files)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Amostras de Máscaras Geradas', fontsize=16)
    
    for i, mask_file in enumerate(samples):
        row = i // 3
        col = i % 3
        
        # Carregar máscara
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Tentar carregar imagem original correspondente
        img_name = mask_file.replace('_mask.png', '.jpg')
        img_path = os.path.join(train_dir, img_name)
        
        if os.path.exists(img_path):
            # Carregar imagem original
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Criar overlay da máscara
            mask_colored = np.zeros_like(img)
            mask_colored[:,:,0] = mask  # Canal vermelho para a máscara
            
            # Combinar imagem e máscara
            overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
            
            axes[row, col].imshow(overlay)
        else:
            # Mostrar apenas a máscara
            axes[row, col].imshow(mask, cmap='gray')
        
        axes[row, col].set_title(f'{img_name}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualização salva como 'mask_samples.png'")

def create_dataset_structure():
    """
    Cria uma estrutura organizada para treino de segmentação
    """
    print("📁 Criando estrutura do dataset...")
    
    # Criar estrutura de diretórios
    dirs_to_create = [
        'dataset/train/images',
        'dataset/train/masks',
        'dataset/val/images', 
        'dataset/val/masks',
        'dataset/test/images',
        'dataset/test/masks'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Estrutura criada:")
    for dir_path in dirs_to_create:
        print(f"   📂 {dir_path}")
    
    print("\n💡 Próximos passos:")
    print("1. Dividir as imagens e máscaras em train/val/test")
    print("2. Copiar os arquivos para a estrutura dataset/")
    print("3. Criar script de treinamento de segmentação")

def count_files():
    """
    Conta arquivos em cada diretório
    """
    train_images = len([f for f in os.listdir('train') if f.endswith('.jpg')])
    train_jsons = len([f for f in os.listdir('train') if f.endswith('.json')])
    masks = len([f for f in os.listdir('masks') if f.endswith('.png')])
    
    print("📊 Resumo do dataset:")
    print(f"   🖼️  Imagens de treino: {train_images}")
    print(f"   📝 Anotações JSON: {train_jsons}")
    print(f"   🎭 Máscaras geradas: {masks}")
    
    # Verificar classes
    cat_masks = len([f for f in os.listdir('masks') if f.startswith('cat')])
    dog_masks = len([f for f in os.listdir('masks') if f.startswith('dog')])
    
    print(f"   🐱 Máscaras de gatos: {cat_masks}")
    print(f"   🐶 Máscaras de cachorros: {dog_masks}")

if __name__ == "__main__":
    print("🔍 Verificando máscaras geradas...\n")
    
    count_files()
    print("\n" + "="*50 + "\n")
    
    try:
        visualize_masks_sample()
    except Exception as e:
        print(f"⚠️  Erro na visualização: {e}")
        print("Continuando sem visualização...")
    
    print("\n" + "="*50 + "\n")
    create_dataset_structure()
