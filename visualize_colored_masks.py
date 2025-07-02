import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def visualize_colored_masks():
    """
    Visualiza máscaras coloridas geradas
    """
    train_dir = "train"
    masks_dir = "masks"
    
    # Pegar amostras de gatos e cachorros
    cat_masks = [f for f in os.listdir(masks_dir) if f.startswith('cat') and f.endswith('_mask.png')]
    dog_masks = [f for f in os.listdir(masks_dir) if f.startswith('dog') and f.endswith('_mask.png')]
    
    # Selecionar 3 de cada
    cat_samples = random.sample(cat_masks, min(3, len(cat_masks)))
    dog_samples = random.sample(dog_masks, min(3, len(dog_masks)))
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('Máscaras Coloridas - Gatos (Vermelho) vs Cachorros (Verde)', fontsize=16)
    
    # Visualizar gatos (linha superior)
    for i, mask_file in enumerate(cat_samples):
        # Imagem original
        img_name = mask_file.replace('_mask.png', '.jpg')
        img_path = os.path.join(train_dir, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i*2].imshow(img)
            axes[0, i*2].set_title(f'Gato Original\n{img_name}')
        
        # Máscara colorida
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        axes[0, i*2+1].imshow(mask)
        axes[0, i*2+1].set_title(f'Máscara (Vermelho)')
        
        axes[0, i*2].axis('off')
        axes[0, i*2+1].axis('off')
    
    # Visualizar cachorros (linha inferior)
    for i, mask_file in enumerate(dog_samples):
        # Imagem original
        img_name = mask_file.replace('_mask.png', '.jpg')
        img_path = os.path.join(train_dir, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, i*2].imshow(img)
            axes[1, i*2].set_title(f'Cachorro Original\n{img_name}')
        
        # Máscara colorida
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        axes[1, i*2+1].imshow(mask)
        axes[1, i*2+1].set_title(f'Máscara (Verde)')
        
        axes[1, i*2].axis('off')
        axes[1, i*2+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('colored_masks_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualização das máscaras coloridas salva como 'colored_masks_samples.png'")

def analyze_mask_colors():
    """
    Analisa as cores presentes nas máscaras
    """
    masks_dir = "masks"
    
    print("🔍 Analisando cores das máscaras...")
    
    cat_count = 0
    dog_count = 0
    
    for mask_file in os.listdir(masks_dir):
        if mask_file.endswith('_mask.png'):
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path)
            
            # Verificar cores únicas (OpenCV lê como BGR)
            unique_pixels = np.unique(mask.reshape(-1, 3), axis=0)
            
            has_red = any(np.array_equal(pixel, [255, 0, 0]) for pixel in unique_pixels)  # Vermelho (B=255, G=0, R=0)
            has_green = any(np.array_equal(pixel, [0, 255, 0]) for pixel in unique_pixels)  # Verde (B=0, G=255, R=0)
            
            if has_red:
                cat_count += 1
            elif has_green:
                dog_count += 1
    
    print(f"📊 Análise das cores:")
    print(f"   🔴 Máscaras de gatos (vermelho): {cat_count}")
    print(f"   🟢 Máscaras de cachorros (verde): {dog_count}")
    print(f"   📱 Total: {cat_count + dog_count}")

if __name__ == "__main__":
    analyze_mask_colors()
    print("\n" + "="*50 + "\n")
    
    try:
        visualize_colored_masks()
    except Exception as e:
        print(f"⚠️  Erro na visualização: {e}")
        print("Verifique se matplotlib está instalado: pip install matplotlib")
