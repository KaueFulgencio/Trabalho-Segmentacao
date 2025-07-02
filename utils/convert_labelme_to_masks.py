import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

def json_to_mask(json_path, output_dir):
    """
    Converte arquivo JSON do LabelMe em máscara colorida com classes diferentes
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Obter dimensões da imagem
        img_height = data['imageHeight']
        img_width = data['imageWidth']
        
        # Criar máscara vazia RGB (fundo = preto)
        mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Definir cores RGB para cada classe
        class_colors = {
            'cat': (255, 0, 0),      # Gato = Vermelho
            'dog': (0, 255, 0)       # Cachorro = Verde
        }
        
        # Processar cada shape/anotação
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                # Obter label da anotação
                label = shape['label'].lower()
                
                # Determinar cor da classe baseado no nome do arquivo
                filename = os.path.basename(json_path).lower()
                if 'cat' in filename:
                    class_color = class_colors['cat']
                    class_name = "gato"
                elif 'dog' in filename:
                    class_color = class_colors['dog']
                    class_name = "cachorro"
                else:
                    # Fallback: usar label se disponível
                    if 'cat' in label or 'gato' in label:
                        class_color = class_colors['cat']
                        class_name = "gato"
                    elif 'dog' in label or 'cachorro' in label:
                        class_color = class_colors['dog']
                        class_name = "cachorro"
                    else:
                        class_color = class_colors['cat']  # Default para gato
                        class_name = "gato"
                
                # Converter pontos para formato correto
                points = [(int(point[0]), int(point[1])) for point in shape['points']]
                
                # Criar imagem PIL temporária RGB
                temp_img = Image.new('RGB', (img_width, img_height), (0, 0, 0))
                ImageDraw.Draw(temp_img).polygon(points, fill=class_color)
                
                # Converter para numpy e adicionar à máscara
                temp_mask = np.array(temp_img)
                # Onde temp_mask não é preto, usar a cor da classe
                non_black_pixels = np.any(temp_mask > 0, axis=2)
                mask[non_black_pixels] = temp_mask[non_black_pixels]
        
        # Salvar máscara
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        cv2.imwrite(mask_path, mask)
        
        # Verificar cores presentes na máscara
        unique_pixels = np.unique(mask.reshape(-1, 3), axis=0)
        colors_found = []
        for pixel in unique_pixels:
            if np.array_equal(pixel, [0, 0, 0]):
                colors_found.append("fundo")
            elif np.array_equal(pixel, [255, 0, 0]):
                colors_found.append("gato")
            elif np.array_equal(pixel, [0, 255, 0]):
                colors_found.append("cachorro")
        
        return True, f"Máscara salva: {mask_path} (classes: {', '.join(colors_found)})"
        
    except Exception as e:
        return False, f"Erro ao processar {json_path}: {str(e)}"

def process_train_jsons():
    """
    Processa todos os arquivos JSON na pasta train/
    """
    train_dir = "train"
    masks_dir = "masks"
    
    # Criar diretório de máscaras se não existir
    os.makedirs(masks_dir, exist_ok=True)
    
    # Listar todos os arquivos JSON
    json_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
    
    print(f"Encontrados {len(json_files)} arquivos JSON para processar...")
    
    success_count = 0
    error_count = 0
    
    # Processar cada arquivo JSON com barra de progresso
    for json_file in tqdm(json_files, desc="Convertendo JSONs para máscaras"):
        json_path = os.path.join(train_dir, json_file)
        success, message = json_to_mask(json_path, masks_dir)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"\n{message}")
    
    print(f"\nProcessamento concluído!")
    print(f"✅ Sucessos: {success_count}")
    print(f"❌ Erros: {error_count}")
    print(f"📁 Máscaras salvas em: {masks_dir}/")

def verify_masks():
    """
    Verifica as máscaras coloridas geradas
    """
    masks_dir = "masks"
    if not os.path.exists(masks_dir):
        print("Diretório de máscaras não encontrado!")
        return
    
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_mask.png')]
    print("\n📊 Estatísticas das máscaras:")
    print(f"Total de máscaras: {len(mask_files)}")
    
    if mask_files:
        # Verificar uma máscara de exemplo
        sample_mask = cv2.imread(os.path.join(masks_dir, mask_files[0]), cv2.IMREAD_COLOR)
        print(f"Dimensões da máscara: {sample_mask.shape}")
        
        # Contar cores únicas
        unique_pixels = np.unique(sample_mask.reshape(-1, 3), axis=0)
        print(f"Cores únicas encontradas:")
        for pixel in unique_pixels:
            if np.array_equal(pixel, [0, 0, 0]):
                print(f"  - Fundo (preto): {pixel}")
            elif np.array_equal(pixel, [0, 0, 255]):  # OpenCV usa BGR
                print(f"  - Gato (vermelho): {pixel}")
            elif np.array_equal(pixel, [0, 255, 0]):
                print(f"  - Cachorro (verde): {pixel}")
            else:
                print(f"  - Cor desconhecida: {pixel}")

if __name__ == "__main__":
    print("🔄 Convertendo anotações LabelMe para máscaras...")
    process_train_jsons()
    verify_masks()
