import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_output_folder():
    """
    Analisa completamente a pasta output para verificar consistência
    """
    output_dir = "output"
    
    print("🔍 Analisando pasta output...")
    print("="*60)
    
    # 1. Verificar estrutura de diretórios
    expected_dirs = [
        "JPEGImages",
        "SegmentationClass", 
        "SegmentationClassNpy",
        "SegmentationClassVisualization",
        "SegmentationObject",
        "SegmentationObjectNpy", 
        "SegmentationObjectVisualization"
    ]
    
    print("📁 Estrutura de diretórios:")
    for dir_name in expected_dirs:
        dir_path = os.path.join(output_dir, dir_name)
        if os.path.exists(dir_path):
            count = len(os.listdir(dir_path))
            print(f"   ✅ {dir_name}: {count} arquivos")
        else:
            print(f"   ❌ {dir_name}: NÃO ENCONTRADO")
    
    # 2. Verificar class_names.txt
    class_file = os.path.join(output_dir, "class_names.txt")
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            classes = f.read().strip().split('\n')
        print(f"\n📝 Classes definidas ({len(classes)}):")
        for i, class_name in enumerate(classes):
            print(f"   {i}: {class_name}")
    else:
        print("\n❌ class_names.txt não encontrado")
    
    # 3. Analisar tipos de máscaras
    print("\n🎭 Análise das máscaras:")
    
    # SegmentationClass
    seg_class_files = os.listdir(os.path.join(output_dir, "SegmentationClass"))[:5]
    print("\n   SegmentationClass (máscaras de classe):")
    for file in seg_class_files:
        mask = cv2.imread(os.path.join(output_dir, "SegmentationClass", file), cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(mask)
        print(f"     {file}: valores {unique_vals}")
    
    # SegmentationObject  
    seg_obj_files = os.listdir(os.path.join(output_dir, "SegmentationObject"))[:5]
    print("\n   SegmentationObject (máscaras de instância):")
    for file in seg_obj_files:
        mask = cv2.imread(os.path.join(output_dir, "SegmentationObject", file), cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(mask)
        print(f"     {file}: valores {unique_vals}")
    
    # NPY files
    npy_files = os.listdir(os.path.join(output_dir, "SegmentationClassNpy"))[:5]
    print("\n   SegmentationClassNpy (arrays NumPy):")
    for file in npy_files:
        data = np.load(os.path.join(output_dir, "SegmentationClassNpy", file))
        unique_vals = np.unique(data)
        print(f"     {file}: shape {data.shape}, valores {unique_vals}")
    
    # 4. Verificar correspondência entre arquivos
    print("\n🔗 Verificando correspondência entre diretórios:")
    
    jpeg_files = set(f.replace('.jpg', '') for f in os.listdir(os.path.join(output_dir, "JPEGImages")))
    seg_class_files = set(f.replace('.png', '') for f in os.listdir(os.path.join(output_dir, "SegmentationClass")))
    npy_files = set(f.replace('.npy', '') for f in os.listdir(os.path.join(output_dir, "SegmentationClassNpy")))
    
    print(f"   JPEGImages: {len(jpeg_files)} arquivos base")
    print(f"   SegmentationClass: {len(seg_class_files)} arquivos base")
    print(f"   SegmentationClassNpy: {len(npy_files)} arquivos base")
    
    # Verificar se todos os conjuntos são iguais
    if jpeg_files == seg_class_files == npy_files:
        print("   ✅ Todos os diretórios têm correspondência perfeita")
    else:
        missing_in_seg = jpeg_files - seg_class_files
        missing_in_npy = jpeg_files - npy_files
        if missing_in_seg:
            print(f"   ⚠️  Arquivos faltando em SegmentationClass: {missing_in_seg}")
        if missing_in_npy:
            print(f"   ⚠️  Arquivos faltando em SegmentationClassNpy: {missing_in_npy}")
    
    # 5. Contar classes
    print("\n📊 Distribuição por classe:")
    cat_count = len([f for f in jpeg_files if f.startswith('cat')])
    dog_count = len([f for f in jpeg_files if f.startswith('dog')])
    
    print(f"   🐱 Gatos: {cat_count}")
    print(f"   🐶 Cachorros: {dog_count}")
    print(f"   📱 Total: {cat_count + dog_count}")

def compare_with_train_masks():
    """
    Compara as máscaras na pasta output com as máscaras coloridas que criamos
    """
    print("\n" + "="*60)
    print("🔄 Comparando com máscaras coloridas criadas...")
    
    # Verificar se temos correspondência
    output_seg = set(f.replace('.png', '') for f in os.listdir("output/SegmentationClass"))
    colorful_masks = set(f.replace('_mask.png', '') for f in os.listdir("masks"))
    
    print(f"\nArquivos na pasta output: {len(output_seg)}")
    print(f"Máscaras coloridas criadas: {len(colorful_masks)}")
    
    if output_seg == colorful_masks:
        print("✅ Correspondência perfeita entre output e máscaras coloridas!")
    else:
        only_output = output_seg - colorful_masks
        only_masks = colorful_masks - output_seg
        
        if only_output:
            print(f"⚠️  Apenas em output: {len(only_output)} arquivos")
        if only_masks:
            print(f"⚠️  Apenas em masks: {len(only_masks)} arquivos")

def check_format_consistency():
    """
    Verifica consistência dos formatos
    """
    print("\n" + "="*60)
    print("🎯 Verificando consistência de formatos...")
    
    # Verificar se os valores nas máscaras fazem sentido
    sample_files = ['cat.0', 'dog.0']
    
    for sample in sample_files:
        if os.path.exists(f"output/SegmentationClass/{sample}.png"):
            # Máscara PNG
            mask_png = cv2.imread(f"output/SegmentationClass/{sample}.png", cv2.IMREAD_GRAYSCALE)
            
            # Máscara NPY
            mask_npy = np.load(f"output/SegmentationClassNpy/{sample}.npy")
            
            # Máscara colorida nossa
            if os.path.exists(f"masks/{sample}_mask.png"):
                mask_color = cv2.imread(f"masks/{sample}_mask.png")
                
                print(f"\n{sample}:")
                print(f"   PNG output: {mask_png.shape} - valores {np.unique(mask_png)}")
                print(f"   NPY output: {mask_npy.shape} - valores {np.unique(mask_npy)}")
                print(f"   Colorida: {mask_color.shape} - valores únicos {len(np.unique(mask_color.reshape(-1, 3), axis=0))}")

if __name__ == "__main__":
    analyze_output_folder()
    compare_with_train_masks()
    check_format_consistency()
    
    print("\n" + "="*60)
    print("✅ Análise completa da pasta output concluída!")
