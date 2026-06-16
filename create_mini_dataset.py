import json
import random
import os

def create_mini_coco(input_json, output_json, num_images=100, seed=42):
    random.seed(seed)
    
    print(f"Leggendo {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # 1. Selezioniamo casualmente le immagini
    all_images = data['images']
    num_images = min(num_images, len(all_images)) # Nel caso in cui il dataset originale sia molto piccolo
    sampled_images = random.sample(all_images, num_images)
    
    # Creiamo un set con gli ID delle immagini selezionate per ricerca rapida
    sampled_image_ids = set([img['id'] for img in sampled_images])
    
    # 2. Manteniamo solo le annotazioni (sia known che unknown) relative a quelle immagini
    print("Filtrando le annotazioni...")
    sampled_annotations = [ann for ann in data['annotations'] if ann['image_id'] in sampled_image_ids]
    
    # 3. Creiamo la nuova struttura del file COCO
    mini_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': data['categories']
    }
    
    print(f"Salvataggio in corso su {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(mini_data, f)
    
    print(f"✅ Fatto! Immagini: {len(sampled_images)} | Annotazioni: {len(sampled_annotations)}\n")

if __name__ == "__main__":
    train_in = "/kaggle/working/task1_uu_train.json"
    train_out = "/kaggle/working/mini_task1_uu_train.json"
    
    if os.path.exists(train_in):
        create_mini_coco(train_in, train_out, num_images=100)
    else:
        print(f"❌ File non trovato: {train_in}. Assicurati di essere nella directory giusta.")
