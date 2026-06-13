import json
from tqdm import tqdm
import random

def generate_unknown_unknown_split(coco_path, output_path, known_classes, future_classes):
    """
    Filtra il dataset COCO per creare uno split Unknown-Unknown per un dato task.
    
    Args:
        coco_path (str): Percorso al file JSON COCO originale (es. instances_train2017.json).
        output_path (str): Percorso dove salvare il nuovo JSON filtrato.
        known_classes (set o list): ID delle classi note fino al task corrente.
        future_classes (set o list): ID delle classi dei task successivi (da escludere).
    """
    known_classes = set(known_classes)
    future_classes = set(future_classes)
    
    print(f"Caricamento di {coco_path}...")
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    # Dizionario veloce per mappare image_id alle sue annotazioni
    img_to_anns = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    valid_images = []
    valid_annotations = []

    print("Filtraggio immagini (Regola Unknown-Unknown)...")
    for img in tqdm(coco['images']):
        img_id = img['id']
        annotations = img_to_anns[img_id]
        
        categories_in_img = set(ann['category_id'] for ann in annotations)
        
        # 1. Deve esserci ALMENO una classe nota
        has_known = not categories_in_img.isdisjoint(known_classes)
        # 2. NON deve esserci NESSUNA classe futura (Unknown-Unknown constraint)
        has_future = not categories_in_img.isdisjoint(future_classes)
        
        if has_known and not has_future:
            valid_images.append(img)
            # Conserviamo solo le annotazioni delle classi note (il resto è background/unknown)
            filtered_anns = [ann for ann in annotations if ann['category_id'] in known_classes]
            valid_annotations.extend(filtered_anns)

    # Creazione del nuovo dizionario COCO
    #la voce licences serve se voglio utilizzare alcune librerie esterne che si aspettano anche quella voce
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": [cat for cat in coco['categories'] if cat['id'] in known_classes]
    }

    print(f"\nRisultati per {output_path}:")
    print(f" - Immagini originali: {len(coco['images'])}")
    print(f" - Immagini trattenute: {len(valid_images)}")
    
    with open(output_path, 'w') as f:
        json.dump(new_coco, f)
    print("Salvataggio completato!\n")
    
    return new_coco

def split_and_save(coco_dict, output_train, output_val, split_ratio=0.9):
    images = coco_dict['images']
    random.shuffle(images) # Mescoliamo casualmente
    
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Filtriamo le annotazioni per il train
    train_img_ids = set(img['id'] for img in train_images)
    train_anns = [ann for ann in coco_dict['annotations'] if ann['image_id'] in train_img_ids]
    
    # Filtriamo le annotazioni per il val
    val_img_ids = set(img['id'] for img in val_images)
    val_anns = [ann for ann in coco_dict['annotations'] if ann['image_id'] in val_img_ids]
    
    # Salva JSON Train
    coco_train = coco_dict.copy()
    coco_train['images'] = train_images
    coco_train['annotations'] = train_anns
    with open(output_train, 'w') as f: json.dump(coco_train, f)
        
    # Salva JSON Val
    coco_val = coco_dict.copy()
    coco_val['images'] = val_images
    coco_val['annotations'] = val_anns
    with open(output_val, 'w') as f: json.dump(coco_val, f)

if __name__ == "__main__":
    # Task 1: 20 classi note (simil VOC). Tutte le altre 60 classi COCO sono future.
    # IDs delle 20 classi note (Task 1):
    TASK_1_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    # ID di tutte le categorie COCO (1-90 con salti)
    COCO_ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
                        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
                        85, 86, 87, 88, 89, 90]
                        
    future_for_task1 = [c for c in COCO_ALL_CLASSES if c not in TASK_1_CLASSES]

    # Percorsi di default per Kaggle (se usi "COCO 2017 Dataset" di Awsaf49)
    # Se la tua cartella si chiama diversamente, aggiorna queste variabili:
    TRAIN_ANNOTATIONS = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    VAL_ANNOTATIONS = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json'

    print("--- Generazione Split Task 1 ---")
    
    # 1. Genera il master dataset dal training originale di COCO
    print("1. Creazione del master split (Unknown-Unknown) dal Train COCO...")
    master_dict = generate_unknown_unknown_split(TRAIN_ANNOTATIONS, 'task1_uu_master.json', TASK_1_CLASSES, future_for_task1)
    
    # 2. Dividilo in 90% Train e 10% Val per l'Early Stopping
    print("2. Suddivisione in Train (90%) e Validation (10%) per Early Stopping...")
    split_and_save(master_dict, 'task1_uu_train.json', 'task1_uu_val.json', split_ratio=0.9)
    
    # 3. Genera il Test Set (usando il val2017 di COCO originale)
    print("3. Creazione del Test set ufficiale dal Val COCO...")
    test_dict = generate_unknown_unknown_split(VAL_ANNOTATIONS, 'task1_uu_test.json', TASK_1_CLASSES, future_for_task1)
    
    print("\n--- TUTTI I JSON GENERATI CON SUCCESSO! ---")