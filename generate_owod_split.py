import json
from tqdm import tqdm
import random

def generate_standard_owod_split(coco_path, output_path, known_classes):
    """
    Genera il dataset seguendo lo Standard OWOD Split.
    Prende TUTTE le immagini che contengono almeno una classe nota,
    anche se sullo sfondo ci sono cani, gatti o altre classi future.
    Le annotazioni delle classi future vengono rimosse, così la rete le
    vedrà ma le tratterà come "sfondo sconosciuto".
    """
    known_classes = set(known_classes)
    
    print(f"Loading {coco_path}...")
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    img_to_anns = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    valid_images = []
    valid_annotations = []

    print("Filtering images (Standard OWOD Rule: keeping images with future classes)...")
    for img in tqdm(coco['images']):
        img_id = img['id']
        annotations = img_to_anns[img_id]
        
        categories_in_img = set(ann['category_id'] for ann in annotations)
        
        # 1. Deve esserci ALMENO una classe nota
        has_known = not categories_in_img.isdisjoint(known_classes)
        
        # 2. Ignoriamo completamente se ci sono classi future (non blocchiamo nulla!)
        
        if has_known:
            valid_images.append(img)
            # Salviamo SOLO i bounding box delle classi note. I cani/gatti resteranno senza etichetta!
            filtered_anns = [ann for ann in annotations if ann['category_id'] in known_classes]
            valid_annotations.extend(filtered_anns)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": [cat for cat in coco['categories'] if cat['id'] in known_classes]
    }

    print(f"\nResults for {output_path}:")
    print(f" - Original images: {len(coco['images'])}")
    print(f" - Retained images (OWOD Task 1): {len(valid_images)}")
    
    with open(output_path, 'w') as f:
        json.dump(new_coco, f)
    print("Save completed!\n")
    
    return new_coco

def split_and_save(coco_dict, output_train, output_val, split_ratio=0.9, subsample_ratio=1.0):
    images = coco_dict['images']
    random.seed(42)
    random.shuffle(images)
    
    if subsample_ratio < 1.0:
        keep_amount = int(len(images) * subsample_ratio)
        images = images[:keep_amount]
        print(f"*** SUBSAMPLING APPLIED: Reduced dataset to {keep_amount} images ({subsample_ratio*100}%) ***")
    else:
        print(f"*** NO SUBSAMPLING: Using all {len(images)} valid images ***")
        
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    train_img_ids = set(img['id'] for img in train_images)
    train_anns = [ann for ann in coco_dict['annotations'] if ann['image_id'] in train_img_ids]
    
    val_img_ids = set(img['id'] for img in val_images)
    val_anns = [ann for ann in coco_dict['annotations'] if ann['image_id'] in val_img_ids]
    
    coco_train = coco_dict.copy()
    coco_train['images'] = train_images
    coco_train['annotations'] = train_anns
    with open(output_train, 'w') as f: json.dump(coco_train, f)
        
    coco_val = coco_dict.copy()
    coco_val['images'] = val_images
    coco_val['annotations'] = val_anns
    with open(output_val, 'w') as f: json.dump(coco_val, f)

if __name__ == "__main__":
    # TASK 1: Le tue famose 10 classi!
    TASK_1_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15]

    TRAIN_ANNOTATIONS = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    
    OUTPUT_FILE_TRAIN = 'task1_owod_train.json'
    OUTPUT_FILE_VAL = 'task1_owod_val.json'
    
    print("=== Generazione Dataset Standard OWOD (Fase 1) ===")
    
    # 1. Creiamo un unico grande dizionario filtrato per il Task 1
    # Nota che qui NON passiamo più le "future_classes" da scartare!
    task1_coco = generate_standard_owod_split(
        coco_path=TRAIN_ANNOTATIONS,
        output_path='temp_owod_all.json',
        known_classes=TASK_1_CLASSES
    )
    
    # 2. Suddividiamo in Train e Validation (nessun subsampling per massimizzare l'apprendimento!)
    print(f"\n=== Splitting in Train e Validation ===")
    split_and_save(
        coco_dict=task1_coco, 
        output_train=OUTPUT_FILE_TRAIN, 
        output_val=OUTPUT_FILE_VAL, 
        split_ratio=0.9, 
        subsample_ratio=1.0 # 1.0 significa che teniamo TUTTE le immagini, per dare alla rete tanta varietà!
    )
    
    print(f"Finito! Usa i file: {OUTPUT_FILE_TRAIN} e {OUTPUT_FILE_VAL} per il tuo addestramento.")
