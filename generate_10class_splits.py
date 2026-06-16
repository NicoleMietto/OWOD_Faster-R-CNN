import json
from tqdm import tqdm
import random

def generate_unknown_unknown_split(coco_path, output_path, known_classes, future_classes):
    known_classes = set(known_classes)
    future_classes = set(future_classes)
    
    print(f"Loading {coco_path}...")
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    img_to_anns = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    valid_images = []
    valid_annotations = []

    print("Filtering images (Unknown-Unknown Rule)...")
    for img in tqdm(coco['images']):
        img_id = img['id']
        annotations = img_to_anns[img_id]
        
        categories_in_img = set(ann['category_id'] for ann in annotations)
        
        # 1. There must be AT LEAST one known class
        has_known = not categories_in_img.isdisjoint(known_classes)
        # 2. There must be NO future class
        has_future = not categories_in_img.isdisjoint(future_classes)
        
        if has_known and not has_future:
            valid_images.append(img)
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
    print(f" - Retained images (TOTAL for Task 1): {len(valid_images)}")
    
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
    # TASK 1: ORA SOLO 10 CLASSI!
    TASK_1_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
    
    COCO_ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
                        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
                        85, 86, 87, 88, 89, 90]
                        
    future_for_task1 = [c for c in COCO_ALL_CLASSES if c not in TASK_1_CLASSES]

    TRAIN_ANNOTATIONS = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    VAL_ANNOTATIONS = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_val2017.json'

    print("--- Generating Task 1 Splits (10 Classes) ---")
    
    master_dict = generate_unknown_unknown_split(TRAIN_ANNOTATIONS, '/kaggle/working/task1_10cls_uu_master.json', TASK_1_CLASSES, future_for_task1)
    
    # Impostato a 0.5 per ottenere circa 6000 immagini (il 50% di 12420)
    split_and_save(master_dict, '/kaggle/working/task1_10cls_uu_train.json', '/kaggle/working/task1_10cls_uu_val.json', split_ratio=0.9, subsample_ratio=0.5)
    
    test_dict = generate_unknown_unknown_split(VAL_ANNOTATIONS, '/kaggle/working/task1_10cls_uu_test.json', TASK_1_CLASSES, future_for_task1)
    
    print("\n--- ALL JSON FILES GENERATED SUCCESSFULLY IN /kaggle/working/ ! ---")
