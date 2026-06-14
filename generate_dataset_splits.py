import json
from tqdm import tqdm
import random

def generate_unknown_unknown_split(coco_path, output_path, known_classes, future_classes):
    """
    Filters the COCO dataset to create an Unknown-Unknown split for a given task.
    
    Args:
        coco_path (str): Path to the original COCO JSON file (e.g. instances_train2017.json).
        output_path (str): Path to save the new filtered JSON.
        known_classes (set or list): IDs of the known classes up to the current task.
        future_classes (set or list): IDs of the future task classes (to be excluded).
    """
    known_classes = set(known_classes)
    future_classes = set(future_classes)
    
    print(f"Loading {coco_path}...")
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    # Fast dictionary to map image_id to its annotations
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
        # 2. There must be NO future class (Unknown-Unknown constraint)
        has_future = not categories_in_img.isdisjoint(future_classes)
        
        if has_known and not has_future:
            valid_images.append(img)
            # Keep only annotations for known classes (the rest is background/unknown)
            filtered_anns = [ann for ann in annotations if ann['category_id'] in known_classes]
            valid_annotations.extend(filtered_anns)

    # Create the new COCO dictionary
    # The 'licenses' entry is kept in case external libraries expect it
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": [cat for cat in coco['categories'] if cat['id'] in known_classes]
    }

    print(f"\nResults for {output_path}:")
    print(f" - Original images: {len(coco['images'])}")
    print(f" - Retained images: {len(valid_images)}")
    
    with open(output_path, 'w') as f:
        json.dump(new_coco, f)
    print("Save completed!\n")
    
    return new_coco

def split_and_save(coco_dict, output_train, output_val, split_ratio=0.9, subsample_ratio=0.3):
    images = coco_dict['images']
    random.shuffle(images) # Shuffle randomly
    
    if subsample_ratio < 1.0:
        keep_amount = int(len(images) * subsample_ratio)
        images = images[:keep_amount]
        print(f"*** SUBSAMPLING APPLIED: Reduced dataset to {keep_amount} images ({subsample_ratio*100}%) ***")
        
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Filter annotations for train
    train_img_ids = set(img['id'] for img in train_images)
    train_anns = [ann for ann in coco_dict['annotations'] if ann['image_id'] in train_img_ids]
    
    # Filter annotations for val
    val_img_ids = set(img['id'] for img in val_images)
    val_anns = [ann for ann in coco_dict['annotations'] if ann['image_id'] in val_img_ids]
    
    # Save Train JSON
    coco_train = coco_dict.copy()
    coco_train['images'] = train_images
    coco_train['annotations'] = train_anns
    with open(output_train, 'w') as f: json.dump(coco_train, f)
        
    # Save Val JSON
    coco_val = coco_dict.copy()
    coco_val['images'] = val_images
    coco_val['annotations'] = val_anns
    with open(output_val, 'w') as f: json.dump(coco_val, f)

if __name__ == "__main__":
    # Task 1: 20 known classes (VOC-like). All other 60 COCO classes are future.
    # IDs of the 20 known classes (Task 1):
    TASK_1_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    # IDs of all COCO categories (1-90 with gaps)
    COCO_ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
                        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
                        85, 86, 87, 88, 89, 90]
                        
    future_for_task1 = [c for c in COCO_ALL_CLASSES if c not in TASK_1_CLASSES]

    # Default paths for Kaggle (if using "COCO 2017 Dataset" by Awsaf49)
    # Update these variables if your folder has a different name:
    TRAIN_ANNOTATIONS = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    VAL_ANNOTATIONS = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_val2017.json'

    print("--- Generating Task 1 Splits ---")
    
    # 1. Generate master dataset from COCO original training
    print("1. Creating master split (Unknown-Unknown) from COCO Train...")
    master_dict = generate_unknown_unknown_split(TRAIN_ANNOTATIONS, '/kaggle/working/task1_uu_master.json', TASK_1_CLASSES, future_for_task1)
    
    # 2. Divide into 90% Train and 10% Val for Early Stopping
    print("2. Splitting into Train (90%) and Validation (10%) for Early Stopping...")
    split_and_save(master_dict, '/kaggle/working/task1_uu_train.json', '/kaggle/working/task1_uu_val.json', split_ratio=0.9, subsample_ratio=0.3)
    
    # 3. Generate Test Set (using COCO original val2017)
    print("3. Creating official Test set from COCO Val...")
    test_dict = generate_unknown_unknown_split(VAL_ANNOTATIONS, '/kaggle/working/task1_uu_test.json', TASK_1_CLASSES, future_for_task1)
    
    print("\n--- ALL JSON FILES GENERATED SUCCESSFULLY IN /kaggle/working/ ! ---")