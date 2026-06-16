import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from OWOD_dataset import OWODDataset
from OWOD_detector import OWODFasterRCNN

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    set_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Starting OVERFIT TRAINING on: {device}")

    # ==========================================
    # 2. DINOv2 Initialization 
    # ==========================================
    print("Loading DINOv2 (ViT-Small)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False

    # ==========================================
    # 3. Mini-Dataset Setup (OVERFIT TEST)
    # ==========================================
    # IMPORTANTE: Usiamo il mini dataset sia per train che per validation
    mini_json_path = "/kaggle/working/mini_task1_uu_train.json"
    
    train_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file=mini_json_path, 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        transform=None 
    )
    val_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file=mini_json_path, # OVERFIT: Test sulle stesse immagini!
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        transform=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 4. OWOD Network 
    # ==========================================
    model = OWODFasterRCNN(num_known_classes=20, use_spatial_cnn=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # ==========================================
    # 5. CONFIGURAZIONE VELOCE
    # ==========================================
    num_epochs = 15 # Test breve e conciso
    start_epoch = 0
    best_val_loss = float('inf')
    
    csv_file = "mini_training_metrics.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'Epoch', 
                'Train_Total', 'Train_Cls', 'Train_BoxReg', 'Train_Obj', 'Train_RPNBoxReg', 'Train_b_unk', 'Train_et',
                'Val_Total', 'Val_Cls', 'Val_BoxReg', 'Val_Obj', 'Val_RPNBoxReg', 'Val_b_unk', 'Val_et'
            ])

    # ==========================================
    # 6. TRAINING LOOP 
    # ==========================================
    for epoch in range(start_epoch, num_epochs):
        # Warm-up ravvicinato per test veloce
        if epoch < 4:
            model.use_etm = False
            model.use_urm = False
        elif epoch < 8:
            model.use_etm = True
            model.use_urm = False
        else:
            model.use_etm = True
            model.use_urm = True
            
        if epoch == 4 or epoch == 8:
            print(f"--> New module activated at Epoch {epoch+1}. Resetting best loss!")
            best_val_loss = float('inf')
            
        print(f"Epoch {epoch+1} - Configuration: ETM={model.use_etm}, URM={model.use_urm}")

        model.train()
        train_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                           'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                           'loss_b_unk': 0.0, 'loss_et': 0.0}
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, targets) in enumerate(loop):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            dino_features_list = None
            if model.use_etm:
                with torch.no_grad():
                    dino_features_list = []
                    for img in images:
                        _, h, w = img.shape
                        new_h, new_w = (h // 14) * 14, (w // 14) * 14
                        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
                        features_dict = dinov2.forward_features(img_resized)
                        patch_tokens = features_dict['x_norm_patchtokens']
                        C = patch_tokens.shape[-1]
                        dino_2d = patch_tokens.permute(0, 2, 1).reshape(1, C, new_h // 14, new_w // 14)
                        dino_features_list.append(dino_2d)
                    
            loss_dict = model(images, targets, dino_features_list)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            train_loss_sums['total'] += losses.item()
            for k, v in loss_dict.items():
                if k in train_loss_sums:
                    train_loss_sums[k] += v.item()
                
            loop.set_postfix(loss=losses.item())
            
        num_batches_train = len(train_loader)
        train_avg = {k: v / num_batches_train for k, v in train_loss_sums.items()}
        print(f"End of Epoch {epoch+1} - Train Loss: {train_avg['total']:.4f} (Cls: {train_avg['loss_classifier']:.4f}, URM: {train_avg['loss_b_unk']:.4f}, ET: {train_avg['loss_et']:.4f})")
        
        # VALIDATION LOOP
        model.train()
        val_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                         'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                         'loss_b_unk': 0.0, 'loss_et': 0.0}
        val_loop = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for images, targets in val_loop:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                val_dino_features = None
                if model.use_etm:
                    val_dino_features = []
                    for img in images:
                        _, h, w = img.shape
                        new_h, new_w = (h // 14) * 14, (w // 14) * 14
                        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
                        features_dict = dinov2.forward_features(img_resized)
                        patch_tokens = features_dict['x_norm_patchtokens']
                        C = patch_tokens.shape[-1]
                        dino_2d = patch_tokens.permute(0, 2, 1).reshape(1, C, new_h // 14, new_w // 14)
                        val_dino_features.append(dino_2d)
                
                loss_dict = model(images, targets, val_dino_features)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss_sums['total'] += losses.item()
                for k, v in loss_dict.items():
                    if k in val_loss_sums:
                        val_loss_sums[k] += v.item()
                        
                val_loop.set_postfix(val_loss=losses.item())
                
        num_batches_val = len(val_loader)
        val_avg = {k: v / num_batches_val for k, v in val_loss_sums.items()}
        print(f"Epoch {epoch+1} - Val Loss: {val_avg['total']:.4f}")
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, 
                f"{train_avg['total']:.4f}", f"{train_avg['loss_classifier']:.4f}", f"{train_avg['loss_box_reg']:.4f}", 
                f"{train_avg['loss_objectness']:.4f}", f"{train_avg['loss_rpn_box_reg']:.4f}", f"{train_avg['loss_b_unk']:.4f}", f"{train_avg['loss_et']:.4f}",
                f"{val_avg['total']:.4f}", f"{val_avg['loss_classifier']:.4f}", f"{val_avg['loss_box_reg']:.4f}", 
                f"{val_avg['loss_objectness']:.4f}", f"{val_avg['loss_rpn_box_reg']:.4f}", f"{val_avg['loss_b_unk']:.4f}", f"{val_avg['loss_et']:.4f}"
            ])
            
        torch.save(model.state_dict(), f"mini_owod_model_epoch_{epoch+1}.pth")
        
        if val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            print(f"New best model found! (Val Loss: {val_avg['total']:.4f}). Saving...")
            torch.save(model.state_dict(), "mini_best_model.pth")
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
