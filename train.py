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
    torch.cuda.manual_seed_all(seed) # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import our custom modules
from OWOD_dataset import OWODDataset
from OWOD_detector import OWODFasterRCNN

# ==========================================
# 1. Collate Function (CRITICAL for Detection)
# ==========================================
# PyTorch default collate tries to stack tensors into perfect matrices.
# Since every image has a DIFFERENT number of bounding boxes, it would crash.
# This function tells DataLoader to keep images and targets in separate lists.
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    set_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Starting training on: {device}")

    # ==========================================
    # 2. DINOv2 Initialization (Foundation Model)
    # ==========================================
    print("Loading DINOv2 (ViT-Small)...")
    # We use the small model (21M parameters) to comfortably fit in VRAM
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)
    dinov2.eval() # Inference only
    for param in dinov2.parameters():
        param.requires_grad = False

    # ==========================================
    # 3. Dataset Setup (Task 1)
    # ==========================================
    train_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file="/kaggle/working/task1_10cls_uu_train.json", 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
        transform=None 
    )
    val_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file="/kaggle/working/task1_10cls_uu_val.json", 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
        transform=None
    )
    
    # Batch size set to 4 to prevent Out of Memory (OOM) errors during training spikes
    # num_workers=4 e pin_memory=True sono FONDAMENTALI per velocizzare il caricamento sul Cluster
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ==========================================
    # 4. OWOD Network and Optimizer Initialization
    # ==========================================
    # Num classes = 10 known (background and dummy class are handled internally)
    # Choose use_spatial_cnn=True for your CNN approach, or False for Paper's MLP
    model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True).to(device)
    
    # Optimizer (AdamW is standard for modern networks. We only train required parameters)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # ==========================================
    # 5. RESUME FROM CHECKPOINT AND CSV LOGGING
    # ==========================================
    num_epochs = 15
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    checkpoint_path = "owod_model_last.pth"
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch + 1} with best_val_loss={best_val_loss:.4f}")
        
    # Setup CSV file for metrics
    csv_file = "training_metrics.csv"
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
        # FAST SCHEDULE (15 EPOCHS) - Anticipato come richiesto
        if epoch < 4:
            model.use_etm = False
            model.use_urm = False
        elif epoch < 6:
            model.use_etm = True
            model.use_urm = False
        else:
            model.use_etm = True
            model.use_urm = True
            
        if epoch == 4 or epoch == 6:
            print(f"--> New module activated at Epoch {epoch+1}. Resetting best loss tracking!")
            best_val_loss = float('inf')
            patience_counter = 0
            
        print(f"Epoch {epoch+1} - Configuration: ETM={model.use_etm}, URM={model.use_urm}")

        model.train()
        # Initialize dictionaries to keep track of individual losses
        train_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                           'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                           'loss_b_unk': 0.0, 'loss_et': 0.0}
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, targets) in enumerate(loop):
            # Move images and targets to GPU
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # --- DINOv2 FEATURE EXTRACTION ---
            dino_features_list = None
            if model.use_etm:
                with torch.no_grad():
                    # DINOv2 requires image to be a multiple of 14.
                    # We resize "on the fly" only for DINO
                    dino_features_list = []
                    for img in images:
                        _, h, w = img.shape
                        new_h, new_w = (h // 14) * 14, (w // 14) * 14
                        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
                        
                        # Extract spatial tokens from DINO (excluding global CLS token)
                        features_dict = dinov2.forward_features(img_resized)
                        patch_tokens = features_dict['x_norm_patchtokens'] # Shape: [1, N, C]
                        
                        # Reshape tokens into a 2D spatial grid [1, C, H_grid, W_grid]
                        C = patch_tokens.shape[-1]
                        dino_2d = patch_tokens.permute(0, 2, 1).reshape(1, C, new_h // 14, new_w // 14)
                        dino_features_list.append(dino_2d)
                    
            # --- OWOD FORWARD PASS ---
            # Pass images, targets and DINO feature list to our detector
            loss_dict = model(images, targets, dino_features_list)
            
            # Sum all losses (RPN, RoI, L_b_unk, L_et)
            losses = sum(loss for loss in loss_dict.values())
            
            # --- BACKPROPAGATION ---
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient Clipping (Prevents mathematical explosions if SAM/DINO output anomalies initially)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate individual losses
            train_loss_sums['total'] += losses.item()
            for k, v in loss_dict.items():
                if k in train_loss_sums:
                    train_loss_sums[k] += v.item()
                
            loop.set_postfix(loss=losses.item())
            
        # Calculate averages for training
        num_batches_train = len(train_loader)
        train_avg = {k: v / num_batches_train for k, v in train_loss_sums.items()}
        print(f"End of Epoch {epoch+1} - Train Loss: {train_avg['total']:.4f} (Cls: {train_avg['loss_classifier']:.4f}, Obj: {train_avg['loss_objectness']:.4f}, URM: {train_avg['loss_b_unk']:.4f}, ET: {train_avg['loss_et']:.4f})")
        
        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        # PyTorch Faster R-CNN only returns losses when model.train() is active.
        # We use torch.no_grad() to ensure weights are not updated.
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
                
        # Calculate averages for validation
        num_batches_val = len(val_loader)
        val_avg = {k: v / num_batches_val for k, v in val_loss_sums.items()}
        print(f"Epoch {epoch+1} - Val Loss: {val_avg['total']:.4f}")
        
        # Write to CSV file
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, 
                f"{train_avg['total']:.4f}", f"{train_avg['loss_classifier']:.4f}", f"{train_avg['loss_box_reg']:.4f}", 
                f"{train_avg['loss_objectness']:.4f}", f"{train_avg['loss_rpn_box_reg']:.4f}", f"{train_avg['loss_b_unk']:.4f}", f"{train_avg['loss_et']:.4f}",
                f"{val_avg['total']:.4f}", f"{val_avg['loss_classifier']:.4f}", f"{val_avg['loss_box_reg']:.4f}", 
                f"{val_avg['loss_objectness']:.4f}", f"{val_avg['loss_rpn_box_reg']:.4f}", f"{val_avg['loss_b_unk']:.4f}", f"{val_avg['loss_et']:.4f}"
            ])
            
        # Save current epoch state for Resume capability
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, "owod_model_last.pth")
        
        # Save model for this specific epoch
        torch.save(checkpoint, f"owod_model_epoch_{epoch+1}.pth")
        
        # EARLY STOPPING LOGIC
        if val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            patience_counter = 0
            print(f"New best model found! (Val Loss: {val_avg['total']:.4f}). Saving...")
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early Stopping triggered! Training interrupted.")
                break
        
        # Empty cache for the next epoch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()