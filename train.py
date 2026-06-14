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
    # 3. Dataset and DataLoader Setup
    # ==========================================
    train_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file="/kaggle/working/task1_uu_train.json", 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        transform=None 
    )
    val_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file="/kaggle/working/task1_uu_val.json", 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        transform=None
    )
    
    # Low batch size (e.g., 2 or 4) due to the models weight on free GPUs
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 4. OWOD Network and Optimizer Initialization
    # ==========================================
    # Num classes = 20 known (background and dummy class are handled internally)
    # Choose use_spatial_cnn=True for your CNN approach, or False for Paper's MLP
    model = OWODFasterRCNN(num_known_classes=20, use_spatial_cnn=True).to(device)
    
    # Optimizer (AdamW is standard for modern networks. We only train required parameters)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # ==========================================
    # 5. RESUME FROM CHECKPOINT AND CSV LOGGING
    # ==========================================
    num_epochs = 20
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
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Train_Loss_b_unk', 'Train_Loss_et'])

    # ==========================================
    # 6. TRAINING LOOP AND EARLY STOPPING
    # ==========================================
    for epoch in range(start_epoch, num_epochs):
        # --- WARM-UP STRATEGY (progressive module activation) ---
        if epoch < 4:
            model.use_etm = False
            model.use_urm = False
        elif epoch < 8:
            model.use_etm = True
            model.use_urm = False
        else:
            model.use_etm = True
            model.use_urm = True
            
        print(f"Epoch {epoch+1} - Configuration: ETM={model.use_etm}, URM={model.use_urm}")

        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in loop:
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
            
            total_loss += losses.item()
            loop.set_postfix(loss=losses.item())
            
        train_loss_avg = total_loss / len(train_loader)
        print(f"End of Epoch {epoch+1} - Avg Train Loss: {train_loss_avg:.4f}")
        
        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        # model is in train() mode, necessary for Faster R-CNN to return losses.
        # We use torch.no_grad() to avoid building the graph.
        val_loss_total = 0.0
        val_loop = tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}")
        
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
                val_loss_total += losses.item()
                val_loop.set_postfix(val_loss=losses.item())
                
        val_loss_avg = val_loss_total / len(val_loader)
        print(f"Epoch {epoch+1} - Avg Val Loss: {val_loss_avg:.4f}")
        
        # Write to CSV file
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Log the losses. Note: extracting exact loss_b_unk and loss_et requires 
            # accumulating them in the train loop, for simplicity we save the CSV state here.
            writer.writerow([epoch+1, f"{train_loss_avg:.4f}", f"{val_loss_avg:.4f}", "", ""])
            
        # Save current epoch state for Resume capability
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, "owod_model_last.pth")
        
        # EARLY STOPPING LOGIC
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            print(f"New best model found! (Val Loss: {val_loss_avg:.4f}). Saving...")
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