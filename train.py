import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import random
import numpy as np
import warnings
import argparse

# Suppress warnings to prevent Jupyter IOPub RAM leaks from spammy PyTorch warnings!
warnings.filterwarnings("ignore")

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

class OWODDataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        images = inputs[0]
        targets = inputs[1]
        dino_features_list = inputs[2] if len(inputs) > 2 else None
        
        batch_size = len(images)
        chunk_size = (batch_size + len(device_ids) - 1) // len(device_ids)
        
        scattered_inputs = []
        for i in range(min(len(device_ids), (batch_size + chunk_size - 1) // chunk_size)):
            dev = torch.device(f'cuda:{device_ids[i]}')
            start = i * chunk_size
            end = min((i + 1) * chunk_size, batch_size)
            
            chunk_images = [img.to(dev) for img in images[start:end]]
            chunk_targets = [{k: v.to(dev) for k, v in t.items()} for t in targets[start:end]]
            
            if dino_features_list is not None:
                chunk_dino = [d.to(dev) for d in dino_features_list[start:end]]
            else:
                chunk_dino = None
            
            scattered_inputs.append((chunk_images, chunk_targets, chunk_dino))
            
        return tuple(scattered_inputs), tuple({} for _ in scattered_inputs)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for OWOD")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], required=True, help="1=Base, 2=ETM, 3=ETM+URM")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train for this run")
    parser.add_argument('--start_epoch', type=int, default=0, help="Starting epoch number (for logging and scheduler)")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--alpha', type=float, default=0.1, help="ETM weight multiplier")
    parser.add_argument('--beta', type=float, default=0.1, help="URM weight multiplier")
    parser.add_argument('--batch_size', type=int, default=6, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--use_spatial_cnn', action='store_true', help="Use CNN instead of MLP for ETM")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Phase {args.phase} training on: {device}")

    print("Loading DINOv2 (ViT-Small)...")
    import logging
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    import sys
    sys.modules['xformers'] = None
    
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False, skip_validation=True)
    dinov2 = dinov2.to(device)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False
    
    # Configure modules based on phase
    use_etm = (args.phase >= 2)
    use_urm = (args.phase == 3)
    save_best = (args.phase == 1)

    base_model = OWODFasterRCNN(
        num_known_classes=10, 
        use_spatial_cnn=args.use_spatial_cnn, 
        alpha=args.alpha,
        beta=args.beta
    )
    base_model.use_etm = use_etm
    base_model.use_urm = use_urm
    base_model.dinov2 = dinov2

    train_dataset = OWODDataset(
        img_dir='/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017', 
        annotation_file='task1_owod_train.json', 
        known_classes=[1, 3, 5, 17, 27, 44, 52, 62, 72, 84],
        transform=None 
    )
    val_dataset = OWODDataset(
        img_dir='/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017', 
        annotation_file='task1_owod_val.json', 
        known_classes=[1, 3, 5, 17, 27, 44, 52, 62, 72, 84],
        transform=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = base_model
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = OWODDataParallel(model)
        
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # Scheduler: Drop LR at epoch 9 (fine-tuning URM stabilization)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.1)

    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # Solo se vogliamo riprendere lo stato dell'ottimizzatore
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'best_val_loss' in checkpoint and save_best:
                best_val_loss = checkpoint['best_val_loss']
        else:
            base_model.load_state_dict(checkpoint, strict=False)
            
        # Fast-forward scheduler to start_epoch
        for _ in range(args.start_epoch):
            scheduler.step()

    csv_file = "/kaggle/working/training_metrics.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'Epoch', 
                'Train_Total', 'Train_Cls', 'Train_BoxReg', 'Train_Obj', 'Train_RPNBoxReg', 'Train_b_unk', 'Train_et',
                'Val_Total', 'Val_Cls', 'Val_BoxReg', 'Val_Obj', 'Val_RPNBoxReg', 'Val_b_unk', 'Val_et'
            ])

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print(f"Epoch {epoch+1} - Configuration: Phase={args.phase}, ETM={base_model.use_etm} (alpha={args.alpha}), URM={base_model.use_urm} (beta={args.beta})")

        model.train()
        train_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                           'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                           'loss_b_unk': 0.0, 'loss_et': 0.0}
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=30.0)
        
        for i, (images, targets) in enumerate(loop):
            loss_dict = model(images, targets, None)
            losses = sum(loss.mean() for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            train_loss_sums['total'] += losses.item()
            for k, v in loss_dict.items():
                if k in train_loss_sums:
                    train_loss_sums[k] += v.mean().item()
                
            loop.set_postfix(loss=losses.item())
            del images, targets, loss_dict, losses
            
        num_batches_train = len(train_loader)
        train_avg = {k: v / num_batches_train for k, v in train_loss_sums.items()}
        print(f"End of Epoch {epoch+1} - Train Loss: {train_avg['total']:.4f} "
              f"(Cls: {train_avg['loss_classifier']:.4f}, Obj: {train_avg['loss_objectness']:.4f}, "
              f"Box: {train_avg['loss_box_reg']:.4f}, RPNBox: {train_avg['loss_rpn_box_reg']:.4f}, "
              f"URM: {train_avg['loss_b_unk']:.4f}, ET: {train_avg['loss_et']:.4f})")
              
        scheduler.step()
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")
        
        model.train()
        val_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                         'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                         'loss_b_unk': 0.0, 'loss_et': 0.0}
        val_loop = tqdm(val_loader, desc="Validation", mininterval=30.0)
        
        with torch.no_grad():
            for images, targets in val_loop:
                loss_dict = model(images, targets, None)
                losses = sum(loss.mean() for loss in loss_dict.values())
                
                val_loss_sums['total'] += losses.item()
                for k, v in loss_dict.items():
                    if k in val_loss_sums:
                        val_loss_sums[k] += v.mean().item()
                        
                val_loop.set_postfix(val_loss=losses.item())
                del images, targets, loss_dict, losses
                
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
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # Save every epoch for total control
        epoch_path = f"/kaggle/working/owod_model_epoch_{epoch+1}.pth"
        torch.save(checkpoint, epoch_path)
        print(f"Saved: {epoch_path}")
        
        # In Phase 1 we track best_val_loss for standard classification
        if save_best and val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            checkpoint['best_val_loss'] = best_val_loss
            best_path = "/kaggle/working/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"--> New Best Model saved to {best_path} (Val Loss: {best_val_loss:.4f})")
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
