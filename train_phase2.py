import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import csv
import random
import numpy as np
import warnings

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
# import config

# ==========================================
# 1. Custom DataParallel Wrapper
# ==========================================
# PyTorch's native DataParallel non sa come spezzare liste custom (images, targets, dino_features_list).
# Questo wrapper divide la lista a metà (o in parti uguali) in base a quante GPU sono disponibili,
# inviando la metà corretta a ciascuna GPU senza far schiantare il codice.
class OWODDataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        images = inputs[0]
        targets = inputs[1]
        dino_features_list = inputs[2] if len(inputs) > 2 else None
        
        # Calcoliamo quante immagini vanno in ciascuna GPU (es. batch 8 su 2 GPU = 4 immagini)
        batch_size = len(images)
        chunk_size = (batch_size + len(device_ids) - 1) // len(device_ids)
        
        scattered_inputs = []
        for i in range(min(len(device_ids), (batch_size + chunk_size - 1) // chunk_size)):
            dev = torch.device(f'cuda:{device_ids[i]}')
            start = i * chunk_size
            end = min((i + 1) * chunk_size, batch_size)
            
            # Trasferiamo i chunk sulle GPU corrette
            chunk_images = [img.to(dev) for img in images[start:end]]
            chunk_targets = [{k: v.to(dev) for k, v in t.items()} for t in targets[start:end]]
            
            if dino_features_list is not None:
                chunk_dino = [d.to(dev) for d in dino_features_list[start:end]]
            else:
                chunk_dino = None
            
            # DataParallel si aspetta una tupla di argomenti posizionali per ogni GPU
            scattered_inputs.append((chunk_images, chunk_targets, chunk_dino))
            
        return tuple(scattered_inputs), tuple({} for _ in scattered_inputs)


# ==========================================
# 2. Collate Function (CRITICAL for Detection)
# ==========================================
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    set_seed(42)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Starting training on: {device}")

    # ==========================================
    # 3. DINOv2 Initialization (Foundation Model)
    # ==========================================
    print("Loading DINOv2 (ViT-Small)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device) # Lo teniamo sulla GPU 0 perché fa pochissimo sforzo (21M parametri)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False
    
    # Attach DINOv2 directly to the base model so DataParallel clones it to both GPUs
    base_model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True, beta=0.1)
    base_model.dinov2 = dinov2
    # ==========================================
    train_dataset = OWODDataset(
        img_dir='/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017', 
        annotation_file='/kaggle/working/task1_10cls_uu_train.json', 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
        transform=None 
    )
    val_dataset = OWODDataset(
        img_dir='/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017', 
        annotation_file='/kaggle/working/task1_10cls_uu_val.json', 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
        transform=None
    )
    
    # BATCH SIZE DIMEZZATO A 4 PER LA FASE 2 (URM)
    # 2 immagini andranno a GPU 0 e 2 immagini a GPU 1.
    # Abbiamo impostato num_workers=0 e pin_memory=False per annientare QUALSIASI memory leak della CPU!
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ==========================================
    # 5. OWOD Network and Optimizer Initialization
    # ==========================================
    # Impostiamo beta=0.1 per l'ETM per bilanciare la loss (~2.4) con quella di classificazione (~0.15)
    model = base_model
    model.to(device) # CRITICAL: Sposta il modello su GPU prima di darlo al DataParallel!
    
    # MULTI-GPU INJECTION
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = OWODDataParallel(model)
        
    # Per accedere comodamente alle variabili (come use_etm), puntiamo sempre al modello "base" (interno al wrapper)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # ==========================================
    # 6. RESUME FROM CHECKPOINT AND CSV LOGGING
    # ==========================================
    num_epochs = 12
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    checkpoint_path = "/kaggle/working/owod_model_last.pth"
    
    # KAGGLE NOTEBOOK TRICK: Inserisci qui il nome esatto della cartella di input
    # che Kaggle ha creato quando hai aggiunto l'output del Notebook 1 al Notebook 2.
    # Di solito è qualcosa tipo: /kaggle/input/nome-del-notebook-1/best_model.pth
    imported_best_path = "/kaggle/input/notebooks/miriamruzza/train-until-ep-7/best_model.pth"
    
    # SETUP: Set to True to force resuming from Epoch 4 (when ETM activates)
    force_resume_epoch_4 = False
    
    if force_resume_epoch_4 and os.path.exists(epoch_4_path):
        pass # Ignorato per la Fase 2
    elif os.path.exists(checkpoint_path):
        print(f"Found checkpoint {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    elif os.path.exists(imported_best_path):
        print(f"Trovati i pesi della FASE 1 in {imported_best_path}!")
        print("Forzo la ripresa dall'Epoca 8 (inizio Fase 2 URM)...")
        base_model.load_state_dict(torch.load(imported_best_path, map_location=device), strict=False)
        start_epoch = 7 # L'indice 7 corrisponde all'Epoca 8
        best_val_loss = float('inf') # Resettiamo per iniziare la nuova fase
        
        # Facciamo una finta "load" di best_path locale per non rompere il curriculum learning dopo
        best_path = imported_best_path 
    else:
        print(f"ATTENZIONE: Non trovo i pesi in {imported_best_path}!")
        print("Assicurati di aver aggiunto l'output del Notebook 1 come dataset e aver corretto il percorso.")
        print("Parto da zero... ma questo è probabilmente un errore se volevi fare la Fase 2!")
        best_path = "/kaggle/working/best_model.pth"
        
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

    # ==========================================
    # 7. TRAINING LOOP 
    # ==========================================
    scaler = torch.cuda.amp.GradScaler() 
    
    for epoch in range(start_epoch, num_epochs):
        if epoch < 4:
            base_model.use_etm = False
            base_model.use_urm = False
        elif epoch < 7:
            base_model.use_etm = True
            base_model.use_urm = False
        else:
            base_model.use_etm = True
            base_model.use_urm = True
            
        if epoch == 4 or epoch == 7:
            print(f"--> New module activated at Epoch {epoch+1}.")
            
            # CURRICULUM LEARNING FIX: Ricarichiamo i pesi migliori della fase precedente prima di passare alla nuova!
            if os.path.exists(best_path):
                print(f"Loading BEST model weights from previous phase to avoid carrying over overfitted weights...")
                base_model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
                
            print("Resetting best loss tracking for the new phase!")
            best_val_loss = float('inf')
            patience_counter = 0
            
        print(f"Epoch {epoch+1} - Configuration: ETM={base_model.use_etm}, URM={base_model.use_urm}")

        model.train()
        train_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                           'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                           'loss_b_unk': 0.0, 'loss_et': 0.0}
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=30.0)
        
        for i, (images, targets) in enumerate(loop):
            # We no longer pre-compute DINOv2 features here! 
            # We let the DataParallel GPUs do it in parallel inside the model forward.
            
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets, None)
                # MULTI-GPU: loss_dict contiene array di loss (una per GPU). Usiamo .mean() per unificarle!
                losses = sum(loss.mean() for loss in loss_dict.values())
            
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sums['total'] += losses.item()
            for k, v in loss_dict.items():
                if k in train_loss_sums:
                    train_loss_sums[k] += v.mean().item() # Facciamo la mean anche per il logging
                
            loop.set_postfix(loss=losses.item())
            
            # CRITICAL MEMORY LEAK FIX: Force Python to delete variables and run Garbage Collection.
            # Python's GC often fails to clean up PyTorch DataParallel cyclic references fast enough!
            del images, targets, loss_dict, losses
            import gc
            gc.collect()
            
        num_batches_train = len(train_loader)
        train_avg = {k: v / num_batches_train for k, v in train_loss_sums.items()}
        print(f"End of Epoch {epoch+1} - Train Loss: {train_avg['total']:.4f} "
              f"(Cls: {train_avg['loss_classifier']:.4f}, Obj: {train_avg['loss_objectness']:.4f}, "
              f"Box: {train_avg['loss_box_reg']:.4f}, RPNBox: {train_avg['loss_rpn_box_reg']:.4f}, "
              f"URM: {train_avg['loss_b_unk']:.4f}, ET: {train_avg['loss_et']:.4f})")
        
        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        model.train() # FastRCNN calculates val losses only in train mode
        val_loss_sums = {'total': 0.0, 'loss_classifier': 0.0, 'loss_box_reg': 0.0, 
                         'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0, 
                         'loss_b_unk': 0.0, 'loss_et': 0.0}
        val_loop = tqdm(val_loader, desc="Validation", mininterval=30.0)
        
        with torch.no_grad():
            for images, targets in val_loop:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets, None)
                    losses = sum(loss.mean() for loss in loss_dict.values())
                
                val_loss_sums['total'] += losses.item()
                for k, v in loss_dict.items():
                    if k in val_loss_sums:
                        val_loss_sums[k] += v.mean().item()
                        
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
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(), # IMPORTANTE: salviamo sempre la rete base!
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        
        epoch_path = f"/kaggle/working/owod_model_epoch_{epoch+1}.pth"
        torch.save(checkpoint, epoch_path)
        
        if val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            patience_counter = 0
            print(f"New best model found! (Val Loss: {val_avg['total']:.4f}). Saving...")
            best_path = "/kaggle/working/best_model.pth"
            torch.save(base_model.state_dict(), best_path) # Anche qui salviamo solo i pesi del modello base
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early Stopping triggered! Training interrupted.")
                break
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
