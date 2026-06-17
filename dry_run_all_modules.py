import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
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
    set_seed(42)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Starting DRY RUN on: {device}")

    print("Loading DINOv2 (ViT-Small)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False
    
    base_model = OWODFasterRCNN(num_known_classes=10, use_spatial_cnn=True, beta=0.1)
    base_model.dinov2 = dinov2
    
    # ACCENDIAMO SUBITO TUTTI I MODULI PER IL TEST
    print("FORZATURA ATTIVA: Accensione immediata di ETM e URM per testare stabilità e tempi!")
    base_model.use_etm = True
    base_model.use_urm = True

    train_dataset = OWODDataset(
        img_dir='/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017', 
        annotation_file='/kaggle/working/task1_10cls_uu_train.json', 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
        transform=None 
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = base_model
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Dry Run!")
        model = OWODDataParallel(model)
        
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() 
    
    print("\n" + "="*50)
    print("INIZIO DRY RUN INTERATTIVA (Eseguirà solo 10 batch!)")
    print("="*50)
    
    model.train()
    
    # Impostiamo il desc a tqdm per leggere i dati di performance (it/s e tempo totale stimato)
    loop = tqdm(train_loader, desc=f"Dry Run (ALL MODULES ON)")
    
    for i, (images, targets) in enumerate(loop):
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets, None)
            losses = sum(loss.mean() for loss in loss_dict.values())
        
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Mostra le loss correnti
        loop.set_postfix(
            loss=f"{losses.item():.4f}", 
            ET=f"{loss_dict.get('loss_et', torch.tensor(0.0)).mean().item():.4f}", 
            URM=f"{loss_dict.get('loss_b_unk', torch.tensor(0.0)).mean().item():.4f}"
        )
        
        # Fermati dopo 10 batch. 10 batch sono sufficienti per far stabilizzare
        # l'indicatore it/s di tqdm e darti una stima perfetta del tempo che ci metterebbe un'intera epoca.
        if i >= 9:
            print("\n" + "="*50)
            print("DRY RUN COMPLETATA CON SUCCESSO!")
            print("Nessun crash multi-GPU rilevato. Guarda i dati 'it/s' di tqdm qui sopra per calcolare il tempo per l'epoca intera.")
            print("="*50)
            break

if __name__ == "__main__":
    main()
