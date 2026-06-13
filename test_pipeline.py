import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from OWOD_dataset import OWODDataset
from OWOD_detector import OWODFasterRCNN

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"=== TEST PIPELINE INIZIALE su {device} ===")

    print("Caricamento DINOv2 (ViT-Small)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)
    dinov2.eval()
    for param in dinov2.parameters():
        param.requires_grad = False

    print("Setup Datasets...")
    # Sostituire i percorsi quando su Kaggle!
    try:
        train_dataset = OWODDataset(
            img_dir="path/to/coco/train2017", 
            annotation_file="path/to/task1_uu_train.json", 
            known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            transform=None
        )
        val_dataset = OWODDataset(
            img_dir="path/to/coco/train2017", 
            annotation_file="path/to/task1_uu_val.json", 
            known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            transform=None
        )
    except FileNotFoundError:
        print("ATTENZIONE: File JSON non trovati. Assicurati di generare gli split e inserire i path corretti!")
        return

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    print("Inizializzazione Modello OWOD...")
    model = OWODFasterRCNN(num_known_classes=20, use_spatial_cnn=True).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    # Forziamo subito tutti i moduli a True per testare se la memoria regge o se ci sono errori logici
    model.use_etm = True
    model.use_urm = True
    print(f"Moduli attivati: ETM={model.use_etm}, URM={model.use_urm}")

    model.train()
    
    # Eseguiamo solo UN batch (2 immagini) per testare il forward, i moduli e il backward pass
    print("\n--- TEST: ITERAZIONE DI TRAINING (1 Batch) ---")
    for i, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # DINO
        dino_features_list = []
        with torch.no_grad():
            for img in images:
                _, h, w = img.shape
                new_h, new_w = (h // 14) * 14, (w // 14) * 14
                img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
                features_dict = dinov2.forward_features(img_resized)
                patch_tokens = features_dict['x_norm_patchtokens']
                C = patch_tokens.shape[-1]
                dino_features_list.append(patch_tokens.permute(0, 2, 1).reshape(1, C, new_h // 14, new_w // 14))
                
        # Forward pass
        loss_dict = model(images, targets, dino_features_list)
        losses = sum(loss for loss in loss_dict.values())
        
        print(f"Loss calcolata con successo: {losses.item():.4f}")
        print("Dettaglio Loss:")
        for k, v in loss_dict.items():
            print(f"  - {k}: {v.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        print("Backward pass eseguito con successo! (Nessun errore sui gradienti)")
        
        # Interrompiamo dopo il primo batch!
        break

    print("\n--- TEST: ITERAZIONE DI VALIDAZIONE (1 Batch) ---")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            val_dino_features = []
            for img in images:
                _, h, w = img.shape
                new_h, new_w = (h // 14) * 14, (w // 14) * 14
                img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
                features_dict = dinov2.forward_features(img_resized)
                patch_tokens = features_dict['x_norm_patchtokens']
                C = patch_tokens.shape[-1]
                val_dino_features.append(patch_tokens.permute(0, 2, 1).reshape(1, C, new_h // 14, new_w // 14))
                
            loss_dict = model(images, targets, val_dino_features)
            val_loss = sum(loss for loss in loss_dict.values())
            print(f"Validation Forward eseguito con successo. Loss: {val_loss.item():.4f}")
            break
            
    print("\n=== TEST COMPLETATO CON SUCCESSO! IL MODELLO È PRONTO PER L'ADDESTRAMENTO ===")

if __name__ == "__main__":
    main()
