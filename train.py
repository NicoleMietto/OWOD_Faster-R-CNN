import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Importiamo i moduli che abbiamo creato
from OWOD_dataset import OWODDataset
from OWOD_detector import OWODFasterRCNN
# import torchvision.transforms.v2 as transforms (assicurati di usare le trasformazioni definite in OWODDataset)

# ==========================================
# 1. Collate Function (FONDAMENTALE per la Detection)
# ==========================================
# PyTorch di default cerca di impilare i tensori in matrici perfette.
# Poiché ogni immagine ha un numero DIVERSO di bounding box, andrebbe in crash.
# Questa funzione dice al DataLoader di tenere immagini e target in liste separate.
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Inizio addestramento su: {device}")

    # ==========================================
    # 2. Inizializzazione DINOv2 (Foundation Model)
    # ==========================================
    print("Caricamento DINOv2 (ViT-Small)...")
    # Usiamo il modello piccolo (21M parametri) per stare comodamente nella VRAM
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)
    dinov2.eval() # Solo inferenza
    for param in dinov2.parameters():
        param.requires_grad = False

    # ==========================================
    # 3. Setup Dataset e DataLoader
    # ==========================================
    # (Sostituisci con le tue transforms se necessario)
    train_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file="task1_uu_train.json", 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        transform=None 
    )
    val_dataset = OWODDataset(
        img_dir="/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017", 
        annotation_file="task1_uu_val.json", 
        known_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        transform=None
    )
    
    # Batch size basso (es. 2 o 4) per via del peso dei modelli su GPU free
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # ==========================================
    # 4. Inizializzazione Rete OWOD e Ottimizzatore
    # ==========================================
    # Numero classi = 20 note (il background e la classe fittizia li gestisce internamente)
    model = OWODFasterRCNN(num_known_classes=20, use_spatial_cnn=True).to(device)
    
    # Optimizer (AdamW è standard per reti moderne. Addestriamo solo i parametri che lo richiedono)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # ==========================================
    # 5. LOOP DI ADDESTRAMENTO E EARLY STOPPING
    # ==========================================
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # --- STRATEGIA DI WARM-UP (attivazione progressiva moduli) ---
        if epoch < 4:
            model.use_etm = False
            model.use_urm = False
        elif epoch < 8:
            model.use_etm = True
            model.use_urm = False
        else:
            model.use_etm = True
            model.use_urm = True
            
        print(f"Epoca {epoch+1} - Configurazione: ETM={model.use_etm}, URM={model.use_urm}")

        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in loop:
            # Spostiamo immagini e target sulla GPU
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # --- ESTRAZIONE FEATURE DINOv2 ---
            dino_features_list = None
            if model.use_etm:
                with torch.no_grad():
                    # DINOv2 richiede che l'immagine sia multipla di 14. 
                    # Facciamo un resize "al volo" solo per DINO
                    dino_features_list = []
                    for img in images:
                        _, h, w = img.shape
                        new_h, new_w = (h // 14) * 14, (w // 14) * 14
                        img_resized = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
                        
                        # Estraiamo i token spaziali da DINO (escludendo il token CLS globale)
                        features_dict = dinov2.forward_features(img_resized)
                        patch_tokens = features_dict['x_norm_patchtokens'] # Shape: [1, N, C]
                        
                        # Rimodelliamo i token in una griglia 2D spaziale [1, C, H_grid, W_grid]
                        C = patch_tokens.shape[-1]
                        dino_2d = patch_tokens.permute(0, 2, 1).reshape(1, C, new_h // 14, new_w // 14)
                        dino_features_list.append(dino_2d)
                    
            # --- FORWARD PASS OWOD ---
            # Passiamo le immagini, i target e la lista delle feature di DINO al nostro detector
            # (Assicurati di aggiungere `dino_features_list` come parametro nel forward di OWOD_detector)
            loss_dict = model(images, targets, dino_features_list)
            
            # Sommiamo tutte le loss (RPN, RoI, L_b_unk, L_et)
            losses = sum(loss for loss in loss_dict.values())
            
            # --- BACKPROPAGATION ---
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient Clipping (Previene esplosioni matematiche se SAM/DINO danno output strani inizialmente)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            total_loss += losses.item()
            loop.set_postfix(loss=losses.item())
            
        train_loss_avg = total_loss / len(train_loader)
        print(f"Fine Epoca {epoch+1} - Train Loss Media: {train_loss_avg:.4f}")
        
        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        # model è in train() mode, necessario per Faster R-CNN per restituire le loss.
        # Usiamo torch.no_grad() per non costruire il grafo.
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
        print(f"Epoca {epoch+1} - Val Loss Media: {val_loss_avg:.4f}")
        
        # Salvataggio epoca corrente
        torch.save(model.state_dict(), f"owod_model_last.pth")
        
        # EARLY STOPPING LOGIC
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            print(f"Nuovo miglior modello trovato! (Val Loss: {val_loss_avg:.4f}). Salvataggio...")
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(f"Nessun miglioramento. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early Stopping innescato! Interruzione dell'addestramento.")
                break
        
        # Svuotiamo la cache per l'epoca successiva
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()