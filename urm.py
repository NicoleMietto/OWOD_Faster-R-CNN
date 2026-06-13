import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F

# MobileSAM deve essere installato nell'ambiente (es. Colab/Kaggle) con:
# !pip install git+https://github.com/ChaoningZhang/MobileSAM.git
from mobile_sam import sam_model_registry, SamPredictor

class UnknownBoxRefineModule(nn.Module):
    """
    Modulo URM: Raffina i riquadri sconosciuti usando MobileSAM.
    Scarta le allucinazioni e calcola la loss di raffinamento per addestrare la RPN.
    """
    def __init__(self, sam_checkpoint_path, device='cuda', iou_filter_thresh=0.5):
        super().__init__()
        self.device = device
        self.iou_filter_thresh = iou_filter_thresh
        
        # 1. Caricamento di MobileSAM (ViT-Tiny per risparmiare VRAM su Kaggle)
        model_type = "vit_t"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=self.device)
        self.sam.eval() # SAM agisce da "insegnante", è sempre in eval mode
        
        # Freeziamo esplicitamente tutti i pesi per non consumare memoria nei gradienti
        for param in self.sam.parameters():
            param.requires_grad = False
            
        self.predictor = SamPredictor(self.sam)

    def forward(self, top_unknown_boxes, raw_image_tensor):
        """
        Args:
            top_unknown_boxes (Tensor): [K, 4] riquadri uscenti dalla RPN (hanno i gradienti attivi).
            raw_image_tensor (Tensor): [3, H, W] l'immagine originale (RGB, range 0-255).
            
        Returns:
            valid_sam_boxes (Tensor): [M, 4] I riquadri perfetti (M <= K), usati per ETM e RoI.
            loss_b_unk (Tensor): Valore scalare della loss di raffinamento.
        """
        # Se la RPN non ha trovato nessun unknown, non facciamo nulla
        if len(top_unknown_boxes) == 0:
            return torch.empty((0, 4), device=self.device), torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- STEP 1: De-normalizzazione e Preparazione per SAM ---
        # torchvision usa questa media e deviazione standard di default
        mean = torch.tensor([0.485, 0.456, 0.406], device=raw_image_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=raw_image_tensor.device).view(3, 1, 1)
        
        # Invertiamo la formula della normalizzazione: (img * std) + mean
        unnorm_img = (raw_image_tensor * std) + mean
        
        # Assicuriamoci che i valori non sbordino e convertiamo in HWC numpy array uint8 (0-255)
        img_np = (unnorm_img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')

        # Passiamo l'immagine corretta all'encoder di SAM
        self.predictor.set_image(img_np)
        
        # --- STEP 2: Batched Prompting (Passiamo tutti i Top-K box insieme) ---
        boxes_np = top_unknown_boxes.detach().cpu().numpy()
        
        # Trasformiamo le coordinate dei box nel formato dimensionale che SAM si aspetta
        input_boxes = self.predictor.transform.apply_boxes(boxes_np, self.predictor.original_size)
        input_boxes_torch = torch.tensor(input_boxes, device=self.predictor.device)
        
        # Inferenza senza gradienti
        with torch.no_grad():
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=input_boxes_torch,
                multimask_output=False, # Vogliamo solo la maschera "migliore" per ogni box
            )
        # 'masks' ha dimensione [K, 1, H, W]
        
        # --- STEP 3: Generazione dei Nuovi Bounding Box ---
        # Rimuoviamo la dimensione del canale (masks[:, 0]) e passiamo da maschera booleana a box
        sam_boxes = ops.masks_to_boxes(masks[:, 0])
        
        # --- STEP 4: Filtraggio delle Allucinazioni ---
        # Calcoliamo la IoU tra le proposals originali della RPN e le maschere perfette di SAM.
        # ops.box_iou restituisce una matrice KxK. A noi interessa solo la diagonale 
        # (ovvero il match tra il box N della RPN e il box N di SAM).
        ious = ops.box_iou(top_unknown_boxes, sam_boxes).diag()
        
        # Teniamo solo quelli in cui SAM ha confermato l'intuizione della RPN
        keep_mask = ious >= self.iou_filter_thresh
        
        valid_proposals = top_unknown_boxes[keep_mask]
        valid_sam_boxes = sam_boxes[keep_mask].detach() # Questi ora sono "Ground Truth", si fa il detach()!
        
        if len(valid_sam_boxes) == 0:
             return torch.empty((0, 4), device=self.device), torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- STEP 5: Calcolo della Loss (L_b,unk) ---
        # Questa loss addestra la RPN (valid_proposals ha i gradienti) a generare
        # fin da subito dei box più simili a quelli perfetti di SAM.
        
        # 5a. L1 Loss (Distanza assoluta tra le coordinate)
        loss_l1 = F.l1_loss(valid_proposals, valid_sam_boxes, reduction='mean')
        
        # 5b. GIoU Loss (Generalised IoU)
        giou_matrix = ops.generalized_box_iou(valid_proposals, valid_sam_boxes)
        loss_giou = 1.0 - giou_matrix.diag().mean()
        
        # Loss totale combinata (come nel paper originale)
        loss_b_unk = loss_l1 + loss_giou 
        
        return valid_sam_boxes, loss_b_unk