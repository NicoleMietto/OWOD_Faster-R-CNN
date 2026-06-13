import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align

# Importiamo i moduli esterni
from labeler import OWOD_Labeler
from urm import UnknownBoxRefineModule
from etm import EmbeddingTransferModule

class EmbeddingHead(nn.Module):
    """
    Questo è il cuore del TUO esperimento. 
    Prende i riquadri estratti dal RoI-Align (N, 256, 7, 7) e produce gli embeddings Z_i.
    """
    def __init__(self, in_channels=256, roi_size=7, embedding_dim=256, use_spatial_cnn=True):
        super().__init__()
        self.use_spatial_cnn = use_spatial_cnn
        
        if self.use_spatial_cnn:
            # ESPERIMENTO NUOVO: Manteniamo l'info spaziale con una CNN
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), # Comprime spazialmente solo alla fine
                nn.Flatten(),
                nn.Linear(256, embedding_dim)
            )
        else:
            # BASELINE (Paper): Appiattisce tutto subito perdendo l'info topologica
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * roi_size * roi_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, embedding_dim)
            )

    def forward(self, x):
        return self.network(x)

class OWODFasterRCNN(nn.Module):
    def __init__(self, num_known_classes, use_spatial_cnn=True, alpha=0.1, beta=1.0):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.alpha = alpha
        self.beta = beta
        self.use_etm = False
        self.use_urm = False
        
        # 1. Carichiamo Faster R-CNN con ResNet50 pre-addestrata
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        
        # 2. FREEZING per Colab/Kaggle: Congeliamo i primi blocchi della ResNet
        # La ResNet50 in torchvision ha una 'backbone.body'
        for name, parameter in self.detector.backbone.body.named_parameters():
            if 'layer1' in name or 'layer2' in name: # Congeliamo i primi due blocchi
                parameter.requires_grad = False
                
        # 3. Adattiamo il classificatore finale
        # num_classes totali = background (0) + known_classes + 1 (classe fittizia per gli unknown)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_known_classes + 2)

        self.rpn_scores = [] # Qui salveremo i punteggi intercettati
        
        # Salviamo la funzione originale di PyTorch
        original_filter = self.detector.rpn.filter_proposals
        
        # Creiamo un "Hook" (Gancio)
        def hook_filter_proposals(*args, **kwargs):
            # Chiamiamo la funzione originale. Restituisce i box e i punteggi!
            boxes, scores = original_filter(*args, **kwargs)
            self.rpn_scores = scores # Li rubiamo e li salviamo qui
            return boxes, scores     # Restituiamo i box a PyTorch come si aspetta
            
        # Sostituiamo la funzione di PyTorch con la nostra!
        self.detector.rpn.filter_proposals = hook_filter_proposals
        
        # 4. Inizializziamo il TUO modulo di Embedding (CNN o MLP)
        self.embedding_head = EmbeddingHead(use_spatial_cnn=use_spatial_cnn)
        
        # 5. Inizializziamo i moduli esterni
        self.labeler = OWOD_Labeler()
        
        # Download automatico di MobileSAM se non presente
        import os
        sam_path = "mobile_sam.pt"
        if not os.path.exists(sam_path):
            import urllib.request
            print("Scaricamento dei pesi di MobileSAM (mobile_sam.pt)...")
            try:
                urllib.request.urlretrieve("https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt", sam_path)
            except Exception as e:
                print(f"Errore download MobileSAM: {e}")
                
        self.urm = UnknownBoxRefineModule(sam_checkpoint_path=sam_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.etm = EmbeddingTransferModule()
        
    def forward(self, images, targets=None, dino_features_list=None):
        if self.training:
            # --- FASE 1: Feature Extraction & RPN ---
            images_list, targets_list = self.detector.transform(images, targets)
            features = self.detector.backbone(images_list.tensors)

            print(self.detector.transform.image_mean)
            # Output: [0.485, 0.456, 0.406]

            print(self.detector.transform.image_std)
            # Output: [0.229, 0.224, 0.225]

            print(self.detector.transform.min_size)
            # Output: (800,) -> Significa che fa il resize del lato corto a 800 pixel
            
            # La RPN genera le proposal grezze (selezione blanda, ne genera migliaia)
            proposals, proposal_losses = self.detector.rpn(images_list, features, targets_list)
            
            # 2. ECCO LA MAGIA: Recuperi gli scores appena rubati dall'hook!
            objectness_scores = self.rpn_scores
            
            device = images_list.tensors.device
            total_loss_b_unk = torch.tensor(0.0, device=device)
            total_loss_et = torch.tensor(0.0, device=device)
            augmented_targets = []
            
            # Iteriamo per ogni immagine nel batch
            for i in range(len(images)):
                # 2. Estraiamo la feature map di DINO per questa specifica immagine
                dino_features_i = dino_features_list[i] if dino_features_list is not None else None

                # --- Pseudo-Labeling ---
                labels_dict = self.labeler.assign_labels_known_unknown_background(
                    proposals[i], 
                    objectness_scores[i], # Ora questo funziona perfettamente!
                    targets_list[i]['boxes'], 
                    targets_list[i]['labels']
                )
                
                # Le PREDIZIONI grezze (imprecise) per i known! (Risposta al tuo dubbio)
                pred_known_boxes = labels_dict['known_boxes'] 
                
                # --- NMS per non sprecare i Top-K (Risposta al tuo dubbio) ---
                raw_unknowns = labels_dict['unknown_boxes']
                unk_scores = labels_dict['unknown_scores'] # Dovrai restituire anche gli scores dal labeler
                
                # Applichiamo NMS tra gli unknown. Se si sovrappongono troppo, tiene solo il migliore.
                keep_idx = torchvision.ops.nms(raw_unknowns, unk_scores, iou_threshold=0.3)
                filtered_unknowns = raw_unknowns[keep_idx]
                
                # ORA prendiamo i Top-K senza sprechi
                top_k = 10
                top_unknowns = filtered_unknowns[:top_k]
                
                # --- FASE 3: URM (MobileSAM) ---
                if self.use_urm:
                    refined_unknowns, loss_b_unk = self.urm(top_unknowns, images_list.tensors[i])
                    total_loss_b_unk = total_loss_b_unk + loss_b_unk
                else:
                    refined_unknowns = torch.empty((0, 4), device=device)
                
                # --- FASE 4: ETM (DINOv2) ---
                valid_boxes_for_etm = torch.cat([pred_known_boxes, refined_unknowns])
                
                if self.use_etm and len(valid_boxes_for_etm) > 0 and dino_features_i is not None:
                    # In FPN, features è un OrderedDict. Usiamo la feature map con maggior risoluzione '0'
                    feat_tensor = features['0'] if isinstance(features, dict) else features
                    roi_features = roi_align(feat_tensor, [valid_boxes_for_etm], output_size=(7, 7), spatial_scale=1/4.0)
                    instance_embeddings = self.embedding_head(roi_features)
                    
                    loss_et = self.etm(dino_features_i, [valid_boxes_for_etm], instance_embeddings)
                    total_loss_et = total_loss_et + loss_et
                
                # --- FASE 5: Iniezione Target per la RoI Head Blanda ---
                # Aggiungiamo i riquadri perfezionati da SAM ai Ground Truth dell'immagine.
                # L'etichetta sarà num_known_classes + 1 (la classe fittizia Unknown)
                new_target = targets_list[i].copy()
                if len(refined_unknowns) > 0:
                    new_labels = torch.full((len(refined_unknowns),), self.num_known_classes + 1, dtype=torch.int64, device=refined_unknowns.device)
                    new_target['boxes'] = torch.cat([new_target['boxes'], refined_unknowns])
                    new_target['labels'] = torch.cat([new_target['labels'], new_labels])
                augmented_targets.append(new_target)

            # --- FASE 6: RoI Head Standard (Detection Blanda) ---
            # Diamo in pasto TUTTE le proposals della RPN e i target aumentati alla rete base.
            # Lei farà il matching blando (IoU > 0.5 per i positivi) e calcolerà le loss di classificazione e regressione.
            detector_losses = self.detector.roi_heads(features, proposals, images_list.image_sizes, augmented_targets)
            
            # Combina tutte le loss
            total_losses = {}
            total_losses.update(proposal_losses)
            total_losses.update(detector_losses)
            total_losses['loss_b_unk'] = (total_loss_b_unk / len(images)) * self.alpha
            total_losses['loss_et'] = (total_loss_et / len(images)) * self.beta
            
            return total_losses
            
        else:
            return self.detector(images)
