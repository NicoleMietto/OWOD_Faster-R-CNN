import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align
from torchvision.models.detection.rpn import RegionProposalNetwork

# Import external modules
from labeler import OWOD_Labeler
from urm import UnknownBoxRefineModule
from etm import EmbeddingTransferModule
from OWOD_detector import EmbeddingHead, CustomRPN

class OWODFasterRCNN_Fixed(nn.Module):
    def __init__(self, num_known_classes, use_spatial_cnn=True, alpha=0.1, beta=1.0):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.alpha = alpha
        self.beta = beta
        self.use_etm = False
        self.use_urm = False
        self.use_obj_loss = False # Disattivata per non rompere la RPN
        
        # 1. Load Faster R-CNN with ResNet50
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone='DEFAULT')
        
        # 2. FREEZING for Colab/Kaggle
        for name, parameter in self.detector.backbone.body.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                parameter.requires_grad = False
                
        # 3. Adapt the final classifier
        # total num_classes = background (0) + known_classes (10) + UNKNOWN (1) = num_known_classes + 2
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_known_classes + 2)
        
        # Capture RPN objectness scores
        self.detector.rpn.__class__ = CustomRPN
        self.detector.rpn.rpn_scores = []
        
        # 4. Initialize Embedding module (CNN or MLP)
        self.embedding_head = EmbeddingHead(use_spatial_cnn=use_spatial_cnn)
        
        # 5. Initialize external modules
        self.labeler = OWOD_Labeler()
        
        import os
        sam_path = "mobile_sam.pt"
        if not os.path.exists(sam_path):
            import urllib.request
            print("Downloading MobileSAM weights (mobile_sam.pt)...")
            try:
                urllib.request.urlretrieve("https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt", sam_path)
            except Exception as e:
                print(f"Error downloading MobileSAM: {e}")
                
        self.urm = UnknownBoxRefineModule(sam_checkpoint_path=sam_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.etm = EmbeddingTransferModule()
        
    def forward(self, images, targets=None, dino_features_list=None):
        if self.training:
            # --- PHASE 1: Feature Extraction & RPN ---
            images_list, targets_list = self.detector.transform(images, targets)
            features = self.detector.backbone(images_list.tensors)
            
            # 1. Primo Passaggio RPN (estrae le proposal, ma la loss calcolata penalizza gli unknown)
            proposals, _ = self.detector.rpn(images_list, features, targets_list)
            
            # --- DINOv2 FEATURE EXTRACTION (Batched & Aligned) ---
            if self.use_etm and hasattr(self, 'dinov2'):
                with torch.no_grad():
                    augmented_images = images_list.tensors
                    pad_h = (14 - augmented_images.shape[2] % 14) % 14
                    pad_w = (14 - augmented_images.shape[3] % 14) % 14
                    img_padded = F.pad(augmented_images, (0, pad_w, 0, pad_h))
                    
                    features_dict = self.dinov2.forward_features(img_padded)
                    patch_tokens = features_dict['x_norm_patchtokens']
                    B_dino, _, C_dino = patch_tokens.shape
                    
                    H_dino = img_padded.shape[2] // 14
                    W_dino = img_padded.shape[3] // 14
                    batched_dino_features = patch_tokens.permute(0, 2, 1).reshape(B_dino, C_dino, H_dino, W_dino)
            else:
                batched_dino_features = None
            
            objectness_scores = self.detector.rpn.rpn_scores
            device = images_list.tensors.device
            
            dummy_loss = (self.embedding_head.network[-1].weight.sum() * 0.0).to(device)
            total_loss_b_unk = dummy_loss.clone()
            total_loss_et = dummy_loss.clone()
            
            all_valid_boxes = [] 
            refined_unknowns_list = [] # Salviamo le predizioni per l'iniezione
            
            for i in range(len(images)):
                labels_dict = self.labeler.assign_labels_known_unknown_background(
                    proposals[i], 
                    objectness_scores[i], 
                    targets_list[i]['boxes'], 
                    targets_list[i]['labels']
                )
                
                pred_known_boxes = labels_dict['known_boxes'][:10]
                
                raw_unknowns = labels_dict['unknown_boxes']
                unk_scores = labels_dict['unknown_scores']
                
                keep_idx = torchvision.ops.nms(raw_unknowns, unk_scores, iou_threshold=0.3)
                filtered_unknowns = raw_unknowns[keep_idx]
                unk_scores = unk_scores[keep_idx]
                
                _, sort_idx = unk_scores.sort(descending=True)
                filtered_unknowns = filtered_unknowns[sort_idx]
                sorted_unk_scores = unk_scores[sort_idx]
                
                top_unknowns = filtered_unknowns[:5]
                top_unk_scores = sorted_unk_scores[:5]
                
                if self.use_urm:
                    # Passiamo None come punteggi per disattivare la loss sperimentale che destabilizza la RPN
                    refined_unknowns, loss_b_unk = self.urm(top_unknowns, images_list.tensors[i], None)
                    total_loss_b_unk = total_loss_b_unk + loss_b_unk
                else:
                    refined_unknowns = top_unknowns
                
                refined_unknowns_list.append(refined_unknowns)
                
                valid_boxes_for_etm = torch.cat([pred_known_boxes, refined_unknowns])
                all_valid_boxes.append(valid_boxes_for_etm)
                
            # --- FASE 5.5: INIEZIONE DEGLI UNKNOWN NEI TARGET ---
            augmented_targets_list = []
            for i in range(len(images)):
                # Clona per evitare side-effects
                tgt = {k: v.clone() for k, v in targets_list[i].items()}
                ref_unk = refined_unknowns_list[i]
                
                # In Fase 1 (burn-in), NON iniettiamo gli unknown. 
                # Lo facciamo solo in Fase 2 o 3 (quando si attivano i moduli Open World).
                if len(ref_unk) > 0 and (self.use_etm or self.use_urm):
                    tgt['boxes'] = torch.cat([tgt['boxes'], ref_unk])
                    # La label '11' per gli unknown (num_known_classes + 1)
                    unk_labels = torch.full((len(ref_unk),), self.num_known_classes + 1, dtype=tgt['labels'].dtype, device=device)
                    tgt['labels'] = torch.cat([tgt['labels'], unk_labels])
                    
                augmented_targets_list.append(tgt)
                
            # --- FASE 6: ETM ---
            if self.use_etm and batched_dino_features is not None:
                total_boxes = sum(len(b) for b in all_valid_boxes)
                if total_boxes > 0:
                    feat_tensor = features['0'] if isinstance(features, dict) else features
                    roi_features = roi_align(feat_tensor, all_valid_boxes, output_size=(7, 7), spatial_scale=1/4.0)
                    all_instance_embeddings = self.embedding_head(roi_features)
                    num_boxes_per_image = [len(b) for b in all_valid_boxes]
                    split_embeddings = torch.split(all_instance_embeddings, num_boxes_per_image)
                    
                    for i in range(len(images)):
                        if num_boxes_per_image[i] > 0:
                            dino_feat_i = batched_dino_features[i:i+1]
                            loss_et = self.etm(dino_feat_i, [all_valid_boxes[i]], split_embeddings[i])
                            total_loss_et = total_loss_et + loss_et
                
            # --- FASE 7: RPN RE-PASS & RoI Head (La magia) ---
            # Ricalcoliamo le loss della RPN usando i target "aumentati".
            # La RPN smetterà di trattare i box unknown come background e inizierà a proporli!
            _, proposal_losses_fixed = self.detector.rpn(images_list, features, augmented_targets_list)
            
            # La RoI Head impara a classificare gli unknown esplicitamente come classe 11
            detections, detector_losses = self.detector.roi_heads(features, proposals, images_list.image_sizes, augmented_targets_list)
            
            total_losses = {}
            total_losses.update(proposal_losses_fixed)
            total_losses.update(detector_losses)
            total_losses['loss_b_unk'] = (total_loss_b_unk / len(images)) * self.alpha
            total_losses['loss_et'] = (total_loss_et / len(images)) * self.beta
            
            self.detector.rpn.rpn_scores = []
            if hasattr(self.detector.roi_heads.box_roi_pool, 'cached_roi_features'):
                self.detector.roi_heads.box_roi_pool.cached_roi_features = None
            
            return total_losses
            
        else:
            # --- INFERENCE ---
            original_image_sizes = []
            for img in images:
                val = img.shape[-2:]
                original_image_sizes.append((val[0], val[1]))
                
            images_list, _ = self.detector.transform(images, None)
            features = self.detector.backbone(images_list.tensors)
            proposals, _ = self.detector.rpn(images_list, features, None)
            
            box_features = self.detector.roi_heads.box_roi_pool(features, proposals, images_list.image_sizes)
            box_features = self.detector.roi_heads.box_head(box_features)
            class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)
            
            # Lasciamo che torchvision gestisca la classe 11 in automatico con la sua NMS nativa!
            boxes, scores, labels = self.detector.roi_heads.postprocess_detections(class_logits, box_regression, proposals, images_list.image_sizes)
            
            # I filtri post-processing standard taglieranno fuori il rumore in automatico
            detections = [{"boxes": b, "scores": s, "labels": l} for b, s, l in zip(boxes, scores, labels)]
            detections = self.detector.transform.postprocess(detections, images_list.image_sizes, original_image_sizes)
            
            self.detector.rpn.rpn_scores = []
            return detections
