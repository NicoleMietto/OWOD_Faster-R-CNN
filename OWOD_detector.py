import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align
from torchvision.models.detection.rpn import RegionProposalNetwork
class CustomRPN(RegionProposalNetwork):
    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        boxes, scores = super().filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)
        self.rpn_scores = scores
        return boxes, scores
# Import external modules
from labeler import OWOD_Labeler
from urm import UnknownBoxRefineModule
from etm import EmbeddingTransferModule
class EmbeddingHead(nn.Module):
    """
    This is the core of YOUR experiment. 
    It takes the boxes extracted by RoI-Align (N, 256, 7, 7) and produces the Z_i embeddings.
    """
    def __init__(self, in_channels=256, roi_size=7, embedding_dim=256, use_spatial_cnn=True):
        super().__init__()
        self.use_spatial_cnn = use_spatial_cnn
        
        if self.use_spatial_cnn:
            # NEW EXPERIMENT: We maintain spatial info with a CNN
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), # Spatially compress only at the end
                nn.Flatten(),
                nn.Linear(256, embedding_dim)
            )
        else:
            # BASELINE (Paper): Flattens everything immediately, losing topological info
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
        
        # 1. Load Faster R-CNN with ResNet50 pre-trained ONLY on ImageNet (not COCO!)
        # It is CRITICAL to use weights=None and weights_backbone='DEFAULT' for Open World.
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone='DEFAULT')
        
        # 2. FREEZING for Colab/Kaggle: Freeze the first blocks of ResNet
        # torchvision's ResNet50 has a 'backbone.body'
        for name, parameter in self.detector.backbone.body.named_parameters():
            if 'layer1' in name or 'layer2' in name: # Freeze the first two blocks
                parameter.requires_grad = False
                
        # 3. Adapt the final classifier
        # total num_classes = background (0) + known_classes (e.g. 20 for task 1)
        # Note: We remove the +2 to not create useless output nodes.
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_known_classes + 1)
        # 3. MAGIC TRICK: Capture RPN objectness scores
        # We change the class of the existing RPN to our CustomRPN dynamically.
        # This safely survives DataParallel deepcopying without closure bugs!
        self.detector.rpn.__class__ = CustomRPN
        self.detector.rpn.rpn_scores = []
        
        # 4. Initialize YOUR Embedding module (CNN or MLP)
        self.embedding_head = EmbeddingHead(use_spatial_cnn=use_spatial_cnn)
        
        # 5. Initialize external modules
        self.labeler = OWOD_Labeler()
        
        # Automatic download of MobileSAM if not present
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
        """
        Args:
            images: list of Tensors (Raw images, unpadded, values 0-255 roughly usually, actually normalized by transform later)
            targets: ...
            dino_features_list: computed internally now!
        """
        # Determine the current device dynamically based on the first image
        current_device = images[0].device
        
        # Removed sequential DINOv2 feature extraction here. 
        # It is now computed internally after Faster R-CNN transformations to guarantee coordinate alignment.
        if self.training:
            # --- PHASE 1: Feature Extraction & RPN ---
            images_list, targets_list = self.detector.transform(images, targets)
            features = self.detector.backbone(images_list.tensors)
            
            # The RPN generates raw proposals (mild selection, generates thousands)
            proposals, proposal_losses = self.detector.rpn(images_list, features, targets_list)
            
            # --- DINOv2 FEATURE EXTRACTION (Batched & Aligned) ---
            if self.use_etm and hasattr(self, 'dinov2'):
                with torch.no_grad():
                    # images_list.tensors is (B, 3, H_pad, W_pad)
                    augmented_images = images_list.tensors
                    
                    # Pad to nearest multiple of 14 for DINOv2
                    pad_h = (14 - augmented_images.shape[2] % 14) % 14
                    pad_w = (14 - augmented_images.shape[3] % 14) % 14
                    img_padded = F.pad(augmented_images, (0, pad_w, 0, pad_h))
                    
                    features_dict = self.dinov2.forward_features(img_padded)
                    patch_tokens = features_dict['x_norm_patchtokens']
                    B_dino, _, C_dino = patch_tokens.shape
                    
                    # Compute spatial dimensions of the DINO feature map
                    H_dino = img_padded.shape[2] // 14
                    W_dino = img_padded.shape[3] // 14
                    batched_dino_features = patch_tokens.permute(0, 2, 1).reshape(B_dino, C_dino, H_dino, W_dino)
            else:
                batched_dino_features = None
            
            # 2. HERE IS THE MAGIC: Retrieve the scores just stolen by the CustomRPN!
            objectness_scores = self.detector.rpn.rpn_scores
            
            device = images_list.tensors.device
            
            # MULTI-GPU CRITICAL FIX: Se un'immagine non ha oggetti sconosciuti, la loss sarebbe 0.0.
            # Se restituiamo un 0.0 "puro" senza grad_fn, DataParallel crasha durante il backward()
            # se l'altra GPU ha invece trovato oggetti e ha una loss valida.
            # Creiamo un 0.0 matematicamente agganciato ai pesi della rete per ingannare l'Autograd!
            dummy_loss = (self.embedding_head.network[-1].weight.sum() * 0.0).to(device)
            total_loss_b_unk = dummy_loss.clone()
            total_loss_et = dummy_loss.clone()
            
            augmented_targets = []
            all_valid_boxes = [] # For ETM Batching
            
            # Iterate for each image in the batch
            for i in range(len(images)):
                # --- Pseudo-Labeling ---
                labels_dict = self.labeler.assign_labels_known_unknown_background(
                    proposals[i], 
                    objectness_scores[i], # This works perfectly now!
                    targets_list[i]['boxes'], 
                    targets_list[i]['labels']
                )
                
                # Raw (imprecise) PREDICTIONS for knowns! 
                pred_known_boxes = labels_dict['known_boxes'] 
                
                # Le proposte RPN sono già ordinate per objectness!
                # Limitiamo i known a 10 per immagine in modo da bilanciare l'ETM
                top_k_known = 10
                pred_known_boxes = pred_known_boxes[:top_k_known]
                
                # --- NMS to not waste Top-K ---
                raw_unknowns = labels_dict['unknown_boxes']
                unk_scores = labels_dict['unknown_scores'] # You will also need to return scores from the labeler
                
                # Apply NMS among unknowns. If they overlap too much, keep only the best.
                keep_idx = torchvision.ops.nms(raw_unknowns, unk_scores, iou_threshold=0.3)
                filtered_unknowns = raw_unknowns[keep_idx]
                
                # Sort by objectness (take the ones the RPN is most confident about)
                unk_scores = objectness_scores[i][keep_idx]
                _, sort_idx = unk_scores.sort(descending=True)
                filtered_unknowns = filtered_unknowns[sort_idx]
                
                # Get the TOP K proposals for unknowns to avoid overwhelming the ETM/URM
                # Abbassato a 5 per ridurre le allucinazioni e velocizzare l'addestramento
                top_k = 5
                top_unknowns = filtered_unknowns[:top_k]
                
                # --- PHASE 3: URM (MobileSAM) ---
                if self.use_urm:
                    refined_unknowns, loss_b_unk = self.urm(top_unknowns, images_list.tensors[i])
                    total_loss_b_unk = total_loss_b_unk + loss_b_unk
                else:
                    # Se l'URM è spento, passiamo comunque all'ETM i riquadri grezzi della RPN!
                    refined_unknowns = top_unknowns
                
                # --- PHASE 4: Prepare Boxes for ETM ---
                valid_boxes_for_etm = torch.cat([pred_known_boxes, refined_unknowns])
                all_valid_boxes.append(valid_boxes_for_etm)
            # --- PHASE 5: ETM (Batched Processing) ---
            if self.use_etm and batched_dino_features is not None:
                total_boxes = sum(len(b) for b in all_valid_boxes)
                if total_boxes > 0:
                    feat_tensor = features['0'] if isinstance(features, dict) else features
                    
                    # roi_align natively handles a list of tensors and assigns the correct batch index to each!
                    roi_features = roi_align(feat_tensor, all_valid_boxes, output_size=(7, 7), spatial_scale=1/4.0)
                    
                    # Process all embeddings in parallel
                    all_instance_embeddings = self.embedding_head(roi_features)
                    
                    # Split them back per image to compute the per-image contrastive loss
                    num_boxes_per_image = [len(b) for b in all_valid_boxes]
                    split_embeddings = torch.split(all_instance_embeddings, num_boxes_per_image)
                    
                    for i in range(len(images)):
                        if num_boxes_per_image[i] > 0:
                            # Pass the slice of batched_dino_features for image i (shape: 1xCxHxW)
                            dino_feat_i = batched_dino_features[i:i+1]
                            loss_et = self.etm(dino_feat_i, [all_valid_boxes[i]], split_embeddings[i])
                            total_loss_et = total_loss_et + loss_et
                
                # The RoI head will only train on the original known classes.
            # --- PHASE 6: Standard RoI Head (Mild Detection) ---
            # We feed ALL RPN proposals and the ORIGINAL targets to the base network.
            # It will compute classification and regression losses ONLY for known classes.
            # Unknown proposals will be naturally treated as background by the RoI head.
            detections, detector_losses = self.detector.roi_heads(features, proposals, images_list.image_sizes, targets_list)
            
            # Combine all losses
            total_losses = {}
            total_losses.update(proposal_losses)
            total_losses.update(detector_losses)
            total_losses['loss_b_unk'] = (total_loss_b_unk / len(images)) * self.alpha
            total_losses['loss_et'] = (total_loss_et / len(images)) * self.beta
            
            # CRITICAL MEMORY LEAK FIX: Clear the cached scores to break the computation graph reference cycle
            self.detector.rpn.rpn_scores = []
            if hasattr(self.detector.roi_heads.box_roi_pool, 'cached_roi_features'):
                self.detector.roi_heads.box_roi_pool.cached_roi_features = None
            
            return total_losses
            
        else:
            # --- CUSTOM INFERENCE FOR OPEN WORLD ---
            # 1. Prepare original image sizes for final rescaling
            original_image_sizes = []
            for img in images:
                val = img.shape[-2:]
                original_image_sizes.append((val[0], val[1]))
            # 2. Extract features and RPN proposals
            images_list, _ = self.detector.transform(images, None)
            features = self.detector.backbone(images_list.tensors)
            proposals, _ = self.detector.rpn(images_list, features, None)
            
            # Retrieve the objectness scores stolen by our CustomRPN
            objectness_scores = self.detector.rpn.rpn_scores
            
            # 3. Fast pass through RoI Head to get raw class logits
            box_features = self.detector.roi_heads.box_roi_pool(features, proposals, images_list.image_sizes)
            box_features = self.detector.roi_heads.box_head(box_features)
            class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)
            
            # 4. Standard postprocess for KNOWN objects
            # This handles NMS and score thresholding for classes 1 to 20
            boxes, scores, labels = self.detector.roi_heads.postprocess_detections(class_logits, box_regression, proposals, images_list.image_sizes)
            detections = [{"boxes": b, "scores": s, "labels": l} for b, s, l in zip(boxes, scores, labels)]
            
            # 5. Extract UNKNOWN objects
            num_proposals_per_img = [p.shape[0] for p in proposals]
            class_logits_per_img = class_logits.split(num_proposals_per_img, dim=0)
            
            for i in range(len(images)):
                logits_i = class_logits_per_img[i]
                probs_i = F.softmax(logits_i, dim=-1)
                
                # Max probability among all known classes (columns 1 to num_known_classes)
                # Note: class 0 is background
                known_scores, _ = probs_i[:, 1:self.num_known_classes+1].max(dim=1)
                
                # Unknown condition: High RPN objectness AND Low known class probability
                obj_scores_i = objectness_scores[i]
                is_unknown = (known_scores < 0.3) & (obj_scores_i >= 0.7)
                
                if is_unknown.any():
                    unk_boxes = proposals[i][is_unknown]
                    unk_scores = obj_scores_i[is_unknown]
                    
                    # Apply NMS specifically for unknown proposals
                    keep = torchvision.ops.nms(unk_boxes, unk_scores, iou_threshold=0.3)
                    unk_boxes = unk_boxes[keep]
                    unk_scores = unk_scores[keep]
                    
                    # --- CROSS-CLASS NMS ---
                    # Filter out unknown boxes that heavily overlap with already detected KNOWN boxes
                    known_boxes = detections[i]['boxes']
                    if len(known_boxes) > 0 and len(unk_boxes) > 0:
                        ious = torchvision.ops.box_iou(unk_boxes, known_boxes) # Shape: (num_unk, num_known)
                        max_ious, _ = ious.max(dim=1)
                        # Keep only unknown boxes that do NOT overlap significantly with any known box
                        cross_keep = max_ious < 0.4
                        unk_boxes = unk_boxes[cross_keep]
                        unk_scores = unk_scores[cross_keep]
                    
                    # Take top-K to avoid flooding predictions
                    # Limitiamo a 5 gli unknown estratti per immagine
                    top_k_infer = 5
                    unk_boxes = unk_boxes[:top_k_infer]
                    unk_scores = unk_scores[:top_k_infer]
                    
                    if len(unk_boxes) > 0:
                        # Add them to the image's detections with the dummy label 81
                        unk_labels = torch.full((len(unk_boxes),), 81, dtype=torch.int64, device=unk_boxes.device)
                        
                        detections[i]['boxes'] = torch.cat([detections[i]['boxes'], unk_boxes])
                        detections[i]['scores'] = torch.cat([detections[i]['scores'], unk_scores])
                        detections[i]['labels'] = torch.cat([detections[i]['labels'], unk_labels])
            
            # 6. Transform all boxes back to original image sizes
            detections = self.detector.transform.postprocess(detections, images_list.image_sizes, original_image_sizes)
            
            # CRITICAL MEMORY LEAK FIX: Clear the cached scores
            self.detector.rpn.rpn_scores = []
            
            return detections