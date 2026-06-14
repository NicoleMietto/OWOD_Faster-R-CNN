import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align

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
        # total num_classes = background (0) + known_classes + 1 (dummy class for unknowns)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_known_classes + 2)

        self.rpn_scores = [] # We will save intercepted scores here
        
        # Save the original PyTorch function
        original_filter = self.detector.rpn.filter_proposals
        
        # Create a "Hook"
        def hook_filter_proposals(*args, **kwargs):
            # Call original function. Returns boxes and scores!
            boxes, scores = original_filter(*args, **kwargs)
            self.rpn_scores = scores # Steal them and save them here
            return boxes, scores     # Return boxes to PyTorch as expected
            
        # Replace PyTorch's function with ours!
        self.detector.rpn.filter_proposals = hook_filter_proposals
        
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
        if self.training:
            # --- PHASE 1: Feature Extraction & RPN ---
            images_list, targets_list = self.detector.transform(images, targets)
            features = self.detector.backbone(images_list.tensors)
            
            # The RPN generates raw proposals (mild selection, generates thousands)
            proposals, proposal_losses = self.detector.rpn(images_list, features, targets_list)
            
            # 2. HERE IS THE MAGIC: Retrieve the scores just stolen by the hook!
            objectness_scores = self.rpn_scores
            
            device = images_list.tensors.device
            total_loss_b_unk = torch.tensor(0.0, device=device)
            total_loss_et = torch.tensor(0.0, device=device)
            augmented_targets = []
            
            # Iterate for each image in the batch
            for i in range(len(images)):
                # 2. Extract DINO feature map for this specific image
                dino_features_i = dino_features_list[i] if dino_features_list is not None else None

                # --- Pseudo-Labeling ---
                labels_dict = self.labeler.assign_labels_known_unknown_background(
                    proposals[i], 
                    objectness_scores[i], # This works perfectly now!
                    targets_list[i]['boxes'], 
                    targets_list[i]['labels']
                )
                
                # Raw (imprecise) PREDICTIONS for knowns! 
                pred_known_boxes = labels_dict['known_boxes'] 
                
                # --- NMS to not waste Top-K ---
                raw_unknowns = labels_dict['unknown_boxes']
                unk_scores = labels_dict['unknown_scores'] # You will also need to return scores from the labeler
                
                # Apply NMS among unknowns. If they overlap too much, keep only the best.
                keep_idx = torchvision.ops.nms(raw_unknowns, unk_scores, iou_threshold=0.3)
                filtered_unknowns = raw_unknowns[keep_idx]
                
                # NOW we take the Top-K without waste
                top_k = 10
                top_unknowns = filtered_unknowns[:top_k]
                
                # --- PHASE 3: URM (MobileSAM) ---
                if self.use_urm:
                    refined_unknowns, loss_b_unk = self.urm(top_unknowns, images_list.tensors[i])
                    total_loss_b_unk = total_loss_b_unk + loss_b_unk
                else:
                    refined_unknowns = torch.empty((0, 4), device=device)
                
                # --- PHASE 4: ETM (DINOv2) ---
                valid_boxes_for_etm = torch.cat([pred_known_boxes, refined_unknowns])
                
                if self.use_etm and len(valid_boxes_for_etm) > 0 and dino_features_i is not None:
                    # In FPN, features is an OrderedDict. We use the highest resolution feature map '0'
                    feat_tensor = features['0'] if isinstance(features, dict) else features
                    roi_features = roi_align(feat_tensor, [valid_boxes_for_etm], output_size=(7, 7), spatial_scale=1/4.0)
                    instance_embeddings = self.embedding_head(roi_features)
                    
                    loss_et = self.etm(dino_features_i, [valid_boxes_for_etm], instance_embeddings)
                    total_loss_et = total_loss_et + loss_et
                
                # --- PHASE 5: Target Injection for Mild RoI Head ---
                # Add SAM-refined boxes to the image's Ground Truths.
                # The label will be num_known_classes + 1 (the dummy Unknown class)
                new_target = targets_list[i].copy()
                if len(refined_unknowns) > 0:
                    new_labels = torch.full((len(refined_unknowns),), self.num_known_classes + 1, dtype=torch.int64, device=refined_unknowns.device)
                    new_target['boxes'] = torch.cat([new_target['boxes'], refined_unknowns])
                    new_target['labels'] = torch.cat([new_target['labels'], new_labels])
                augmented_targets.append(new_target)

            # --- PHASE 6: Standard RoI Head (Mild Detection) ---
            # We feed ALL RPN proposals and augmented targets to the base network.
            # It will do mild matching (IoU > 0.5 for positives) and calculate classification and regression losses.
            # roi_heads returns (predictions, dict_of_losses)
            detections, detector_losses = self.detector.roi_heads(features, proposals, images_list.image_sizes, augmented_targets)
            
            # Combine all losses
            total_losses = {}
            total_losses.update(proposal_losses)
            total_losses.update(detector_losses)
            total_losses['loss_b_unk'] = (total_loss_b_unk / len(images)) * self.alpha
            total_losses['loss_et'] = (total_loss_et / len(images)) * self.beta
            
            return total_losses
            
        else:
            return self.detector(images)