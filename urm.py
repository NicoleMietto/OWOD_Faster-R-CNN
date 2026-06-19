import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
# MobileSAM must be installed in the environment (e.g. Colab/Kaggle) with:
# !pip install git+https://github.com/ChaoningZhang/MobileSAM.git
from mobile_sam import sam_model_registry, SamPredictor
class UnknownBoxRefineModule(nn.Module):
    """
    URM Module: Refines unknown bounding boxes using MobileSAM.
    Discards hallucinations and computes refinement loss to train the RPN.
    """
    def __init__(self, sam_checkpoint_path, device='cuda', iou_filter_thresh=0.5):
        super().__init__()
        self.device = device
        self.iou_filter_thresh = iou_filter_thresh
        
        # 1. Load MobileSAM (ViT-Tiny to save VRAM on Kaggle)
        model_type = "vit_t"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=self.device)
        self.sam.eval() # SAM acts as a "teacher", it is always in eval mode
        
        # Explicitly freeze all weights to save memory on gradients
        for param in self.sam.parameters():
            param.requires_grad = False
            
    def forward(self, top_unknown_boxes, raw_image_tensor):
        """
        Args:
            top_unknown_boxes (Tensor): [K, 4] bounding boxes coming from RPN (gradients are active).
            raw_image_tensor (Tensor): [3, H, W] the original image (RGB, range 0-255).
            
        Returns:
            valid_sam_boxes (Tensor): [M, 4] The perfect boxes (M <= K), used for ETM and RoI.
            loss_b_unk (Tensor): Scalar value of the refinement loss.
        """
        current_device = top_unknown_boxes.device
        # If RPN did not find any unknown, do nothing
        if len(top_unknown_boxes) == 0:
            return torch.empty((0, 4), device=current_device), torch.tensor(0.0, device=current_device, requires_grad=True)
            
        # MUST be instantiated inside forward so DataParallel uses the correct GPU replica.
        predictor = SamPredictor(self.sam)
        # --- STEP 1: De-normalization and Preparation for SAM ---
        # torchvision uses this mean and std by default
        mean = torch.tensor([0.485, 0.456, 0.406], device=raw_image_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=raw_image_tensor.device).view(3, 1, 1)
        
        # Reverse the normalization formula: (img * std) + mean
        unnorm_img = (raw_image_tensor * std) + mean
        
        original_h, original_w = unnorm_img.shape[1], unnorm_img.shape[2]
        
        # Pure GPU Pipeline: Resize image directly on GPU instead of CPU
        target_length = predictor.transform.target_length
        new_size = predictor.transform.get_preprocess_shape(original_h, original_w, target_length)
        
        unnorm_img_255 = (unnorm_img.clamp(0, 1) * 255.0).unsqueeze(0)
        transformed_img = F.interpolate(unnorm_img_255, size=new_size, mode='bilinear', align_corners=False)
        
        # Pass the pre-processed tensor directly to SAM
        predictor.set_torch_image(transformed_img, (original_h, original_w))
        
        # --- STEP 2: Batched Prompting (Pass all Top-K boxes together) ---
        # Scale bounding boxes natively on GPU
        ratio_h = new_size[0] / original_h
        ratio_w = new_size[1] / original_w
        
        input_boxes_torch = top_unknown_boxes.clone()
        input_boxes_torch[:, [0, 2]] *= ratio_w
        input_boxes_torch[:, [1, 3]] *= ratio_h
        
        # Inference without gradients
        with torch.no_grad():
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=input_boxes_torch,
                multimask_output=False, # We only want the "best" mask for each box
            )
        # 'masks' has dimension [K, 1, H, W]
        
        # Free SAM internal cache immediately after prediction to prevent RAM leaks
        predictor.reset_image()
        
        # --- STEP 3: Generation of New Bounding Boxes ---
        # A mask might be completely empty (SAM didn't find anything).
        # masks_to_boxes crashes on empty masks. We must filter them first.
        mask_bool = masks[:, 0] > 0.0
        valid_mask_idx = mask_bool.view(mask_bool.shape[0], -1).any(dim=1)
        
        # Ensure the boolean mask is on the correct GPU before indexing
        valid_mask_idx = valid_mask_idx.to(current_device)
        
        if not valid_mask_idx.any():
            return torch.empty((0, 4), device=current_device), torch.tensor(0.0, device=current_device, requires_grad=True)
            
        valid_masks = mask_bool[valid_mask_idx]
        sam_boxes = ops.masks_to_boxes(valid_masks)
        top_unknown_boxes_filtered = top_unknown_boxes[valid_mask_idx]
        
        # --- STEP 4: Hallucination Filtering ---
        # Calculate the IoU between original RPN proposals and perfect SAM masks.
        # ops.box_iou returns a KxK matrix. We are only interested in the diagonal
        # (i.e. the match between RPN box N and SAM box N).
        ious = ops.box_iou(top_unknown_boxes_filtered, sam_boxes).diag()
        
        # Keep only those where SAM confirmed RPN's intuition
        keep_mask = ious >= self.iou_filter_thresh
        
        valid_proposals = top_unknown_boxes_filtered[keep_mask]
        valid_sam_boxes = sam_boxes[keep_mask].detach() # These are now "Ground Truth", detach() is applied!
        
        if len(valid_sam_boxes) == 0:
             return torch.empty((0, 4), device=current_device), torch.tensor(0.0, device=current_device, requires_grad=True)
        # --- STEP 5: Loss Calculation (L_b,unk) ---
        # This loss trains the RPN (valid_proposals has active gradients) to generate
        # boxes that are more similar to the perfect SAM boxes right from the start.
        
        # 5a. L1 Loss (Absolute distance between coordinates)
        loss_l1 = F.l1_loss(valid_proposals, valid_sam_boxes, reduction='mean')
        
        # 5b. GIoU Loss (Generalised IoU)
        giou_matrix = ops.generalized_box_iou(valid_proposals, valid_sam_boxes)
        loss_giou = 1.0 - giou_matrix.diag().mean()
        
        # Combined total loss (as in the original paper)
        loss_b_unk = loss_l1 + loss_giou 
        
        return valid_sam_boxes, loss_b_unk