import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F

# MobileSAM must be installed in the environment (e.g. Colab/Kaggle) with:
# !pip install git+https://github.com/ChaoningZhang/MobileSAM.git
from mobile_sam import sam_model_registry

class UnknownBoxRefineModule(nn.Module):
    """
    URM Module: Refines unknown bounding boxes using MobileSAM.
    Discards hallucinations and computes refinement loss to train the RPN.
    """
    def __init__(self, sam_checkpoint_path, device='cuda', iou_filter_thresh=0.5):
        super().__init__()
        self.sam_checkpoint_path = sam_checkpoint_path
        self.iou_filter_thresh = iou_filter_thresh

        # Inizializziamo SAM DIRETTAMENTE in self.
        # Questo permette a DataParallel di replicarlo correttamente senza rompere il context C++ di PyTorch!
        model_type = "vit_t"
        self.sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint_path)
        self.sam.to(device=device)
        self.sam.eval() # SAM acts as a "teacher", it is always in eval mode
        for param in self.sam.parameters():
            param.requires_grad = False

        # Register normalization constants as buffers to prevent CPU memory leaks in DataParallel threads
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        self.register_buffer('pixel_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))


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
            return torch.empty((0, 4), device=current_device), torch.tensor(0.0, device=current_device)

        # Dynamically load/fetch the SAM model for the current GPU
        sam = self._load_sam(current_device)
            
        # --- STATELESS SAM INFERENCE (Prevents CPU/GPU Memory Leaks) ---
        # 1. Pure GPU Pipeline: Resize image directly on GPU
        target_length = sam.image_encoder.img_size # usually 1024
        
        # Reverse the normalization formula: (img * std) + mean
        mean = self.img_mean.to(current_device)
        std = self.img_std.to(current_device)
        unnorm_img = (raw_image_tensor * std) + mean
        
        original_h, original_w = unnorm_img.shape[1], unnorm_img.shape[2]
        scale = target_length / max(original_h, original_w)
        new_size = (int(original_h * scale + 0.5), int(original_w * scale + 0.5))
        
        unnorm_img_255 = (unnorm_img.clamp(0, 1) * 255.0).unsqueeze(0)
        transformed_img = F.interpolate(unnorm_img_255, size=new_size, mode='bilinear', align_corners=False)
        
        # 2. SAM Preprocess (Normalize and Pad)
        pixel_mean = self.pixel_mean.to(current_device)
        pixel_std = self.pixel_std.to(current_device)
        x = (transformed_img - pixel_mean) / pixel_std
        
        padh = target_length - x.shape[2]
        padw = target_length - x.shape[3]
        input_image = F.pad(x, (0, padw, 0, padh))
        
        # 3. Batched Prompting
        ratio_h = new_size[0] / original_h
        ratio_w = new_size[1] / original_w
        
        # CRITICAL: Detach the boxes before feeding to SAM to prevent graph retention!
        input_boxes_torch = top_unknown_boxes.detach().clone()
        input_boxes_torch[:, [0, 2]] *= ratio_w
        input_boxes_torch[:, [1, 3]] *= ratio_h
        
        # Inference without gradients
        with torch.no_grad():
            features = self.sam.image_encoder(input_image)
            
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=input_boxes_torch,
                masks=None,
            )
            
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=features,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Postprocess Masks
            masks = F.interpolate(low_res_masks, size=(target_length, target_length), mode="bilinear", align_corners=False)
            masks = masks[..., : new_size[0], : new_size[1]]
            masks = F.interpolate(masks, size=(original_h, original_w), mode="bilinear", align_corners=False)
            
            # 'masks' has dimension [K, 1, H, W]
        
        # --- STEP 3: Generation of New Bounding Boxes ---
        # CRITICAL FIX: Convert float logits to boolean mask! 
        # masks_to_boxes expects boolean. If float is passed, it sees all non-zero floats as True, returning the whole image!
        mask_bool = masks[:, 0] > 0.0
        valid_mask_idx = mask_bool.view(mask_bool.shape[0], -1).any(dim=1)
        
        # Ensure the boolean mask is on the correct GPU before indexing
        valid_mask_idx = valid_mask_idx.to(current_device)
        
        if not valid_mask_idx.any():
            return torch.empty((0, 4), device=current_device), torch.tensor(0.0, device=current_device)
            
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
             return torch.empty((0, 4), device=current_device), torch.tensor(0.0, device=current_device)

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

        # Immediate cleanup of intermediate tensors to prevent CPU/GPU RAM accumulation
        del features, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, masks
        del input_image, transformed_img, unnorm_img_255, unnorm_img, x
        
        return valid_sam_boxes, loss_b_unk
