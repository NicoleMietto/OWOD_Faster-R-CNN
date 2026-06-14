import torch
import torch.nn as nn
import torchvision.ops as ops

class EmbeddingTransferModule(nn.Module):
    """
    ETM Module (Embedding Transfer Module) as described in the OW-Rep paper.
    Transfers spatial/semantic knowledge from DINOv2 to the instance embeddings.
    """
    def __init__(self, sigma=1.0, margin_delta=1.0, dino_patch_size=14):
        super().__init__()
        # Hyperparameters from the paper (Algorithm 1) for the ETM loss
        self.sigma = sigma
        self.delta = margin_delta
        # DINOv2 ViT reduces spatial resolution. ViT-Small/Base use 14x14 patches.
        # This is needed to calculate the RoI-Align spatial scale.
        self.spatial_scale = 1.0 / dino_patch_size

    def forward(self, dino_features, boxes, instance_embeddings):
        """
        Computes the Relaxed Contrastive Loss (L_et).
        
        Args:
            dino_features (Tensor): DINOv2 feature map of shape [B, C, H, W].
            boxes (List[Tensor]): List containing tensors [N_i, 4] of boxes (Known + Top-K Unknown) for each image.
            instance_embeddings (Tensor): Your Z_i embeddings of shape [N_tot, d] outputted from the spatial CNN/MLP.
            
        Returns:
            Tensor: Scalar value of the L_et loss.
        """
        N = len(instance_embeddings)
        if N == 0:
            return torch.tensor(0.0, device=dino_features.device, requires_grad=True)

        # --- Step 1: Extraction of Source Embeddings (S_i) from DINOv2 ---
        # We use RoI-Align to extract the exact region from the DINO feature map
        aligned_dino_features = ops.roi_align(
            dino_features, 
            boxes, 
            output_size=(7, 7), 
            spatial_scale=self.spatial_scale
        )
        
        # Pass from [N, C, 7, 7] to [N, C] via Global Average Pooling
        source_embeddings = aligned_dino_features.mean(dim=[2, 3]) 

        # --- Step 2: Semantic Similarity Calculation (W_ij) ---
        dist_S = torch.cdist(source_embeddings, source_embeddings, p=2.0)
        W = torch.exp(-(dist_S ** 2) / self.sigma)
                                        
        # --- Step 3: Distance Calculation between Instance Embeddings (d_ij) ---
        dist_Z = torch.cdist(instance_embeddings, instance_embeddings, p=2.0)

        # --- Step 4: Relaxed Contrastive Loss (L_et) Calculation ---
        pull_loss = W * (dist_Z ** 2)
        
        # The push_loss uses a ReLU. It activates ONLY for items closer than delta.
        push_loss = (1 - W) * (torch.relu(self.delta - dist_Z) ** 2)
        
        # PAPER FIDELITY (CVPR 2021/2024): 
        # The total sum of the matrix is O(N) due to the sparsity of positive matches 
        # and hard negatives (filtered by the ReLU).
        # Therefore, the correct normalization to maintain a constant O(1) loss is dividing by N, not N^2.
        matrix_sum = (pull_loss + push_loss).sum()
        L_et = matrix_sum / N

        return L_et