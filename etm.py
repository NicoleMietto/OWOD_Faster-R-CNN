import torch
import torch.nn as nn
import torchvision.ops as ops

class EmbeddingTransferModule(nn.Module):
    """
    Modulo ETM (Embedding Transfer Module) come descritto nel paper OW-Rep.
    Trasferisce la conoscenza spaziale/semantica da DINOv2 agli instance embeddings.
    """
    def __init__(self, sigma=1.0, margin_delta=1.0, dino_patch_size=14):
        super().__init__()
        # Iperparametri dal paper (Algorithm 1) per la loss dell'ETM
        self.sigma = sigma
        self.delta = margin_delta
        # DINOv2 ViT riduce la risoluzione spaziale. ViT-Small/Base usano patch 14x14.
        # Questo ci serve per calcolare la scala del RoI-Align.
        self.spatial_scale = 1.0 / dino_patch_size

    def forward(self, dino_features, boxes, instance_embeddings):
        """
        Calcola la Relaxed Contrastive Loss (L_et).
        
        Args:
            dino_features (Tensor): Feature map di DINOv2 di shape [B, C, H, W].
            boxes (List[Tensor]): Lista contenente i tensori [N_i, 4] dei box (Known + Top-K Unknown) per ogni immagine.
            instance_embeddings (Tensor): I tuoi embeddings Z_i di shape [N_tot, d] usciti dalla CNN/MLP spaziale.
            
        Returns:
            Tensor: Valore scalare della loss L_et.
        """
        # --- Step 1: Estrazione dei Source Embeddings (S_i) da DINOv2 ---
        # Usiamo RoI-Align per estrarre la regione esatta dalla feature map di DINO
        # Assumiamo un output 7x7 per allinearci con la RoI Head standard
        aligned_dino_features = ops.roi_align(
            dino_features, 
            boxes, 
            output_size=(7, 7), 
            spatial_scale=self.spatial_scale
        )
        
        # Il paper applica un Global Average Pooling per ottenere il Source Embedding finale
        # Passiamo da [N, C, 7, 7] a [N, C]
        source_embeddings = aligned_dino_features.mean(dim=[2, 3]) 

        # --- Step 2: Calcolo della Similarità Semantica (W_ij) ---
        # Calcoliamo la distanza Euclidea a coppie (Matrice N x N)
        # torch.cdist è altamente ottimizzato in C++ per operazioni vettoriali
        dist_S = torch.cdist(source_embeddings, source_embeddings, p=2.0)
        
        # Applichiamo il kernel Gaussiano per ottenere i pesi di similarità (W_ij)
        W = torch.exp(-(dist_S ** 2) / self.sigma)
                                        
        # --- Step 3: Calcolo della Distanza tra le Instance Embeddings (d_ij) ---
        dist_Z = torch.cdist(instance_embeddings, instance_embeddings, p=2.0)

        # --- Step 4: Calcolo della Relaxed Contrastive Loss (L_et) ---
        # Componente di Attrazione: spinge vicini gli Z_i se DINO dice che sono simili
        pull_loss = W * (dist_Z ** 2)
        
        # Componente di Repulsione: allontana gli Z_i se DINO dice che sono diversi (fino al margine delta)
        # Usiamo torch.relu per la formula [delta - d_ij]_+ (prende solo i valori > 0)
        push_loss = (1 - W) * (torch.relu(self.delta - dist_Z) ** 2)
        
        # La loss totale è la media su tutte le coppie NxN
        L_et = (pull_loss + push_loss).mean()

        return L_et