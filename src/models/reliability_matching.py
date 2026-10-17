import torch
import torch.nn as nn
import torch.nn.functional as F

class ReliabilityAwareMatching(nn.Module):
    def __init__(self, mode='post_softmax', eps=1e-6):
        """
        mode: 'post_softmax' or 'logit_space'
        """
        super().__init__()
        self.mode = mode
        self.eps = eps
        
    def forward(self, text_seq, video_seq, c, q, text_mask=None, apply_weights=True):
        """
        text_seq: (B_q, L_t, D)
        video_seq: (B_g, T, D)
        c: (B_q, L_t) - Query confidence
        q: (B_g, T) - Frame quality
        text_mask: (B_q, L_t) - 1 for valid, 0 for padding
        apply_weights: bool - whether to apply c_i and q_j reweighting
        
        Returns:
        sim_matrix: (B_q, B_g)
        """
        B_q, L_t, D = text_seq.shape
        B_g, T, D = video_seq.shape
        
        # Normalize features
        text_seq = F.normalize(text_seq, dim=-1)
        video_seq = F.normalize(video_seq, dim=-1)
        
        # Compute raw logits l_ij: (B_q, B_g, L_t, T)
        # text_seq: (B_q, 1, L_t, D)
        # video_seq: (1, B_g, T, D)
        l_ij = torch.einsum('qid,gjd->qgij', text_seq, video_seq)
        
        if not apply_weights:
            alpha = F.softmax(l_ij, dim=-1)
            score_ij = alpha * l_ij
            if text_mask is not None:
                score_ij = score_ij * text_mask.unsqueeze(1).unsqueeze(-1)
            return score_ij.sum(dim=(2, 3))
        
        # Expand c and q for broadcast
        # c: (B_q, 1, L_t, 1)
        c_expanded = c.unsqueeze(1).unsqueeze(-1)
        # q: (1, B_g, 1, T)
        q_expanded = q.unsqueeze(0).unsqueeze(2)
        
        if self.mode == 'post_softmax':
            # α_ij = softmax(l_ij) over j (video tokens)
            alpha = F.softmax(l_ij, dim=-1)
            
            # α'_ij = α_ij * q_j (c_i cancels out in re-normalization over j)
            alpha_prime = alpha * q_expanded
            
            # Re-normalize over j
            alpha_prime = alpha_prime / (alpha_prime.sum(dim=-1, keepdim=True) + self.eps)
            
        elif self.mode == 'logit_space':
            # l'_ij = l_ij + log(q_j + eps)
            l_prime = l_ij + torch.log(q_expanded + self.eps)
            
            # α'_ij = softmax(l'_ij) over j
            alpha_prime = F.softmax(l_prime, dim=-1)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # Compute final score by aggregating the weighted similarities
        # score = sum_i c_i * sum_j (α'_ij * l_ij)
        score_ij = alpha_prime * l_ij
        
        # Apply c_i as the outer token-level confidence weight
        score_ij = score_ij * c_expanded
        
        if text_mask is not None:
            # text_mask: (B_q, L_t) -> (B_q, 1, L_t, 1)
            mask_expanded = text_mask.unsqueeze(1).unsqueeze(-1)
            score_ij = score_ij * mask_expanded
            
        # Resulting score matrix: (B_q, B_g)
        score = score_ij.sum(dim=(2, 3))
        
        return score
