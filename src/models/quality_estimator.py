import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class FrameQualityEstimator(nn.Module):
    """
    Estimates frame quality q_j based on sharpness and illumination.
    Outputs q in [0, 1].
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel for sharpness/blur detection
        self.register_buffer('laplacian_kernel', torch.tensor([
            [[[0.0,  1.0, 0.0],
              [1.0, -4.0, 1.0],
              [0.0,  1.0, 0.0]]]
        ]))
        
    def forward(self, frames):
        """
        frames: (B, T, C, H, W) or (B*T, C, H, W)
        Returns: q_j of shape (B, T) or (B*T,)
        """
        is_5d = frames.dim() == 5
        if is_5d:
            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W)
            
        # Convert to grayscale
        gray = 0.2989 * frames[:, 0:1] + 0.5870 * frames[:, 1:2] + 0.1140 * frames[:, 2:3]
        
        # 1. Illumination (mean brightness on 0-1 scale)
        illum = gray.mean(dim=[1, 2, 3])
        illum_penalty = 1.0 - 4.0 * (illum - 0.5)**2
        
        # 2. Sharpness (Variance of Laplacian)
        # Scale up to 0-255 scale for standard variance magnitudes
        gray_255 = gray * 255.0
        lap = nn.functional.conv2d(gray_255, self.laplacian_kernel, padding=1)
        
        # Crop 10 pixels from borders to avoid padding/blur edge artifacts dominating variance
        lap = lap[:, :, 10:-10, 10:-10]
        sharpness = lap.var(dim=[1, 2, 3])
        
        # Calibrated Sigmoid centered at variance 50, scale 20
        # Blurry frames (~1.0 variance) -> ~0.08 score
        # Clear frames (~100.0 variance) -> ~0.92 score
        sharpness_score = torch.sigmoid((sharpness - 50.0) / 20.0)
        
        # Combine: heavily weight sharpness, use illumination as minor penalty
        q_j = 0.8 * sharpness_score + 0.2 * illum_penalty
        
        if is_5d:
            q_j = q_j.view(B, T)
            
        return q_j
