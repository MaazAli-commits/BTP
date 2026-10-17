import random
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance

class VisualCorruptor:
    def __init__(self, corruption_ratio=0.5, blur_radius=3.0, darken_factor=0.3):
        """
        corruption_ratio: Fraction of frames in a tracklet to corrupt.
        blur_radius: Severity of Gaussian Blur.
        darken_factor: Multiplier for brightness (lower = darker).
        """
        self.corruption_ratio = corruption_ratio
        self.blur_radius = blur_radius
        self.darken_factor = darken_factor
        
    def __call__(self, frames):
        """
        frames: list of PIL Images
        Returns: list of PIL Images with random subset corrupted.
        """
        if self.corruption_ratio <= 0:
            return frames
            
        num_frames = len(frames)
        num_corrupt = int(num_frames * self.corruption_ratio)
        
        # Randomly select which frames to corrupt
        corrupt_indices = random.sample(range(num_frames), num_corrupt)
        
        corrupted_frames = []
        for i, frame in enumerate(frames):
            if i in corrupt_indices:
                # Randomly choose between blur and dark (or both)
                choice = random.random()
                img = frame.copy()
                if choice < 0.5:
                    # Apply blur
                    img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
                else:
                    # Apply darkness
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(self.darken_factor)
                corrupted_frames.append(img)
            else:
                corrupted_frames.append(frame)
                
        return corrupted_frames

class TextCorruptor:
    def __init__(self, drop_prob=0.5):
        """
        drop_prob: Probability of dropping a specific attribute category 
                   to simulate an incomplete witness statement.
        """
        self.drop_prob = drop_prob
        
    def __call__(self, attributes_dict):
        """
        attributes_dict: Dictionary of attributes e.g., {'gender': 'male', 'top_color': 'red'}
        Returns: A subset dictionary.
        """
        if self.drop_prob <= 0:
            return attributes_dict
            
        corrupted_dict = {}
        for k, v in attributes_dict.items():
            # Keep attribute if random > drop_prob
            if random.random() > self.drop_prob:
                corrupted_dict[k] = v
                
        return corrupted_dict
        
def generate_robustness_query(attributes):
    """
    Generates a natural language sentence from whatever attributes are left.
    Handles incomplete dictionaries gracefully.
    """
    parts = []
    
    gender = attributes.get('gender', None)
    if gender:
        parts.append(f"A {gender}")
    else:
        parts.append("A person")
        
    clothing = []
    top = attributes.get('top_color', None)
    if top:
        clothing.append(f"a {top} top")
        
    bottom = attributes.get('bottom_color', None)
    if bottom:
        clothing.append(f"{bottom} bottom")
        
    if clothing:
        parts.append("wearing " + " and ".join(clothing) + ".")
    else:
        parts.append("whose clothing is unknown.")
        
    return " ".join(parts)
