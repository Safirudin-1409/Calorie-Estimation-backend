"""
Image preprocessing service using OpenCV and PIL
"""
import numpy as np
import cv2
from PIL import Image
from typing import Union

from app.config import settings

class ImageProcessor:
    """Image preprocessing for model input"""
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image array
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to model input size
        image = cv2.resize(
            image,
            (settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate image format and properties
        
        Args:
            image: PIL Image
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if image can be converted to RGB
            image.convert('RGB')
            
            # Check minimum size
            if image.width < 50 or image.height < 50:
                return False
            
            return True
            
        except Exception:
            return False
