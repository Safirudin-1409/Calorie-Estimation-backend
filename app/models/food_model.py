"""
Food recognition model using TensorFlow
"""
import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

class FoodModel:
    """Food recognition model wrapper"""
    
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.class_names: List[str] = []
        
    def load_model(self):
        """Load the trained model or create a demo model"""
        try:
            # Load class names
            self._load_class_names()
            
            # Try to load existing model
            if os.path.exists(settings.MODEL_PATH):
                logger.info(f"Loading model from {settings.MODEL_PATH}")
                self.model = tf.keras.models.load_model(settings.MODEL_PATH)
                logger.info("Model loaded successfully!")
            else:
                logger.warning(f"Model file not found at {settings.MODEL_PATH}")
                logger.info("Creating demo model with MobileNetV2...")
                self.model = self._create_demo_model()
                logger.info("Demo model created successfully!")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Creating demo model as fallback...")
            self.model = self._create_demo_model()
    
    def _load_class_names(self):
        """Load class names from JSON file"""
        try:
            # __file__ is app/models/food_model.py
            # go up 2 levels: models/ -> app/ -> backend/ (project root)
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            class_names_path = os.path.join(backend_dir, settings.CLASS_NAMES_PATH)
            
            if os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    data = json.load(f)
                    self.class_names = data.get("classes", [])
                logger.info(f"Loaded {len(self.class_names)} class names from {class_names_path}")
            else:
                logger.warning(f"Class names file not found at {class_names_path}")
                self.class_names = self._get_default_class_names()
                
        except Exception as e:
            logger.error(f"Error loading class names: {str(e)}")
            self.class_names = self._get_default_class_names()
    
    def _get_default_class_names(self) -> List[str]:
        """Get default Food-101 class names"""
        return [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
            "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
            "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
            "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
            "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
            "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
            "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
            "mussels", "nachos", "omelette", "onion_rings", "oysters",
            "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
            "pho", "pizza", "pork_chop", "poutine", "prime_rib",
            "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
            "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
            "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
            "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
        ]
    
    def _create_demo_model(self) -> tf.keras.Model:
        """Create a demo model using MobileNetV2"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Add classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Initialize with random weights for the classification head
        model.build((None, settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3))
        
        logger.info("Demo model architecture created")
        return model
    
    def predict(self, image: np.ndarray) -> List[Dict[str, float]]:
        """
        Predict food class from preprocessed image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            List of predictions with class names and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            if confidence >= settings.CONFIDENCE_THRESHOLD:
                results.append({
                    "class": self.class_names[idx],
                    "confidence": confidence
                })
        
        # If no predictions meet threshold, return top prediction anyway
        if not results:
            top_idx = top_indices[0]
            results.append({
                "class": self.class_names[top_idx],
                "confidence": float(predictions[0][top_idx])
            })
        
        return results
