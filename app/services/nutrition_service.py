"""
Nutrition database service
"""
import os
import json
import logging
from typing import Dict

from app.config import settings

logger = logging.getLogger(__name__)

class NutritionService:
    """Service for retrieving nutritional information"""
    
    def __init__(self):
        self.nutrition_db: Dict = {}
        self._load_nutrition_db()
    
    def _load_nutrition_db(self):
        """Load nutrition database from JSON file"""
        try:
            # __file__ is app/services/nutrition_service.py
            # go up 2 levels: services/ -> app/ -> backend/ (project root)
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(backend_dir, settings.NUTRITION_DB_PATH)
            
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    self.nutrition_db = json.load(f)
                logger.info(f"Loaded nutrition data for {len(self.nutrition_db)} foods from {db_path}")
            else:
                logger.warning(f"Nutrition database not found at {db_path}")
                self.nutrition_db = self._get_default_nutrition_db()
                
        except Exception as e:
            logger.error(f"Error loading nutrition database: {str(e)}")
            self.nutrition_db = self._get_default_nutrition_db()
    
    def _get_default_nutrition_db(self) -> Dict:
        """Get default nutrition values (per 100g)"""
        return {
            "default": {
                "calories_per_100g": 200,
                "protein_per_100g": 8,
                "carbs_per_100g": 25,
                "fat_per_100g": 8,
                "fiber_per_100g": 2
            }
        }
    
    def get_nutrition(self, food_name: str) -> Dict[str, float]:
        """
        Get nutritional information for a food item
        
        Args:
            food_name: Name of the food
            
        Returns:
            Dictionary with nutritional values per 100g
        """
        # Normalize food name
        food_key = food_name.lower().replace(" ", "_")
        
        # Get nutrition data or use default
        if food_key in self.nutrition_db:
            return self.nutrition_db[food_key]
        else:
            logger.warning(f"No nutrition data found for {food_name}, using defaults")
            return self.nutrition_db.get("default", {
                "calories_per_100g": 200,
                "protein_per_100g": 8,
                "carbs_per_100g": 25,
                "fat_per_100g": 8,
                "fiber_per_100g": 2
            })
