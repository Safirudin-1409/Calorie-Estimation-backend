"""
Calorie estimation service
"""
from typing import Dict

class CalorieEstimator:
    """Service for estimating calories and portion sizes"""
    
    def __init__(self):
        # Standard portion sizes in grams
        self.portion_sizes: Dict[str, int] = {
            # Desserts and sweets
            "apple_pie": 125,
            "baklava": 70,
            "bread_pudding": 150,
            "cannoli": 85,
            "carrot_cake": 100,
            "cheesecake": 125,
            "chocolate_cake": 100,
            "chocolate_mousse": 100,
            "churros": 90,
            "creme_brulee": 120,
            "cup_cakes": 50,
            "donuts": 60,
            "frozen_yogurt": 150,
            "ice_cream": 100,
            "macarons": 40,
            "panna_cotta": 120,
            "red_velvet_cake": 100,
            "strawberry_shortcake": 120,
            "tiramisu": 100,
            "waffles": 75,
            
            # Main dishes
            "baby_back_ribs": 250,
            "beef_carpaccio": 100,
            "beef_tartare": 120,
            "bibimbap": 400,
            "breakfast_burrito": 300,
            "chicken_curry": 350,
            "chicken_quesadilla": 200,
            "chicken_wings": 150,
            "club_sandwich": 250,
            "crab_cakes": 100,
            "croque_madame": 200,
            "eggs_benedict": 250,
            "filet_mignon": 200,
            "fish_and_chips": 350,
            "foie_gras": 80,
            "grilled_cheese_sandwich": 150,
            "grilled_salmon": 150,
            "gyoza": 120,
            "hamburger": 250,
            "hot_dog": 150,
            "huevos_rancheros": 300,
            "lasagna": 300,
            "lobster_roll_sandwich": 200,
            "macaroni_and_cheese": 250,
            "omelette": 150,
            "pad_thai": 350,
            "paella": 350,
            "pancakes": 150,
            "peking_duck": 200,
            "pho": 400,
            "pizza": 200,
            "pork_chop": 200,
            "poutine": 300,
            "prime_rib": 250,
            "pulled_pork_sandwich": 250,
            "ramen": 400,
            "ravioli": 250,
            "risotto": 300,
            "spaghetti_bolognese": 350,
            "spaghetti_carbonara": 300,
            "steak": 250,
            "sushi": 200,
            "tacos": 150,
            "takoyaki": 120,
            
            # Appetizers and sides
            "beet_salad": 150,
            "beignets": 80,
            "bruschetta": 100,
            "caesar_salad": 200,
            "caprese_salad": 150,
            "ceviche": 120,
            "cheese_plate": 100,
            "clam_chowder": 250,
            "deviled_eggs": 60,
            "dumplings": 120,
            "edamame": 100,
            "escargots": 100,
            "falafel": 100,
            "french_fries": 150,
            "french_onion_soup": 300,
            "french_toast": 150,
            "fried_calamari": 150,
            "fried_rice": 250,
            "garlic_bread": 80,
            "gnocchi": 200,
            "greek_salad": 200,
            "guacamole": 100,
            "hot_and_sour_soup": 250,
            "hummus": 100,
            "lobster_bisque": 250,
            "miso_soup": 200,
            "mussels": 200,
            "nachos": 200,
            "onion_rings": 120,
            "oysters": 100,
            "samosa": 80,
            "sashimi": 120,
            "scallops": 120,
            "seaweed_salad": 100,
            "shrimp_and_grits": 300,
            "spring_rolls": 100,
            "tuna_tartare": 120
        }
    
    def estimate_portion_size(self, food_name: str) -> int:
        """
        Estimate portion size for a food item
        
        Args:
            food_name: Name of the food
            
        Returns:
            Estimated portion size in grams
        """
        # Normalize food name
        food_key = food_name.lower().replace(" ", "_")
        
        # Return portion size or default
        return self.portion_sizes.get(food_key, 200)
    
    def calculate_calories(self, calories_per_100g: float, portion_size: int) -> float:
        """
        Calculate total calories based on portion size
        
        Args:
            calories_per_100g: Calories per 100g
            portion_size: Portion size in grams
            
        Returns:
            Total calories
        """
        return (calories_per_100g * portion_size) / 100
