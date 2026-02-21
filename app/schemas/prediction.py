"""
Pydantic schemas for API responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Union

class TopPrediction(BaseModel):
    """Single prediction result"""
    class_name: str = Field(..., alias="class")
    confidence: float
    
    class Config:
        populate_by_name = True

class NutritionalInfo(BaseModel):
    """Nutritional information"""
    total_calories: float
    protein: float
    carbohydrates: float
    fat: float
    fiber: float

class PredictionResponse(BaseModel):
    """Complete prediction response"""
    food_name: str
    confidence: float
    calories: float
    portion_size: int
    nutritional_info: Dict[str, float]
    top_predictions: List[Dict[str, Union[str, float]]]  # Allow both string and float values
    
    class Config:
        json_schema_extra = {
            "example": {
                "food_name": "Pizza",
                "confidence": 0.9523,
                "calories": 532.0,
                "portion_size": 200,
                "nutritional_info": {
                    "total_calories": 532.0,
                    "protein": 22.0,
                    "carbohydrates": 66.0,
                    "fat": 20.0,
                    "fiber": 4.6
                },
                "top_predictions": [
                    {"class": "pizza", "confidence": 0.9523},
                    {"class": "lasagna", "confidence": 0.0234},
                    {"class": "spaghetti_bolognese", "confidence": 0.0156}
                ]
            }
        }
