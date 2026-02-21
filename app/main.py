"""
FastAPI application for food recognition and calorie estimation.
"""
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io

from app.config import settings
from app.models.food_model import FoodModel
from app.services.image_processor import ImageProcessor
from app.services.nutrition_service import NutritionService
from app.services.calorie_estimator import CalorieEstimator
from app.schemas.prediction import PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Recognition API",
    description="AI-powered food recognition and calorie estimation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
food_model = FoodModel()
image_processor = ImageProcessor()
nutrition_service = NutritionService()
calorie_estimator = CalorieEstimator()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Food Recognition API...")
    food_model.load_model()
    logger.info("Model loaded successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Food Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/api/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": food_model.model is not None,
        "version": "1.0.0"
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_food(file: UploadFile = File(...)):
    """
    Predict food from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        PredictionResponse with food name, calories, and nutritional info
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, BMP)"
            )
        
        # Read image file
        contents = await file.read()
        
        # Validate file size
        if len(contents) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
            )
        
        # Open and validate image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Preprocess image
        processed_image = image_processor.preprocess(image)
        
        # Get prediction
        predictions = food_model.predict(processed_image)
        
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions"
            )
        
        # Get top prediction
        top_prediction = predictions[0]
        food_name = top_prediction["class"]
        confidence = top_prediction["confidence"]
        
        # Get nutritional information
        nutrition_info = nutrition_service.get_nutrition(food_name)
        
        # Estimate portion size and calories
        portion_size = calorie_estimator.estimate_portion_size(food_name)
        calories = calorie_estimator.calculate_calories(
            nutrition_info["calories_per_100g"],
            portion_size
        )
        
        # Calculate macronutrients based on portion size
        nutritional_info = {
            "total_calories": round(calories, 1),
            "protein": round(nutrition_info["protein_per_100g"] * portion_size / 100, 1),
            "carbohydrates": round(nutrition_info["carbs_per_100g"] * portion_size / 100, 1),
            "fat": round(nutrition_info["fat_per_100g"] * portion_size / 100, 1),
            "fiber": round(nutrition_info["fiber_per_100g"] * portion_size / 100, 1)
        }
        
        # Format response
        response = PredictionResponse(
            food_name=food_name.replace("_", " ").title(),
            confidence=confidence,
            calories=calories,
            portion_size=portion_size,
            nutritional_info=nutritional_info,
            top_predictions=predictions[:5]
        )
        
        logger.info(f"Prediction: {food_name} (confidence: {confidence:.2f})")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
