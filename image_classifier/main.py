# 1. IMPORTS SECTION (at the top)
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from torchvision import transforms
from PIL import Image
import io
from transformers import pipeline  
import os

# 2. APP INITIALIZATION
app = FastAPI(title="Image Classification API")

# 3. STATIC FILES CONFIGURATION
app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. TEMPLATES CONFIGURATION
templates = Jinja2Templates(directory="templates")

# 5. MODEL LOADING (keep your existing code)
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Load image captioning model
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# 6. IMAGE PREPROCESSING (keep your existing preprocess_image function)

# Image preprocessing function
def preprocess_image(image_bytes):
    try:
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize(256),          # Resize to 256x256
            transforms.CenterCrop(224),      # Crop center 224x224
            transforms.ToTensor(),           # Convert to tensor
            transforms.Normalize(            # Normalize with ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # Open image and apply transformations
        image = Image.open(io.BytesIO(image_bytes))
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

# Home page with form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image!")

    try:
        # Read image
        image_bytes = await file.read()
        
        # Save the uploaded image
        temp_image_path = "static/uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        # Classification
        tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        
        # Image captioning
        description = captioner(temp_image_path)[0]['generated_text']
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "class_id": class_id,
                "description": description,  # New
                "image_uploaded": True
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get the uploaded image
@app.get("/uploaded-image")
async def get_uploaded_image():
    return FileResponse("static/uploaded_image.jpg")