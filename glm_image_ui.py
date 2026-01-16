import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import io
import base64
import uvicorn
from typing import List
from contextlib import asynccontextmanager

# Global variable to store the pipeline
pipe = None

def load_model():
    """Load the GLM-Image model"""
    global pipe
    print("Loading GLM-Image model...")
    pipe = GlmImagePipeline.from_pretrained(
        "zai-org/GLM-Image",
        device_map="cuda"
    )
    print("Model loaded successfully!")
    return pipe

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup: Load model
    load_model()
    yield
    # Shutdown: Cleanup (if needed)
    # Currently no cleanup needed

app = FastAPI(title="GLM-Image Web UI", lifespan=lifespan)

# Pydantic models for request validation
class TextToImageRequest(BaseModel):
    prompt: str
    height: int = 32 * 32
    width: int = 36 * 32
    num_inference_steps: int = 50
    guidance_scale: float = 1.5
    seed: int = 42


def text_to_image(
    prompt: str,
    height: int = 32 * 32,
    width: int = 36 * 32,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: int = 42
):
    """Generate image from text prompt"""
    global pipe
    try:
        # Validate inputs
        if not prompt or not prompt.strip():
            raise ValueError("Please enter a prompt")
        
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait for the model to finish loading.")
        
        # Set up generator with seed
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
        
        # Generate image
        print(f"Generating image with prompt: {prompt}")
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        return image
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def image_to_image(
    images: List[Image.Image],
    prompt: str,
    height: int = 33 * 32,
    width: int = 32 * 32,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: int = 42
):
    """Generate image from input images and prompt"""
    global pipe
    try:
        # Validate inputs
        if images is None or len(images) == 0:
            raise ValueError("Please upload at least one image")
        
        if not prompt or not prompt.strip():
            raise ValueError("Please enter a prompt")
        
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait for the model to finish loading.")
        
        # Set up generator with seed
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
        
        # Generate image
        print(f"Generating image with prompt: {prompt} and {len(images)} input image(s)")
        result = pipe(
            prompt=prompt,
            image=images,  # Can input multiple images
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        image = result.images[0]
        return image
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLM-Image Web UI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .tabs {
            display: flex;
            background: #f5f5f5;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab-button {
            flex: 1;
            padding: 20px;
            background: none;
            border: none;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 500;
        }
        
        .tab-button:hover {
            background: #e8e8e8;
        }
        
        .tab-button.active {
            background: white;
            border-bottom: 3px solid #667eea;
            color: #667eea;
        }
        
        .tab-content {
            display: none;
            padding: 30px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #333;
        }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        .slider-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .slider-container {
            display: flex;
            flex-direction: column;
        }
        
        .slider-value {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #e0e0e0;
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            border: none;
        }
        
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
        }
        
        .file-upload {
            border: 2px dashed #667eea;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        
        .file-upload:hover {
            background: #f0f2ff;
            border-color: #764ba2;
        }
        
        .file-upload input[type="file"] {
            display: none;
        }
        
        .file-list {
            margin-top: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .file-item {
            background: #e8e8e8;
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .file-item img {
            width: 40px;
            height: 40px;
            object-fit: cover;
            border-radius: 4px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 500;
            width: 100%;
            margin-top: 20px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            background: #f5f5f5;
            color: #666;
            font-size: 0.9em;
        }
        
        .status.error {
            background: #fee;
            color: #c33;
        }
        
        .status.success {
            background: #efe;
            color: #3c3;
        }
        
        .result-container {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 12px;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        
        .download-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1em;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s;
        }
        
        .download-btn:hover {
            background: #218838;
            transform: translateY(-2px);
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        @media (max-width: 768px) {
            .two-column {
                grid-template-columns: 1fr;
            }
            
            .slider-group {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GLM-Image Web UI</h1>
            <p>Generate images using the zai-org/GLM-Image model</p>
        </div>
        
        <div class="tabs">
            <button class="tab-button active" onclick="switchTab('t2i')">Text-to-Image</button>
            <button class="tab-button" onclick="switchTab('i2i')">Image-to-Image</button>
        </div>
        
        <!-- Text-to-Image Tab -->
        <div id="t2i" class="tab-content active">
            <div class="two-column">
                <div>
                    <form id="t2i-form">
                        <div class="form-group">
                            <label for="t2i-prompt">Prompt</label>
                            <textarea id="t2i-prompt" name="prompt" placeholder="Enter your prompt here..." required></textarea>
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="t2i-height">Height</label>
                                    <span id="t2i-height-value">1024</span>
                                </div>
                                <input type="range" id="t2i-height" name="height" min="32" max="2048" value="1024" step="32">
                            </div>
                            
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="t2i-width">Width</label>
                                    <span id="t2i-width-value">1152</span>
                                </div>
                                <input type="range" id="t2i-width" name="width" min="32" max="2048" value="1152" step="32">
                            </div>
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="t2i-steps">Inference Steps</label>
                                    <span id="t2i-steps-value">50</span>
                                </div>
                                <input type="range" id="t2i-steps" name="steps" min="10" max="100" value="50" step="1">
                            </div>
                            
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="t2i-guidance">Guidance Scale</label>
                                    <span id="t2i-guidance-value">1.5</span>
                                </div>
                                <input type="range" id="t2i-guidance" name="guidance" min="1.0" max="10.0" value="1.5" step="0.1">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="t2i-seed">Seed (use -1 for random)</label>
                            <input type="number" id="t2i-seed" name="seed" value="42" step="1">
                        </div>
                        
                        <button type="submit" class="btn" id="t2i-btn">
                            <span id="t2i-btn-text">Generate</span>
                        </button>
                        
                        <div id="t2i-status" class="status" style="display: none;"></div>
                    </form>
                </div>
                
                <div>
                    <div id="t2i-result" class="result-container" style="display: none;">
                        <h3>Generated Image</h3>
                        <img id="t2i-image" class="result-image" alt="Generated image">
                        <a id="t2i-download" class="download-btn" download="output_t2i.png">Download Image</a>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Image-to-Image Tab -->
        <div id="i2i" class="tab-content">
            <div class="two-column">
                <div>
                    <form id="i2i-form">
                        <div class="form-group">
                            <label>Upload Images (can upload multiple)</label>
                            <div class="file-upload" onclick="document.getElementById('i2i-files').click()">
                                <input type="file" id="i2i-files" name="files" multiple accept="image/*" onchange="handleFileSelect(this, 'i2i')">
                                <p>Click to upload or drag and drop</p>
                                <p style="font-size: 0.9em; color: #666;">PNG, JPG, JPEG up to multiple files</p>
                            </div>
                            <div id="i2i-file-list" class="file-list"></div>
                        </div>
                        
                        <div class="form-group">
                            <label for="i2i-prompt">Prompt</label>
                            <textarea id="i2i-prompt" name="prompt" placeholder="Enter your prompt here..." required></textarea>
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="i2i-height">Height</label>
                                    <span id="i2i-height-value">1056</span>
                                </div>
                                <input type="range" id="i2i-height" name="height" min="32" max="2048" value="1056" step="32">
                            </div>
                            
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="i2i-width">Width</label>
                                    <span id="i2i-width-value">1024</span>
                                </div>
                                <input type="range" id="i2i-width" name="width" min="32" max="2048" value="1024" step="32">
                            </div>
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="i2i-steps">Inference Steps</label>
                                    <span id="i2i-steps-value">50</span>
                                </div>
                                <input type="range" id="i2i-steps" name="steps" min="10" max="100" value="50" step="1">
                            </div>
                            
                            <div class="slider-container">
                                <div class="slider-value">
                                    <label for="i2i-guidance">Guidance Scale</label>
                                    <span id="i2i-guidance-value">1.5</span>
                                </div>
                                <input type="range" id="i2i-guidance" name="guidance" min="1.0" max="10.0" value="1.5" step="0.1">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label for="i2i-seed">Seed (use -1 for random)</label>
                            <input type="number" id="i2i-seed" name="seed" value="42" step="1">
                        </div>
                        
                        <button type="submit" class="btn" id="i2i-btn">
                            <span id="i2i-btn-text">Generate</span>
                        </button>
                        
                        <div id="i2i-status" class="status" style="display: none;"></div>
                    </form>
                </div>
                
                <div>
                    <div id="i2i-result" class="result-container" style="display: none;">
                        <h3>Generated Image</h3>
                        <img id="i2i-image" class="result-image" alt="Generated image">
                        <a id="i2i-download" class="download-btn" download="output_i2i.png">Download Image</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            document.getElementById(tab).classList.add('active');
            event.target.classList.add('active');
        }
        
        // Update slider values
        function setupSliders(prefix) {
            const sliders = ['height', 'width', 'steps', 'guidance'];
            sliders.forEach(name => {
                const slider = document.getElementById(`${prefix}-${name}`);
                const valueSpan = document.getElementById(`${prefix}-${name}-value`);
                if (slider && valueSpan) {
                    slider.addEventListener('input', (e) => {
                        valueSpan.textContent = e.target.value;
                    });
                }
            });
        }
        
        setupSliders('t2i');
        setupSliders('i2i');
        
        // File upload handling
        let uploadedFiles = { t2i: [], i2i: [] };
        
        function handleFileSelect(input, prefix) {
            const files = Array.from(input.files);
            uploadedFiles[prefix] = files;
            const fileList = document.getElementById(`${prefix}-file-list`);
            fileList.innerHTML = '';
            
            files.forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const div = document.createElement('div');
                    div.className = 'file-item';
                    div.innerHTML = `
                        <img src="${e.target.result}" alt="${file.name}">
                        <span>${file.name}</span>
                    `;
                    fileList.appendChild(div);
                };
                reader.readAsDataURL(file);
            });
        }
        
        // Text-to-Image form submission
        document.getElementById('t2i-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('t2i-btn');
            const btnText = document.getElementById('t2i-btn-text');
            const status = document.getElementById('t2i-status');
            const result = document.getElementById('t2i-result');
            const image = document.getElementById('t2i-image');
            const download = document.getElementById('t2i-download');
            
            btn.disabled = true;
            btnText.innerHTML = '<span class="loading"></span> Generating...';
            status.style.display = 'block';
            status.className = 'status';
            status.textContent = 'Generating image...';
            result.style.display = 'none';
            
            const formData = {
                prompt: document.getElementById('t2i-prompt').value,
                height: parseInt(document.getElementById('t2i-height').value),
                width: parseInt(document.getElementById('t2i-width').value),
                num_inference_steps: parseInt(document.getElementById('t2i-steps').value),
                guidance_scale: parseFloat(document.getElementById('t2i-guidance').value),
                seed: parseInt(document.getElementById('t2i-seed').value)
            };
            
            try {
                const response = await fetch('/api/text-to-image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Generation failed');
                }
                
                const data = await response.json();
                image.src = data.image;
                download.href = data.image;
                result.style.display = 'block';
                status.className = 'status success';
                status.textContent = 'Image generated successfully!';
            } catch (error) {
                status.className = 'status error';
                status.textContent = error.message;
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Generate';
            }
        });
        
        // Image-to-Image form submission
        document.getElementById('i2i-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('i2i-btn');
            const btnText = document.getElementById('i2i-btn-text');
            const status = document.getElementById('i2i-status');
            const result = document.getElementById('i2i-result');
            const image = document.getElementById('i2i-image');
            const download = document.getElementById('i2i-download');
            
            if (uploadedFiles.i2i.length === 0) {
                status.style.display = 'block';
                status.className = 'status error';
                status.textContent = 'Please upload at least one image';
                return;
            }
            
            btn.disabled = true;
            btnText.innerHTML = '<span class="loading"></span> Generating...';
            status.style.display = 'block';
            status.className = 'status';
            status.textContent = 'Generating image...';
            result.style.display = 'none';
            
            const formData = new FormData();
            uploadedFiles.i2i.forEach(file => {
                formData.append('files', file);
            });
            formData.append('prompt', document.getElementById('i2i-prompt').value);
            formData.append('height', document.getElementById('i2i-height').value);
            formData.append('width', document.getElementById('i2i-width').value);
            formData.append('num_inference_steps', document.getElementById('i2i-steps').value);
            formData.append('guidance_scale', document.getElementById('i2i-guidance').value);
            formData.append('seed', document.getElementById('i2i-seed').value);
            
            try {
                const response = await fetch('/api/image-to-image', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Generation failed');
                }
                
                const data = await response.json();
                image.src = data.image;
                download.href = data.image;
                result.style.display = 'block';
                status.className = 'status success';
                status.textContent = 'Image generated successfully!';
            } catch (error) {
                status.className = 'status error';
                status.textContent = error.message;
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Generate';
            }
        });
    </script>
</body>
</html>
    """
    return html_content

@app.post("/api/text-to-image")
async def api_text_to_image(request: TextToImageRequest):
    """API endpoint for text-to-image generation"""
    try:
        image = text_to_image(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        image_base64 = image_to_base64(image)
        return JSONResponse(content={"image": image_base64, "status": "success"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image-to-image")
async def api_image_to_image(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
    height: int = Form(1056),
    width: int = Form(1024),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42)
):
    """API endpoint for image-to-image generation"""
    try:
        # Process uploaded files
        pil_images = []
        for file in files:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            pil_images.append(img)
        
        image = image_to_image(
            images=pil_images,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        image_base64 = image_to_base64(image)
        return JSONResponse(content={"image": image_base64, "status": "success"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
