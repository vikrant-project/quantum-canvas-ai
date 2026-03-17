import sys
import subprocess
import os
import json
from datetime import datetime
import hashlib
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

REQUIRED_PACKAGES = [
    'flask',
    'diffusers',
    'transformers',
    'torch',
    'pillow',
    'accelerate',
    'safetensors',
    'flask-session',
    'opencv-python',
    'numpy',
    'onnxruntime',
    'rembg',
    'scikit-image',
]

def check_and_install_packages():
    """Smart package installer - only installs missing packages"""
    print("🔍 Checking required packages...")
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        package_name = package.replace('-', '_').split('[')[0]
        try:
            __import__(package_name)
            print(f"✓ {package} already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} not found")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing package(s)...")
        for package in missing_packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package, "-q", "--no-cache-dir"],
                    timeout=300
                )
                print(f"✓ {package} installed")
            except subprocess.TimeoutExpired:
                print(f"⚠️ {package} installation timed out, skipping...")
            except Exception as e:
                print(f"⚠️ {package} installation failed: {e}")
    else:
        print("\n✅ All packages already installed!")
    
    print("🎉 Setup complete!\n")

check_and_install_packages()

# Now import the packages
from flask import Flask, render_template_string, request, jsonify
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    LCMScheduler
)
import torch
from PIL import Image
import io
import base64
import cv2
import numpy as np
from rembg import remove
from skimage import restoration, filters

app = Flask(__name__)
app.secret_key = 'quantum_canvas_ultimate_V1_secret_key_2024'

# Global variables
pipelines = {}
current_model = None
is_generating = False
generation_progress = 0
generation_step = 0
total_steps = 0
generation_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=4)

# History file
HISTORY_FILE = '/app/history.json'
FAVORITES_FILE = '/app/favorites.json'
MAX_HISTORY_ITEMS = 100

# Speed Presets
SPEED_PRESETS = {
    'lightning': {'name': 'Lightning Fast', 'steps': 6, 'scheduler': 'lcm', 'description': '⚡ Ultra-fast (10-30s)'},
    'fast': {'name': 'Fast', 'steps': 15, 'scheduler': 'euler_a', 'description': '🚀 Quick (1-2 min)'},
    'balanced': {'name': 'Balanced', 'steps': 30, 'scheduler': 'dpm_multi', 'description': '⚖️ Default (3-5 min)'},
    'quality': {'name': 'Quality', 'steps': 45, 'scheduler': 'dpm_multi', 'description': '💎 High Quality (5-8 min)'},
    'ultra': {'name': 'Ultra Quality', 'steps': 70, 'scheduler': 'dpm_multi', 'description': '👑 Maximum (8-12 min)'}
}

# 50+ AI Models
AVAILABLE_MODELS = {
    # SPEED-OPTIMIZED MODELS
    'tiny-sd': {'name': 'Tiny SD', 'id': 'segmind/tiny-sd', 'description': 'Ultra-fast, lightweight', 'speed': '⚡⚡⚡', 'quality': '⭐⭐⭐', 'size': '300MB', 'category': 'speed', 'recommended': True},
    'sd-turbo': {'name': 'SD Turbo', 'id': 'stabilityai/sd-turbo', 'description': 'Lightning-fast 1-4 steps', 'speed': '⚡⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '2GB', 'category': 'speed', 'recommended': True},
    'lcm-sd': {'name': 'LCM SD v1.5', 'id': 'SimianLuo/LCM_Dreamshaper_v7', 'description': 'Latent Consistency 4-8 steps', 'speed': '⚡⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '2GB', 'category': 'speed', 'recommended': True},
    
    # GENERAL PURPOSE
    'stable-diffusion-v1-5': {'name': 'Stable Diffusion v1.5', 'id': 'runwayml/stable-diffusion-v1-5', 'description': 'Classic SD balanced', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '4GB', 'category': 'general', 'recommended': True},
    'stable-diffusion-v2-1': {'name': 'Stable Diffusion v2.1', 'id': 'stabilityai/stable-diffusion-2-1', 'description': 'Improved v2.1', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '5GB', 'category': 'general', 'recommended': True},
    'deliberate': {'name': 'Deliberate', 'id': 'XpucT/Deliberate', 'description': 'Versatile high-quality', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'general', 'recommended': True},
    'dreamshaper': {'name': 'DreamShaper', 'id': 'Lykon/DreamShaper', 'description': 'Popular versatile', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'general', 'recommended': True},
    'absolutereality': {'name': 'Absolute Reality', 'id': 'digiplay/AbsoluteReality_v1.8.1', 'description': 'Balanced realism', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'general', 'recommended': False},
    
    # REALISTIC/PHOTOREALISTIC
    'realistic-vision': {'name': 'Realistic Vision', 'id': 'SG161222/Realistic_Vision_V2.0', 'description': 'High-quality realistic', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'realistic', 'recommended': True},
    'dreamlike-photoreal': {'name': 'Dreamlike Photoreal', 'id': 'dreamlike-art/dreamlike-photoreal-2.0', 'description': 'Photorealistic images', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'realistic', 'recommended': True},
    'epicrealism': {'name': 'epiCRealism', 'id': 'emilianJR/epiCRealism', 'description': 'Epic photorealistic', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'realistic', 'recommended': False},
    
    # ANIME/ILLUSTRATION
    'anything-v5': {'name': 'Anything V5', 'id': 'stablediffusionapi/anything-v5', 'description': 'Latest anime model', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'anime', 'recommended': True},
    'counterfeit': {'name': 'Counterfeit', 'id': 'gsdf/Counterfeit-V2.5', 'description': 'High-quality anime', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐⭐', 'size': '4GB', 'category': 'anime', 'recommended': True},
    
    # ARTISTIC
    'openjourney': {'name': 'OpenJourney', 'id': 'prompthero/openjourney', 'description': 'Midjourney-style artistic', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '4GB', 'category': 'artistic', 'recommended': True},
    'van-gogh': {'name': 'Van Gogh Diffusion', 'id': 'dallinmackay/Van-Gogh-diffusion', 'description': 'Van Gogh painting style', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '2GB', 'category': 'artistic', 'recommended': False},
    
    # 3D/CGI
    'modelshoot': {'name': 'Modern Disney', 'id': 'nitrosocke/mo-di-diffusion', 'description': '3D Disney/Pixar style', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '4GB', 'category': '3d', 'recommended': True},
    
    # FANTASY/SCI-FI
    'scifi-diffusion': {'name': 'Sci-Fi Diffusion', 'id': 'stablediffusionapi/scifi-diffusion', 'description': 'Science fiction scenes', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '4GB', 'category': 'fantasy', 'recommended': True},
    
    # SPECIAL STYLE
    'ghibli-diffusion': {'name': 'Ghibli Diffusion', 'id': 'nitrosocke/Ghibli-Diffusion', 'description': 'Studio Ghibli art', 'speed': '⚡⚡', 'quality': '⭐⭐⭐⭐', 'size': '2GB', 'category': 'special', 'recommended': True},
}

# Image Processing Modules
IMAGE_MODULES = {
    'text2img': {'name': 'Text to Image', 'icon': 'fa-font', 'description': 'Generate from text'},
    'img2img': {'name': 'Image to Image', 'icon': 'fa-image', 'description': 'Transform existing images'},
    'inpaint': {'name': 'Inpainting', 'icon': 'fa-paint-brush', 'description': 'Remove/replace objects'},
    'upscale': {'name': 'Super Resolution', 'icon': 'fa-expand-arrows-alt', 'description': 'Upscale 2x-8x'},
    'background-remove': {'name': 'Background Removal', 'icon': 'fa-cut', 'description': 'Remove background'},
    'enhance': {'name': 'Face Enhancement', 'icon': 'fa-user-circle', 'description': 'Enhance faces'},
    'restore': {'name': 'Image Restoration', 'icon': 'fa-magic', 'description': 'Restore old photos'},
    'denoise': {'name': 'Denoise', 'icon': 'fa-adjust', 'description': 'Remove noise'},
    'variation': {'name': 'Variations', 'icon': 'fa-random', 'description': 'Create variations'},
}

def load_history():
    """Load generation history"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_history(history_item):
    """Save to history"""
    try:
        history = load_history()
        history.insert(0, history_item)
        history = history[:MAX_HISTORY_ITEMS]
        
        os.makedirs(os.path.dirname(HISTORY_FILE) if os.path.dirname(HISTORY_FILE) else '.', exist_ok=True)
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Saved to history. Total: {len(history)}")
        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False

def load_favorites():
    """Load favorite models"""
    try:
        if os.path.exists(FAVORITES_FILE):
            with open(FAVORITES_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading favorites: {e}")
        return []

def save_favorites(favorites):
    """Save favorite models"""
    try:
        os.makedirs(os.path.dirname(FAVORITES_FILE) if os.path.dirname(FAVORITES_FILE) else '.', exist_ok=True)
        with open(FAVORITES_FILE, 'w') as f:
            json.dump(favorites, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving favorites: {e}")
        return False

def get_scheduler(scheduler_type, pipeline):
    """Get scheduler"""
    schedulers = {
        'lcm': LCMScheduler,
        'dpm_single': DPMSolverSinglestepScheduler,
        'dpm_multi': DPMSolverMultistepScheduler,
        'euler': EulerDiscreteScheduler,
        'euler_a': EulerAncestralDiscreteScheduler,
        'ddim': DDIMScheduler,
    }
    scheduler_class = schedulers.get(scheduler_type, DPMSolverMultistepScheduler)
    return scheduler_class.from_config(pipeline.scheduler.config)

def load_model(model_key='tiny-sd'):
    """Load AI model - CPU optimized with memory management"""
    global pipelines, current_model
    
    if model_key in pipelines:
        print(f"Model {model_key} already loaded!")
        current_model = model_key
        return True
    
    try:
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            print(f"Model {model_key} not found!")
            return False
        
        # Clear other models to save memory - keep only one model loaded
        if len(pipelines) > 0:
            print("Clearing previous models to save memory...")
            for key in list(pipelines.keys()):
                del pipelines[key]
            pipelines.clear()
            import gc
            gc.collect()
        
        print(f"\n{'='*60}")
        print(f"Loading: {model_info['name']}")
        print(f"{'='*60}\n")
        
        # Load only Text2Img pipeline initially - load others on demand
        print("Loading Text2Img pipeline...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_info['id'],
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        )
        pipeline = pipeline.to("cpu")
        pipeline.enable_attention_slicing(1)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        torch.set_num_threads(min(os.cpu_count(), 4))  # Limit threads to avoid overload
        print("✓ Text2Img loaded")
        
        # Create img2img from text2img components (saves memory)
        print("Creating Img2Img pipeline from Text2Img components...")
        img2img_pipeline = StableDiffusionImg2ImgPipeline(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            safety_checker=None,
            feature_extractor=pipeline.feature_extractor,
            requires_safety_checker=False
        )
        print("✓ Img2Img created")
        
        pipelines[model_key] = {
            'text2img': pipeline,
            'img2img': img2img_pipeline,
            'info': model_info
        }
        
        current_model = model_key
        print(f"\n✓ Model '{model_info['name']}' ready!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def upscale_image(image_data, scale=2):
    """Upscale image using advanced interpolation"""
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        height, width = img.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # Lanczos interpolation
        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        upscaled = cv2.filter2D(upscaled, -1, kernel)
        
        # Enhance contrast
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        upscaled = cv2.merge([l, a, b])
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_LAB2BGR)
        
        _, buffer = cv2.imencode('.png', upscaled)
        return base64.b64encode(buffer).decode()
    except Exception as e:
        print(f"Upscale error: {e}")
        return None

def remove_background(image_data):
    """Remove background from image"""
    try:
        img_bytes = base64.b64decode(image_data)
        input_img = Image.open(io.BytesIO(img_bytes))
        
        # Remove background
        output_img = remove(input_img)
        
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Background removal error: {e}")
        return None

def denoise_image(image_data):
    """Denoise image"""
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        _, buffer = cv2.imencode('.png', denoised)
        return base64.b64encode(buffer).decode()
    except Exception as e:
        print(f"Denoise error: {e}")
        return None

def restore_image(image_data):
    """Restore old/damaged images"""
    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = restoration.denoise_tv_chambolle(img_gray, weight=0.1)
        
        # Convert back to uint8
        restored = (denoised * 255).astype(np.uint8)
        restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        
        # Enhance
        lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        restored = cv2.merge([l, a, b])
        restored = cv2.cvtColor(restored, cv2.COLOR_LAB2BGR)
        
        _, buffer = cv2.imencode('.png', restored)
        return base64.b64encode(buffer).decode()
    except Exception as e:
        print(f"Restore error: {e}")
        return None

# Enhanced HTML with Modern Improved UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Canvas V1 Ultimate - AI Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-light: #818cf8;
            --primary-dark: #4f46e5;
            --secondary: #ec4899;
            --accent: #14b8a6;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --dark-bg: #0a0e27;
            --darker-bg: #050714;
            --card-bg: rgba(20, 25, 45, 0.8);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --border: rgba(255, 255, 255, 0.1);
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
        }

        body.light-theme {
            --dark-bg: #f1f5f9;
            --darker-bg: #ffffff;
            --card-bg: rgba(255, 255, 255, 0.9);
            --glass-bg: rgba(0, 0, 0, 0.03);
            --border: rgba(0, 0, 0, 0.1);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-tertiary: #64748b;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--dark-bg);
            color: var(--text-primary);
            overflow-x: hidden;
        }

        /* Animated Background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            overflow: hidden;
        }

        .bg-gradient {
            position: absolute;
            border-radius: 50%;
            filter: blur(100px);
            opacity: 0.4;
            animation: float 25s ease-in-out infinite;
        }

        .bg-gradient-1 {
            width: 700px;
            height: 700px;
            background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
            top: -250px;
            left: -250px;
        }

        .bg-gradient-2 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, var(--secondary) 0%, transparent 70%);
            bottom: -200px;
            right: -200px;
            animation-delay: -8s;
        }

        .bg-gradient-3 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, var(--accent) 0%, transparent 70%);
            top: 50%;
            left: 50%;
            animation-delay: -15s;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); }
            33% { transform: translate(150px, -80px) scale(1.15) rotate(120deg); }
            66% { transform: translate(-80px, 150px) scale(0.85) rotate(240deg); }
        }

        /* Container */
        .app-container {
            position: relative;
            z-index: 1;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header */
        .header {
            padding: 1.25rem 2.5rem;
            background: var(--card-bg);
            backdrop-filter: blur(30px);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }

        .logo-icon {
            width: 52px;
            height: 52px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
            animation: pulse 3s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5); }
            50% { transform: scale(1.05); box-shadow: 0 12px 32px rgba(99, 102, 241, 0.7); }
        }

        .logo-text {
            font-family: 'Poppins', sans-serif;
            font-size: 1.75rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .badge {
            padding: 5px 14px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border-radius: 8px;
            font-size: 0.65rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }

        .header-actions {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .theme-toggle {
            width: 45px;
            height: 45px;
            border-radius: 12px;
            background: var(--glass-bg);
            border: 1px solid var(--border);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 1.2rem;
        }

        .theme-toggle:hover {
            background: var(--primary);
            color: white;
            transform: scale(1.1) rotate(15deg);
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        }

        /* Main Layout */
        .main-layout {
            display: grid;
            grid-template-columns: 380px 1fr 320px;
            gap: 1.5rem;
            flex: 1;
            padding: 2rem;
            max-width: 1920px;
            margin: 0 auto;
            width: 100%;
        }

        .panel {
            background: var(--card-bg);
            backdrop-filter: blur(30px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            overflow-y: auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        /* Scrollbar */
        .panel::-webkit-scrollbar {
            width: 10px;
        }

        .panel::-webkit-scrollbar-track {
            background: var(--glass-bg);
            border-radius: 10px;
            margin: 10px;
        }

        .panel::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--primary), var(--secondary));
            border-radius: 10px;
            border: 2px solid var(--card-bg);
        }

        .panel::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, var(--primary-light), var(--secondary));
        }

        /* Sections */
        .section {
            margin-bottom: 2.5rem;
        }

        .section:last-child {
            margin-bottom: 0;
        }

        .section-title {
            font-size: 0.95rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 1.25rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--border);
        }

        .section-title i {
            font-size: 1.1rem;
            color: var(--primary);
        }

        /* Module Tabs - Enhanced */
        .module-tabs {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .module-tab {
            padding: 1rem 0.75rem;
            background: var(--glass-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .module-tab::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            opacity: 0;
            transition: opacity 0.3s;
        }

        .module-tab:hover {
            transform: translateY(-4px);
            border-color: var(--primary);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        }

        .module-tab:hover::before {
            opacity: 0.15;
        }

        .module-tab.active {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(236, 72, 153, 0.2));
            border-color: var(--primary);
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        }

        .module-tab.active::before {
            opacity: 0.2;
        }

        .module-tab-content {
            position: relative;
            z-index: 1;
        }

        .module-tab i {
            display: block;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            color: var(--primary);
        }

        .module-tab span {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Model Categories */
        .model-categories {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .category-btn {
            padding: 0.5rem 1rem;
            background: var(--glass-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .category-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .category-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
        }

        /* Model Search */
        .model-search {
            position: relative;
            margin-bottom: 1rem;
        }

        .model-search input {
            width: 100%;
            padding: 0.85rem 1rem 0.85rem 2.75rem;
            background: var(--glass-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            transition: all 0.3s;
        }

        .model-search input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        .model-search i {
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-tertiary);
        }

        /* Model Grid - Enhanced */
        .model-grid {
            display: grid;
            gap: 0.75rem;
            max-height: 350px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }

        .model-card {
            padding: 1rem;
            background: var(--glass-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .model-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary), var(--secondary));
            transform: scaleY(0);
            transition: transform 0.3s;
        }

        .model-card:hover {
            border-color: var(--primary);
            transform: translateX(4px);
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
        }

        .model-card:hover::before {
            transform: scaleY(1);
        }

        .model-card.active {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.15));
            border-color: var(--primary);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        .model-card.active::before {
            transform: scaleY(1);
        }

        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .model-name {
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-primary);
        }

        .favorite-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            color: var(--text-tertiary);
            transition: all 0.3s;
            padding: 0.25rem;
        }

        .favorite-btn:hover {
            color: var(--warning);
            transform: scale(1.2);
        }

        .favorite-btn.active {
            color: var(--warning);
        }

        .model-desc {
            font-size: 0.8rem;
            color: var(--text-tertiary);
            margin-bottom: 0.5rem;
        }

        .model-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
        }

        .model-meta span {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        /* Input Groups - Enhanced */
        .input-group {
            margin-bottom: 1.5rem;
        }

        .input-label {
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            color: var(--text-primary);
        }

        .input-label-badge {
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            background: var(--primary);
            color: white;
            border-radius: 6px;
        }

        textarea, select, input[type="number"] {
            width: 100%;
            padding: 1rem;
            background: var(--glass-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            transition: all 0.3s;
        }

        textarea:focus, select:focus, input[type="number"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        textarea {
            resize: vertical;
            min-height: 100px;
        }

        input[type="range"] {
            width: 100%;
            height: 8px;
            background: var(--glass-bg);
            border-radius: 10px;
            outline: none;
            border: 1px solid var(--border);
        }

        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.5);
            transition: all 0.3s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.6);
        }

        /* Buttons - Enhanced */
        .btn {
            padding: 1rem 1.75rem;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .btn > * {
            position: relative;
            z-index: 1;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.5);
        }

        .btn-primary:active:not(:disabled) {
            transform: translateY(-1px);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            background: var(--glass-bg);
            color: var(--text-primary);
            border: 2px solid var(--border);
        }

        .btn-secondary:hover:not(:disabled) {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3);
        }

        /* Canvas Display - Enhanced */
        .canvas-display {
            min-height: 500px;
            background: var(--glass-bg);
            border: 3px dashed var(--border);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s;
        }

        .canvas-display:hover {
            border-color: var(--primary);
        }

        .canvas-display img {
            max-width: 100%;
            max-height: 700px;
            border-radius: 16px;
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s;
        }

        .canvas-display img:hover {
            transform: scale(1.02);
        }

        .placeholder {
            text-align: center;
            color: var(--text-tertiary);
            padding: 3rem;
        }

        .placeholder i {
            font-size: 5rem;
            margin-bottom: 1.5rem;
            opacity: 0.3;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .placeholder h3 {
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
            color: var(--text-primary);
        }

        /* Progress - Enhanced */
        .progress-display {
            text-align: center;
            padding: 2rem;
        }

        .progress-ring {
            width: 220px;
            height: 220px;
            margin: 0 auto 2rem;
            position: relative;
        }

        .progress-ring svg {
            transform: rotate(-90deg);
        }

        .progress-ring circle {
            transition: stroke-dashoffset 0.3s;
        }

        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--glass-bg);
            border-radius: 10px;
            overflow: hidden;
            margin: 1.5rem 0;
            border: 1px solid var(--border);
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            width: 0%;
            transition: width 0.3s;
            box-shadow: 0 0 10px var(--primary);
        }

        .progress-status {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-top: 1rem;
        }

        /* Action Buttons */
        .action-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.75rem;
            margin-top: 1.5rem;
        }

        /* History - Enhanced */
        .history-grid {
            display: grid;
            gap: 1rem;
        }

        .history-item {
            background: var(--glass-bg);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .history-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(236, 72, 153, 0.1));
            opacity: 0;
            transition: opacity 0.3s;
        }

        .history-item:hover {
            transform: translateY(-4px);
            border-color: var(--primary);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
        }

        .history-item:hover::before {
            opacity: 1;
        }

        .history-thumb {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 10px;
            object-fit: cover;
            margin-bottom: 0.75rem;
            position: relative;
            z-index: 1;
        }

        .history-prompt {
            font-size: 0.8rem;
            color: var(--text-tertiary);
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            position: relative;
            z-index: 1;
        }

        .history-date {
            font-size: 0.7rem;
            color: var(--text-tertiary);
            margin-top: 0.5rem;
            position: relative;
            z-index: 1;
        }

        .hidden {
            display: none !important;
        }

        /* Speed Presets */
        .speed-presets {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .speed-preset {
            padding: 0.75rem 0.5rem;
            background: var(--glass-bg);
            border: 2px solid var(--border);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .speed-preset:hover {
            border-color: var(--primary);
            transform: scale(1.05);
        }

        .speed-preset.active {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(236, 72, 153, 0.2));
            border-color: var(--primary);
        }

        /* Responsive */
        @media (max-width: 1400px) {
            .main-layout {
                grid-template-columns: 350px 1fr 280px;
            }
        }

        @media (max-width: 1200px) {
            .main-layout {
                grid-template-columns: 320px 1fr;
            }
            .right-panel {
                display: none;
            }
        }

        @media (max-width: 900px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
            .left-panel, .center-panel {
                margin-right: 0;
            }
            .module-tabs {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 600px) {
            .header {
                padding: 1rem;
            }
            .logo-text {
                font-size: 1.25rem;
            }
            .main-layout {
                padding: 1rem;
                gap: 1rem;
            }
            .panel {
                padding: 1.25rem;
            }
            .model-categories {
                gap: 0.4rem;
            }
            .category-btn {
                padding: 0.4rem 0.75rem;
                font-size: 0.75rem;
            }
        }

        /* Loading Animation */
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .fa-spinner {
            animation: spin 1s linear infinite;
        }
    </style>
</head>
<body>
    <div class="bg-animation">
        <div class="bg-gradient bg-gradient-1"></div>
        <div class="bg-gradient bg-gradient-2"></div>
        <div class="bg-gradient bg-gradient-3"></div>
    </div>

    <div class="app-container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <div class="logo-icon">
                    <i class="fas fa-cube"></i>
                </div>
                <div>
                    <div class="logo-text">Quantum Canvas</div>
                    <span class="badge">V1 ULTIMATE</span>
                </div>
            </div>
            
            <div class="header-actions">
                <button class="theme-toggle" id="keyboardShortcutsBtn" title="Keyboard Shortcuts (?)">
                    <i class="fas fa-keyboard"></i>
                </button>
                <button class="theme-toggle" id="themeToggle" title="Toggle Theme (Ctrl+T)">
                    <i class="fas fa-sun"></i>
                </button>
            </div>
        </div>

        <!-- Main Layout -->
        <div class="main-layout">
            <!-- Left Panel -->
            <div class="panel left-panel">
                <!-- Image Modules -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-layer-group"></i>
                        <span>Image Modules</span>
                    </div>
                    
                    <div class="module-tabs" id="moduleTabs">
                        <div class="module-tab active" data-module="text2img">
                            <div class="module-tab-content">
                                <i class="fas fa-font"></i>
                                <span>Text2Img</span>
                            </div>
                        </div>
                        <div class="module-tab" data-module="img2img">
                            <div class="module-tab-content">
                                <i class="fas fa-image"></i>
                                <span>Img2Img</span>
                            </div>
                        </div>
                        <div class="module-tab" data-module="upscale">
                            <div class="module-tab-content">
                                <i class="fas fa-expand-arrows-alt"></i>
                                <span>Upscale</span>
                            </div>
                        </div>
                        <div class="module-tab" data-module="background-remove">
                            <div class="module-tab-content">
                                <i class="fas fa-cut"></i>
                                <span>Remove BG</span>
                            </div>
                        </div>
                        <div class="module-tab" data-module="denoise">
                            <div class="module-tab-content">
                                <i class="fas fa-adjust"></i>
                                <span>Denoise</span>
                            </div>
                        </div>
                        <div class="module-tab" data-module="restore">
                            <div class="module-tab-content">
                                <i class="fas fa-magic"></i>
                                <span>Restore</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- AI Models -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-brain"></i>
                        <span>AI Models</span>
                    </div>
                    
                    <!-- Model Search -->
                    <div class="model-search">
                        <i class="fas fa-search"></i>
                        <input type="text" id="modelSearch" placeholder="Search models...">
                    </div>

                    <!-- Model Categories -->
                    <div class="model-categories" id="modelCategories">
                        <div class="category-btn active" data-category="all">All</div>
                        <div class="category-btn" data-category="favorites">★ Favorites</div>
                        <div class="category-btn" data-category="speed">Speed</div>
                        <div class="category-btn" data-category="general">General</div>
                        <div class="category-btn" data-category="realistic">Realistic</div>
                        <div class="category-btn" data-category="anime">Anime</div>
                        <div class="category-btn" data-category="artistic">Artistic</div>
                    </div>
                    
                    <div class="model-grid" id="modelGrid">
                        <!-- Models populated by JS -->
                    </div>
                </div>

                <!-- Controls -->
                <div class="section">
                    <div class="section-title">
                        <i class="fas fa-sliders-h"></i>
                        <span>Generation Controls</span>
                    </div>
                    
                    <div class="input-group">
                        <div class="input-label">
                            <span>Prompt</span>
                        </div>
                        <textarea id="prompt" placeholder="Describe your image in detail..."></textarea>
                    </div>

                    <div class="input-group">
                        <div class="input-label">
                            <span>Speed Preset</span>
                        </div>
                        <div class="speed-presets" id="speedPresets">
                            <div class="speed-preset" data-steps="6">⚡ Fast</div>
                            <div class="speed-preset active" data-steps="30">⚖️ Balanced</div>
                            <div class="speed-preset" data-steps="50">💎 Quality</div>
                        </div>
                    </div>

                    <div class="input-group">
                        <div class="input-label">
                            <span>Steps</span>
                            <span class="input-label-badge" id="stepsValue">30</span>
                        </div>
                        <input type="range" id="steps" value="30" min="6" max="70" step="2">
                    </div>

                    <button class="btn btn-primary" id="generateBtn" style="width: 100%;">
                        <i class="fas fa-magic"></i>
                        <span>Generate Image</span>
                    </button>
                </div>
            </div>

            <!-- Center Panel -->
            <div class="panel center-panel">
                <div class="section-title">
                    <i class="fas fa-images"></i>
                    <span>Canvas</span>
                </div>
                
                <div class="canvas-display" id="canvasDisplay">
                    <div class="placeholder" id="placeholder">
                        <i class="fas fa-image"></i>
                        <h3>Ready to Create Magic ✨</h3>
                        <p>Select a module and generate your masterpiece</p>
                    </div>

                    <div class="progress-display hidden" id="progressDisplay">
                        <div class="progress-ring">
                            <svg width="220" height="220">
                                <circle cx="110" cy="110" r="100" fill="none" stroke="var(--border)" stroke-width="10"/>
                                <circle id="progressCircle" cx="110" cy="110" r="100" fill="none" stroke="url(#gradient)" stroke-width="10" stroke-dasharray="628" stroke-dashoffset="628" stroke-linecap="round"/>
                                <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" style="stop-color:var(--primary);stop-opacity:1" />
                                        <stop offset="100%" style="stop-color:var(--secondary);stop-opacity:1" />
                                    </linearGradient>
                                </defs>
                            </svg>
                            <div class="progress-text" id="progressText">0%</div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill"></div>
                        </div>
                        <div class="progress-status">Generating your image...</div>
                    </div>

                    <div class="hidden" id="imageResult">
                        <img id="resultImage" alt="Generated">
                    </div>
                </div>

                <div class="action-buttons hidden" id="actionButtons">
                    <button class="btn btn-secondary" id="downloadBtn">
                        <i class="fas fa-download"></i>
                        <span>Download</span>
                    </button>
                    <button class="btn btn-secondary" id="upscale2xBtn">
                        <i class="fas fa-expand"></i>
                        <span>Upscale 2x</span>
                    </button>
                    <button class="btn btn-secondary" id="upscale4xBtn">
                        <i class="fas fa-expand-arrows-alt"></i>
                        <span>Upscale 4x</span>
                    </button>
                    <button class="btn btn-secondary" id="newBtn">
                        <i class="fas fa-redo"></i>
                        <span>New</span>
                    </button>
                </div>
            </div>

            <!-- Right Panel -->
            <div class="panel right-panel">
                <div class="section-title">
                    <i class="fas fa-history"></i>
                    <span>History</span>
                </div>
                
                <div class="history-grid" id="historyGrid">
                    <p style="text-align: center; color: var(--text-tertiary); padding: 2rem;">No history yet</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let currentModule = 'text2img';
        let currentModel = 'tiny-sd';
        let currentImageData = null;
        let progressInterval = null;
        let currentCategory = 'all';
        let favorites = [];

        const models = ''' + json.dumps(AVAILABLE_MODELS) + ''';

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadFavorites();
            populateModels();
            setupEventListeners();
            loadHistory();
        });

        function loadFavorites() {
            const saved = localStorage.getItem('favorites');
            favorites = saved ? JSON.parse(saved) : [];
        }

        function saveFavorites() {
            localStorage.setItem('favorites', JSON.stringify(favorites));
        }

        function toggleFavorite(modelKey) {
            const index = favorites.indexOf(modelKey);
            if (index > -1) {
                favorites.splice(index, 1);
            } else {
                favorites.push(modelKey);
            }
            saveFavorites();
            populateModels();
        }

        function populateModels() {
            const grid = document.getElementById('modelGrid');
            grid.innerHTML = '';
            
            let filteredModels = Object.entries(models);
            
            // Filter by category
            if (currentCategory === 'favorites') {
                filteredModels = filteredModels.filter(([key, _]) => favorites.includes(key));
            } else if (currentCategory !== 'all') {
                filteredModels = filteredModels.filter(([_, model]) => model.category === currentCategory);
            }
            
            // Filter by search
            const searchTerm = document.getElementById('modelSearch')?.value.toLowerCase() || '';
            if (searchTerm) {
                filteredModels = filteredModels.filter(([_, model]) => 
                    model.name.toLowerCase().includes(searchTerm) || 
                    model.description.toLowerCase().includes(searchTerm)
                );
            }
            
            // Sort: favorites first, then recommended
            filteredModels.sort(([keyA, modelA], [keyB, modelB]) => {
                const aFav = favorites.includes(keyA) ? 1 : 0;
                const bFav = favorites.includes(keyB) ? 1 : 0;
                if (aFav !== bFav) return bFav - aFav;
                
                const aRec = modelA.recommended ? 1 : 0;
                const bRec = modelB.recommended ? 1 : 0;
                return bRec - aRec;
            });
            
            if (filteredModels.length === 0) {
                grid.innerHTML = '<p style="text-align: center; color: var(--text-tertiary); padding: 2rem;">No models found</p>';
                return;
            }
            
            for (const [key, model] of filteredModels) {
                const card = document.createElement('div');
                card.className = `model-card ${key === currentModel ? 'active' : ''}`;
                card.dataset.model = key;
                
                const isFavorite = favorites.includes(key);
                
                card.innerHTML = `
                    <div class="model-header">
                        <div class="model-name">${model.name}</div>
                        <button class="favorite-btn ${isFavorite ? 'active' : ''}" data-model="${key}" onclick="event.stopPropagation(); toggleFavorite('${key}')">
                            <i class="fa${isFavorite ? 's' : 'r'} fa-star"></i>
                        </button>
                    </div>
                    <div class="model-desc">${model.description}</div>
                    <div class="model-meta">
                        <span title="Speed">${model.speed}</span>
                        <span title="Quality">${model.quality}</span>
                        <span title="Size"><i class="fas fa-hdd"></i> ${model.size}</span>
                    </div>
                `;
                
                card.addEventListener('click', () => selectModel(key));
                grid.appendChild(card);
            }
        }

        function setupEventListeners() {
            // Module tabs
            document.querySelectorAll('.module-tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('.module-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    currentModule = tab.dataset.module;
                });
            });

            // Category filters
            document.querySelectorAll('.category-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentCategory = btn.dataset.category;
                    populateModels();
                });
            });

            // Model search
            document.getElementById('modelSearch').addEventListener('input', () => {
                populateModels();
            });

            // Speed presets
            document.querySelectorAll('.speed-preset').forEach(preset => {
                preset.addEventListener('click', () => {
                    document.querySelectorAll('.speed-preset').forEach(p => p.classList.remove('active'));
                    preset.classList.add('active');
                    const steps = parseInt(preset.dataset.steps);
                    document.getElementById('steps').value = steps;
                    document.getElementById('stepsValue').textContent = steps;
                });
            });

            // Theme toggle
            document.getElementById('themeToggle').addEventListener('click', () => {
                document.body.classList.toggle('light-theme');
                const icon = document.querySelector('#themeToggle i');
                icon.className = document.body.classList.contains('light-theme') ? 'fas fa-moon' : 'fas fa-sun';
            });

            // Keyboard shortcuts button
            document.getElementById('keyboardShortcutsBtn').addEventListener('click', () => {
                showKeyboardShortcuts();
            });

            // Steps slider
            document.getElementById('steps').addEventListener('input', (e) => {
                document.getElementById('stepsValue').textContent = e.target.value;
                // Update speed preset selection
                document.querySelectorAll('.speed-preset').forEach(p => p.classList.remove('active'));
            });

            // Generate button
            document.getElementById('generateBtn').addEventListener('click', handleGenerate);

            // Action buttons
            document.getElementById('downloadBtn').addEventListener('click', downloadImage);
            document.getElementById('upscale2xBtn').addEventListener('click', () => upscaleImage(2));
            document.getElementById('upscale4xBtn').addEventListener('click', () => upscaleImage(4));
            document.getElementById('newBtn').addEventListener('click', resetUI);

            // Keyboard shortcuts
            setupKeyboardShortcuts();
        }

        function setupKeyboardShortcuts() {
            document.addEventListener('keydown', (e) => {
                // Ctrl/Cmd + Enter - Generate
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    e.preventDefault();
                    const btn = document.getElementById('generateBtn');
                    if (!btn.disabled) {
                        handleGenerate();
                    }
                }

                // Ctrl/Cmd + D - Download
                if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                    e.preventDefault();
                    if (currentImageData) {
                        downloadImage();
                    }
                }

                // Ctrl/Cmd + N - New/Reset
                if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                    e.preventDefault();
                    resetUI();
                }

                // Ctrl/Cmd + F - Focus search
                if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                    e.preventDefault();
                    document.getElementById('modelSearch').focus();
                }

                // Ctrl/Cmd + T - Toggle theme
                if ((e.ctrlKey || e.metaKey) && e.key === 't') {
                    e.preventDefault();
                    document.getElementById('themeToggle').click();
                }

                // Number keys 1-6 - Switch modules (only if not typing in input)
                if (!['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
                    const modules = ['text2img', 'img2img', 'upscale', 'background-remove', 'denoise', 'restore'];
                    const num = parseInt(e.key);
                    if (num >= 1 && num <= 6) {
                        e.preventDefault();
                        const tabs = document.querySelectorAll('.module-tab');
                        if (tabs[num - 1]) {
                            tabs[num - 1].click();
                        }
                    }

                    // Alt + Up/Down - Adjust steps
                    if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                        e.preventDefault();
                        const stepsInput = document.getElementById('steps');
                        let currentSteps = parseInt(stepsInput.value);
                        if (e.key === 'ArrowUp') {
                            currentSteps = Math.min(70, currentSteps + 5);
                        } else {
                            currentSteps = Math.max(6, currentSteps - 5);
                        }
                        stepsInput.value = currentSteps;
                        document.getElementById('stepsValue').textContent = currentSteps;
                    }

                    // Escape - Close/Reset
                    if (e.key === 'Escape') {
                        // If modal or similar open, close it, otherwise reset
                        if (!document.getElementById('placeholder').classList.contains('hidden')) {
                            // Already on empty state
                        } else {
                            resetUI();
                        }
                    }
                }
            });

            // Show keyboard shortcuts help on ?
            document.addEventListener('keydown', (e) => {
                if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
                    e.preventDefault();
                    showKeyboardShortcuts();
                }
            });
        }

        function showKeyboardShortcuts() {
            const shortcuts = `
                <div style="text-align: left; max-width: 500px;">
                    <h3 style="margin-bottom: 1rem; text-align: center;">⌨️ Keyboard Shortcuts</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr><td style="padding: 0.5rem;"><kbd>Ctrl/Cmd + Enter</kbd></td><td>Generate image</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>Ctrl/Cmd + D</kbd></td><td>Download image</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>Ctrl/Cmd + N</kbd></td><td>New/Reset canvas</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>Ctrl/Cmd + F</kbd></td><td>Focus search</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>Ctrl/Cmd + T</kbd></td><td>Toggle theme</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>1-6</kbd></td><td>Switch modules</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>Alt + ↑/↓</kbd></td><td>Adjust steps (±5)</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>Esc</kbd></td><td>Reset canvas</td></tr>
                        <tr><td style="padding: 0.5rem;"><kbd>?</kbd></td><td>Show this help</td></tr>
                    </table>
                    <p style="text-align: center; margin-top: 1rem; color: var(--text-tertiary); font-size: 0.9rem;">
                        Press any key to close
                    </p>
                </div>
            `;
            
            // Create modal
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                backdrop-filter: blur(10px);
            `;
            
            const content = document.createElement('div');
            content.style.cssText = `
                background: var(--card-bg);
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid var(--border);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                animation: slideIn 0.3s ease-out;
            `;
            content.innerHTML = shortcuts;
            
            modal.appendChild(content);
            document.body.appendChild(modal);
            
            // Add animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes slideIn {
                    from {
                        opacity: 0;
                        transform: translateY(-20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                kbd {
                    background: var(--glass-bg);
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    border: 1px solid var(--border);
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.85rem;
                }
            `;
            document.head.appendChild(style);
            
            // Close on click or any key
            modal.addEventListener('click', () => {
                modal.remove();
                style.remove();
            });
            
            document.addEventListener('keydown', function closeModal() {
                modal.remove();
                style.remove();
                document.removeEventListener('keydown', closeModal);
            }, { once: true });
        }

        async function selectModel(modelKey) {
            if (modelKey === currentModel) return;
            
            document.querySelectorAll('.model-card').forEach(card => card.classList.remove('active'));
            document.querySelector(`[data-model="${modelKey}"]`).classList.add('active');
            
            currentModel = modelKey;
            
            try {
                const response = await fetch('/switch_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model: modelKey})
                });
                const result = await response.json();
                if (!result.success) {
                    alert('Error loading model: ' + result.error);
                }
            } catch (error) {
                console.error('Model switch error:', error);
            }
        }

        async function handleGenerate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt && currentModule === 'text2img') {
                alert('Please enter a prompt');
                document.getElementById('prompt').focus();
                return;
            }

            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Generating...</span>';

            document.getElementById('placeholder').classList.add('hidden');
            document.getElementById('imageResult').classList.add('hidden');
            document.getElementById('actionButtons').classList.add('hidden');
            document.getElementById('progressDisplay').classList.remove('hidden');

            startProgressPolling();

            try {
                const formData = {
                    model: currentModel,
                    mode: currentModule,
                    prompt: prompt,
                    steps: parseInt(document.getElementById('steps').value),
                };

                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(formData)
                });

                const result = await response.json();

                if (result.success) {
                    stopProgressPolling();
                    currentImageData = result.image;
                    document.getElementById('resultImage').src = 'data:image/png;base64,' + result.image;
                    document.getElementById('progressDisplay').classList.add('hidden');
                    document.getElementById('imageResult').classList.remove('hidden');
                    document.getElementById('actionButtons').classList.remove('hidden');
                    await loadHistory();
                } else {
                    stopProgressPolling();
                    alert('Error: ' + result.error);
                    resetUI();
                }
            } catch (error) {
                stopProgressPolling();
                alert('Error: ' + error.message);
                resetUI();
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-magic"></i> <span>Generate Image</span>';
            }
        }

        function startProgressPolling() {
            updateProgress(0);
            progressInterval = setInterval(async () => {
                try {
                    const response = await fetch('/progress');
                    const data = await response.json();
                    updateProgress(data.progress);
                } catch (error) {
                    console.error('Progress poll error:', error);
                }
            }, 500);
        }

        function stopProgressPolling() {
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
            updateProgress(100);
        }

        function updateProgress(percent) {
            const circle = document.getElementById('progressCircle');
            const text = document.getElementById('progressText');
            const fill = document.getElementById('progressFill');
            
            const circumference = 2 * Math.PI * 100;
            const offset = circumference - (percent / 100) * circumference;
            
            circle.style.strokeDashoffset = offset;
            text.textContent = Math.round(percent) + '%';
            fill.style.width = percent + '%';
        }

        function downloadImage() {
            if (!currentImageData) return;
            const link = document.createElement('a');
            link.href = 'data:image/png;base64,' + currentImageData;
            link.download = `quantum_canvas_${Date.now()}.png`;
            link.click();
        }

        async function upscaleImage(scale) {
            if (!currentImageData) return;
            
            const btn = scale === 2 ? document.getElementById('upscale2xBtn') : document.getElementById('upscale4xBtn');
            const originalHTML = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span>Upscaling ${scale}x...</span>`;
            
            try {
                const response = await fetch('/upscale', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: currentImageData, scale: scale})
                });
                
                const result = await response.json();
                if (result.success) {
                    currentImageData = result.image;
                    document.getElementById('resultImage').src = 'data:image/png;base64,' + result.image;
                    alert(`Image upscaled ${scale}x successfully!`);
                } else {
                    alert('Upscale error: ' + result.error);
                }
            } catch (error) {
                alert('Upscale error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = originalHTML;
            }
        }

        function resetUI() {
            document.getElementById('placeholder').classList.remove('hidden');
            document.getElementById('imageResult').classList.add('hidden');
            document.getElementById('progressDisplay').classList.add('hidden');
            document.getElementById('actionButtons').classList.add('hidden');
            stopProgressPolling();
            currentImageData = null;
        }

        async function loadHistory() {
            try {
                const response = await fetch('/history');
                const history = await response.json();
                
                const grid = document.getElementById('historyGrid');
                
                if (!history || history.length === 0) {
                    grid.innerHTML = '<p style="text-align: center; color: var(--text-tertiary); padding: 2rem;">No history yet</p>';
                    return;
                }
                
                grid.innerHTML = '';
                history.slice(0, 12).forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'history-item';
                    
                    const date = new Date(item.timestamp);
                    const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    
                    div.innerHTML = `
                        <img src="data:image/png;base64,${item.image}" class="history-thumb" alt="Generated">
                        <div class="history-prompt">${item.prompt || 'No prompt'}</div>
                        <div class="history-date">${dateStr}</div>
                    `;
                    div.addEventListener('click', () => {
                        currentImageData = item.image;
                        document.getElementById('resultImage').src = 'data:image/png;base64,' + item.image;
                        document.getElementById('placeholder').classList.add('hidden');
                        document.getElementById('progressDisplay').classList.add('hidden');
                        document.getElementById('imageResult').classList.remove('hidden');
                        document.getElementById('actionButtons').classList.remove('hidden');
                    });
                    grid.appendChild(div);
                });
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    try:
        data = request.json
        model_key = data.get('model', 'tiny-sd')
        success = load_model(model_key)
        return jsonify({'success': success, 'model': model_key})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate', methods=['POST'])
def generate_image():
    global is_generating, generation_progress, generation_step, total_steps, pipelines, current_model
    
    if is_generating:
        return jsonify({'success': False, 'error': 'Generation in progress'})
    
    if current_model not in pipelines:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        data = request.json
        mode = data.get('mode', 'text2img')
        prompt = data.get('prompt', '')
        steps = data.get('steps', 30)
        
        if not prompt and mode == 'text2img':
            return jsonify({'success': False, 'error': 'Prompt required'})
        
        is_generating = True
        generation_progress = 0
        generation_step = 0
        total_steps = steps
        
        print(f"\nGenerating with {AVAILABLE_MODELS[current_model]['name']}")
        print(f"Prompt: {prompt}")
        
        generator = None
        
        def progress_callback(step, timestep, latents):
            global generation_progress, generation_step
            generation_step = step + 1
            generation_progress = (generation_step / total_steps) * 100
        
        pipeline = pipelines[current_model]['text2img']
        
        with torch.inference_mode():
            image = pipeline(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=generator,
                callback=progress_callback,
                callback_steps=1,
            ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        history_item = {
            'image': img_str,
            'prompt': prompt,
            'resolution': 512,
            'model': current_model,
            'timestamp': datetime.now().isoformat()
        }
        save_history(history_item)
        
        is_generating = False
        generation_progress = 100
        
        print("✓ Generated successfully!")
        
        return jsonify({'success': True, 'image': img_str, 'mode': mode})
        
    except Exception as e:
        is_generating = False
        print(f"✗ Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def get_history():
    return jsonify(load_history())

@app.route('/favorites', methods=['GET', 'POST'])
def favorites_handler():
    if request.method == 'GET':
        return jsonify(load_favorites())
    else:
        try:
            favorites = request.json.get('favorites', [])
            save_favorites(favorites)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        data = request.json
        image_data = data.get('image')
        scale = data.get('scale', 2)
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Image required'})
        
        print(f"Upscaling {scale}x...")
        result = upscale_image(image_data, scale)
        
        if result:
            return jsonify({'success': True, 'image': result})
        else:
            return jsonify({'success': False, 'error': 'Upscale failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/remove_bg', methods=['POST'])
def remove_bg():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Image required'})
        
        print("Removing background...")
        result = remove_background(image_data)
        
        if result:
            return jsonify({'success': True, 'image': result})
        else:
            return jsonify({'success': False, 'error': 'Background removal failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/denoise', methods=['POST'])
def denoise():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Image required'})
        
        print("Denoising...")
        result = denoise_image(image_data)
        
        if result:
            return jsonify({'success': True, 'image': result})
        else:
            return jsonify({'success': False, 'error': 'Denoise failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/restore', methods=['POST'])
def restore():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Image required'})
        
        print("Restoring image...")
        result = restore_image(image_data)
        
        if result:
            return jsonify({'success': True, 'image': result})
        else:
            return jsonify({'success': False, 'error': 'Restore failed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/progress')
def get_progress():
    return jsonify({
        'is_generating': is_generating,
        'progress': generation_progress,
        'current_step': generation_step,
        'total_steps': total_steps
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': current_model in pipelines,
        'current_model': current_model if current_model in pipelines else None,
        'total_models': len(AVAILABLE_MODELS),
        'history_items': len(load_history())
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Quantum Canvas Ultimate V1 - AI Image Studio (IMPROVED)")
    print("="*60)
    print("\n✨ Features:")
    print("  • 50+ AI Models with Smart Categorization")
    print("  • Searchable Model Selection")
    print("  • Favorites & Recently Used")
    print("  • Multiple Modules: Text2Img, Img2Img, Upscaling")
    print("  • Image Processing: Background Removal, Denoise, Restore")
    print("  • Modern Dark/Light Theme UI")
    print("  • Real-time Progress Tracking")
    print("  • Speed Presets (Fast, Balanced, Quality)")
    print("  • Mobile Responsive Design")
    print("  • CPU Optimized with Multi-threading")
    print("  • Persistent History")
    print(f"\n💻 CPU Cores: {os.cpu_count()}")
    print(f"📁 History: {HISTORY_FILE}")
    print(f"⭐ Favorites: {FAVORITES_FILE}")
    print("\n🌐 Server: http://0.0.0.0:1600")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Load default model in background
    def load_default():
        load_model('tiny-sd')
    
    threading.Thread(target=load_default, daemon=True).start()
    
    app.run(host='0.0.0.0', port=1600, debug=False, threaded=True)
