"""
Notebook API Server for ReColor Frontend Integration.
Provides API endpoints to interact with Jupyter notebooks and K-means functionality.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import logging
import numpy as np
import cv2
import base64
from typing import Dict, Any, List
from PIL import Image
import io

from kmeans_color_detector import KMeansColorDetector, get_kmeans_color_info, get_multiple_kmeans_colors
from dataset_manager import ColorDatasetGenerator
from improved_color_detection import get_improved_color_info
from image_processor import ImageProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DATASETS_DIR = os.path.join(BACKEND_DIR, "color_datasets")
NOTEBOOKS_DIR = BACKEND_DIR

# K-means model path
KMEANS_MODEL_PATH = os.path.join(BACKEND_DIR, "kmeans_lab_model.pkl")

# Initialize K-means detector
kmeans_detector = None
if os.path.exists(KMEANS_MODEL_PATH):
    kmeans_detector = KMeansColorDetector(model_path=KMEANS_MODEL_PATH)
    logger.info(f"K-means model loaded from {KMEANS_MODEL_PATH}")
else:
    kmeans_detector = KMeansColorDetector()
    logger.info("Using default K-means model")

# Initialize dataset generator
dataset_generator = ColorDatasetGenerator(base_dir=DATASETS_DIR)

# Initialize image processor
image_processor = ImageProcessor(save_images=True, output_dir=os.path.join(BACKEND_DIR, "processed_images"))


@app.route('/api/notebooks/list', methods=['GET'])
def list_notebooks():
    """List available Jupyter notebooks."""
    try:
        notebooks = []
        
        # Find all .ipynb files in the backend directory
        for file in os.listdir(NOTEBOOKS_DIR):
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(NOTEBOOKS_DIR, file)
                notebooks.append({
                    'name': file,
                    'path': notebook_path,
                    'size': os.path.getsize(notebook_path),
                    'modified': os.path.getmtime(notebook_path)
                })
        
        return jsonify({
            'success': True,
            'notebooks': notebooks,
            'count': len(notebooks)
        })
        
    except Exception as e:
        logger.error(f"Error listing notebooks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/notebooks/content/<notebook_name>', methods=['GET'])
def get_notebook_content(notebook_name):
    """Get notebook content."""
    try:
        notebook_path = os.path.join(NOTEBOOKS_DIR, notebook_name)
        
        if not os.path.exists(notebook_path):
            return jsonify({
                'success': False,
                'error': 'Notebook not found'
            }), 404
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        return jsonify({
            'success': True,
            'content': content,
            'name': notebook_name
        })
        
    except Exception as e:
        logger.error(f"Error reading notebook: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/kmeans/detect', methods=['POST'])
def kmeans_detect_color():
    """Detect color using K-means clustering."""
    try:
        # Get image from request
        if 'image' not in request.files:
            # Try to get base64 encoded image from JSON
            data = request.get_json()
            if 'image' in data:
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_np = np.array(image)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No image provided'
                }), 400
        else:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
            image_np = np.array(image)
        
        # Detect color using K-means
        color_info = kmeans_detector.detect_color_kmeans(image_np)
        
        return jsonify({
            'success': True,
            'color': {
                'name': color_info.name,
                'rgb': color_info.rgb,
                'hex': color_info.hex,
                'lab': color_info.lab,
                'hsv': color_info.hsv,
                'confidence': color_info.confidence,
                'cluster_id': color_info.cluster_id,
                'detection_method': color_info.detection_method
            }
        })
        
    except Exception as e:
        logger.error(f"Error in K-means detection: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/kmeans/detect-multiple', methods=['POST'])
def kmeans_detect_multiple_colors():
    """Detect multiple dominant colors using K-means."""
    try:
        # Get parameters
        data = request.get_json() if request.is_json else {}
        top_n = int(data.get('top_n', 3))
        
        # Get image
        if 'image' not in request.files:
            if 'image' in data:
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_np = np.array(image)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No image provided'
                }), 400
        else:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
            image_np = np.array(image)
        
        # Detect multiple colors
        colors = kmeans_detector.detect_multiple_colors(image_np, top_n=top_n)
        
        colors_data = [{
            'name': c.name,
            'rgb': c.rgb,
            'hex': c.hex,
            'lab': c.lab,
            'hsv': c.hsv,
            'confidence': c.confidence,
            'cluster_id': c.cluster_id
        } for c in colors]
        
        return jsonify({
            'success': True,
            'colors': colors_data,
            'count': len(colors_data)
        })
        
    except Exception as e:
        logger.error(f"Error in multiple color detection: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/kmeans/train', methods=['POST'])
def kmeans_train():
    """Train K-means model on uploaded image."""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)
        
        # Train model
        success = kmeans_detector.train_on_image(image_np, save_path=KMEANS_MODEL_PATH)
        
        return jsonify({
            'success': success,
            'message': 'K-means model trained successfully' if success else 'Training failed',
            'model_path': KMEANS_MODEL_PATH if success else None
        })
        
    except Exception as e:
        logger.error(f"Error training K-means: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dataset/generate', methods=['POST'])
def generate_dataset():
    """Generate synthetic color dataset."""
    try:
        data = request.get_json()
        
        dataset_type = data.get('type', 'rgb_hsv')  # 'rgb_hsv' or 'varied'
        samples_per_family = int(data.get('samples_per_family', 100))
        output_dir = data.get('output_dir', f'dataset_{dataset_type}')
        
        if dataset_type == 'rgb_hsv':
            success = dataset_generator.generate_rgb_hsv_dataset(
                output_dir=output_dir,
                samples_per_family=samples_per_family
            )
        elif dataset_type == 'varied':
            success = dataset_generator.generate_varied_dataset(
                output_dir=output_dir,
                samples_per_family=samples_per_family
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid dataset type. Use "rgb_hsv" or "varied"'
            }), 400
        
        return jsonify({
            'success': success,
            'message': f'Dataset generated successfully' if success else 'Generation failed',
            'output_dir': os.path.join(DATASETS_DIR, output_dir)
        })
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dataset/stats/<dataset_name>', methods=['GET'])
def get_dataset_stats(dataset_name):
    """Get dataset statistics."""
    try:
        stats = dataset_generator.get_dataset_stats(dataset_name)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'dataset_name': dataset_name
        })
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/color/compare', methods=['POST'])
def compare_detection_methods():
    """Compare different color detection methods."""
    try:
        # Get image
        if 'image' not in request.files:
            data = request.get_json()
            if 'image' in data:
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_np = np.array(image)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No image provided'
                }), 400
        else:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
            image_np = np.array(image)
        
        # K-means detection
        kmeans_info = kmeans_detector.detect_color_kmeans(image_np)
        
        # Improved HSV detection
        improved_info = get_improved_color_info(image_np)
        
        return jsonify({
            'success': True,
            'methods': {
                'kmeans': {
                    'name': kmeans_info.name,
                    'rgb': kmeans_info.rgb,
                    'hex': kmeans_info.hex,
                    'confidence': kmeans_info.confidence,
                    'method': 'K-means LAB clustering'
                },
                'improved_hsv': {
                    'name': improved_info['name'],
                    'rgb': improved_info['rgb'],
                    'hex': improved_info['hex'],
                    'confidence': improved_info['confidence'],
                    'method': 'HSV range-based detection'
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error comparing detection methods: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/image/process', methods=['POST'])
def process_real_image():
    """Process real camera/uploaded image with before/after visualization."""
    try:
        # Get parameters
        data = request.form if request.files else request.get_json()
        n_colors = int(data.get('n_colors', 9))
        
        # Get image
        if 'image' not in request.files:
            if data and 'image' in data:
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_np = np.array(image)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No image provided'
                }), 400
        else:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
            image_np = np.array(image)
        
        # Process image with before/after
        result = image_processor.process_camera_image(image_np, n_colors=n_colors)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing real image: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/image/history', methods=['GET'])
def get_processing_history():
    """Get image processing history."""
    try:
        limit = int(request.args.get('limit', 10))
        history = image_processor.get_processing_history(limit=limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting processing history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'components': {
            'kmeans_detector': kmeans_detector is not None,
            'dataset_generator': dataset_generator is not None,
            'image_processor': image_processor is not None,
            'notebooks_available': len([f for f in os.listdir(NOTEBOOKS_DIR) if f.endswith('.ipynb')])
        }
    })


@app.route('/', methods=['GET'])
def index():
    """Index route."""
    return jsonify({
        'service': 'ReColor Notebook API',
        'version': '2.0.0',
        'endpoints': {
            'notebooks': {
                'list': '/api/notebooks/list',
                'content': '/api/notebooks/content/<notebook_name>'
            },
            'kmeans': {
                'detect': '/api/kmeans/detect',
                'detect_multiple': '/api/kmeans/detect-multiple',
                'train': '/api/kmeans/train'
            },
            'image': {
                'process': '/api/image/process',
                'history': '/api/image/history'
            },
            'dataset': {
                'generate': '/api/dataset/generate',
                'stats': '/api/dataset/stats/<dataset_name>'
            },
            'utils': {
                'compare': '/api/color/compare',
                'health': '/api/health'
            }
        }
    })


if __name__ == '__main__':
    logger.info("Starting ReColor Notebook API Server...")
    logger.info(f"Backend directory: {BACKEND_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Datasets directory: {DATASETS_DIR}")
    
    # Create necessary directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)
