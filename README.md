# Rice Image Classification Project

## Project Overview
A deep learning project that classifies 5 different types of rice grains using Convolutional Neural Networks (CNN). The model achieves 99.11% accuracy in classifying rice images across different varieties.

## Dataset
- Source: [Rice Image Dataset (Kaggle)](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)
- Total Images: 75,000
- Distribution: 15,000 images per class, ensuring balanced representation
- Image Format: Grayscale, 150x150 pixels

## Model Architecture
The CNN model consists of:
- 3 Convolutional blocks with:
  - Conv2D layers (32 filters)
  - BatchNormalization
  - MaxPooling2D
- Fully Connected Layers:
  - Dense (128 units) with ReLU
  - Dropout (0.5)
  - Dense (64 units) with ReLU
  - Dropout (0.3)
  - Output layer (5 units) with Softmax

## Performance
- Validation Accuracy: 99.11%
- F1-Score: >0.98 for all classes
- Consistent performance across all rice varieties
- Evaluated using confusion matrix and detailed classification reports

## Model Deployment
The trained model is exported in multiple formats for different deployment scenarios:
- TensorFlow SavedModel
- TensorFlow.js (for web deployment)
- TensorFlow Lite (for mobile/edge devices)

## Technical Implementation
### Libraries Used
- TensorFlow/Keras for model development
- Scikit-learn for data splitting and evaluation
- OpenCV and PIL for image processing
- Matplotlib and Seaborn for visualization

### Data Processing
- Image normalization (rescaling to 0-1)
- Train/validation/test split
- Data augmentation capabilities
- Grayscale conversion for efficient processing

## Author
- **Name:** Tok Se Ka
- **Email:** sekacoding@gmail.com
- **Dicoding ID:** MC185D5Y2370

## Project Structure
```
├── notebook.ipynb          # Main development notebook
├── requirements.txt        # Project dependencies
├── saved_model/           # TensorFlow SavedModel format
├── tfjs_model/            # TensorFlow.js model files
└── tflite/               # TensorFlow Lite model
```

## Model Training
- Batch Size: 32
- Optimizer: RMSprop
- Loss Function: Categorical Crossentropy
- Training Monitoring: Early Stopping and Model Checkpoint
- Class weights: Balanced for optimal performance