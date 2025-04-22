# Food41 Image Classification

This project implements a deep learning model for classifying food images from the Food-41 dataset using TensorFlow and MobileNetV2. The model is designed to recognize 41 different food categories with high accuracy, leveraging transfer learning and GPU optimization for efficient training.

## Features
- **Pre-trained Model**: Utilizes MobileNetV2 as the base model for transfer learning.
- **Data Augmentation**: Includes techniques like rotation, zoom, and flipping to improve model robustness.
- **GPU Optimization**: Configures TensorFlow to dynamically allocate GPU memory, enhancing training performance.
- **Callbacks**: Implements early stopping, model checkpointing, learning rate reduction, and TensorBoard logging for better training control.
- **Flexible Environment**: Supports Google Colab, Kaggle, and local environments with dynamic path handling.
- **Visualization**: Plots training and validation accuracy/loss for performance analysis.
- **Inference**: Provides functionality to predict food categories from new images with confidence scores.

## Requirements
- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- Google Colab (optional, for cloud-based training)
- Kaggle (optional, for dataset access)
- GPU (recommended for faster training)

## Usage
1. **Train the Model**:
   Run the main script to train the model:
   ```bash
   python trainer.py
   ```
   - The script automatically detects the environment (Colab, Kaggle, or local) and sets up paths.
   - Training uses the Food-41 dataset, with 80% for training and 20% for validation.
   - Model checkpoints and logs are saved in the specified `SAVE_DIR`.

2. **Monitor Training**:
   Use TensorBoard to visualize training progress:
   ```bash
   tensorboard --logdir food41_model/logs
   ```

3. **Make Predictions**:
   After training, the script tests the model on a sample image and outputs the predicted class and confidence.

## Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet, frozen layers).
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (1024 units, ReLU) with 50% Dropout
  - Dense (512 units, ReLU) with 30% Dropout
  - Dense (41 units, softmax)
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy

## Improving Accuracy with More GPU Support and Time
The model's accuracy can be significantly enhanced with additional computational resources and training time:
- **More GPU Power**:
  - Training on multiple high-end GPUs (e.g., NVIDIA A100 or V100) reduces computation time and allows for larger batch sizes (e.g., `BATCH_SIZE=512` or higher), improving gradient stability and convergence.
  - Multi-GPU setups with TensorFlow's `MirroredStrategy` can distribute training across devices, enabling faster processing of the Food-41 dataset (101,000+ images).
  - Example modification for multi-GPU support:
    ```python
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(num_classes)
        model.compile(...)
    ```
- **Extended Training Time**:
  - Increasing `EPOCHS` (e.g., from 20 to 50 or 100) allows the model to fine-tune weights further, potentially improving accuracy.
  - Fine-tuning the MobileNetV2 base layers (by setting `layer.trainable=True` for some layers) can adapt the model to the Food-41 dataset, but requires more time and compute power.
- **Expected Impact**:
  - With a single GPU (e.g., NVIDIA RTX 3060), the model achieves ~85-90% validation accuracy after 20 epochs.
  - With multiple GPUs and 50+ epochs, accuracy could exceed 92-95%, especially with fine-tuning and hyperparameter optimization (e.g., learning rate schedules).
  - Additional data augmentation and larger datasets can further boost performance but require proportional GPU resources.

## Results
# version 1:
![Alt text](images/v1.png)

## Future Improvements
- Fine-tune MobileNetV2 layers for better feature extraction.
- Experiment with other architectures (e.g., EfficientNet, ResNet).
- Implement cross-validation for more robust performance metrics.
- Add support for real-time inference with a web or mobile interface.



