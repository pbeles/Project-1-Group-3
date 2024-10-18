# Project Title: Comparative Analysis of Custom CNN and Pre-trained ResNet50 on CIFAR-10 Dataset

## Project Overview

This project involves a comparative study between two convolutional neural network (CNN) architectures for image classification using the CIFAR-10 dataset: a custom CNN built from scratch and a pre-trained ResNet50 model. The goal is to evaluate the effectiveness of transfer learning compared to developing a model from scratch for a well-known benchmark dataset. The CIFAR-10 dataset consists of 60,000 images divided into 10 classes (e.g., airplane, automobile, bird), and presents challenges such as small image size and inter-class similarity.

We selected the CIFAR-10 dataset due to its manageable complexity and its suitability as a benchmark for CNN models, facilitating meaningful insights into the models' architectural and performance differences.

## Models Employed

### Custom Convolutional Neural Network

- **Architecture**: The custom CNN comprises five main convolutional blocks, each followed by BatchNormalization, MaxPooling, and Dropout layers to prevent overfitting. The output from the convolutional blocks is flattened and passed through fully connected layers.
- **Regularization**: Techniques like Dropout and BatchNormalization were employed to improve model generalization and prevent overfitting.
- **Training**: The model was trained using the Adam optimizer, with data augmentation applied to increase the diversity of the training set and improve model robustness.

### Pre-trained ResNet50

- **Architecture**: ResNet50, a widely used residual network pre-trained on the ImageNet dataset, was employed as a feature extractor. Custom classification layers were added to adapt the model for CIFAR-10 classification.
- **Transfer Learning**: The base layers of the ResNet50 model were frozen initially, with custom layers trained on top. Gradual fine-tuning of the pre-trained layers was performed in stages to adapt the features learned on ImageNet to the CIFAR-10 dataset.
- **Training Strategy**: Training was conducted over three cycles, progressively unfreezing more layers to fine-tune specific features, ensuring optimal performance.

## Data Preprocessing

- **Resizing**: CIFAR-10 images were resized from 32x32 to 64x64 for the custom CNN and 96x96 for the ResNet50.
- **Normalization**: Images were normalized by dividing pixel values by 255 to scale them between 0 and 1, improving convergence during training.
- **Data Augmentation**: Horizontal and vertical flips, zooming, and shifting were applied to augment the dataset and reduce overfitting.

## Training Details

- **Custom CNN**: Utilized the Adam optimizer with an initial learning rate of 0.001, exponential learning rate decay, and early stopping to prevent overfitting.
- **ResNet50**: Used the Adam optimizer with a lower learning rate of 0.0001 and a ReduceLROnPlateau callback for finer adjustments during training. Early stopping was also implemented.

## Results and Evaluation

- **Custom CNN**: Achieved a test accuracy of **92.02%**, with notable performance on most classes, but struggled with visually similar categories like 'cat' and 'dog'.
- **Pre-trained ResNet50**: Outperformed the custom CNN, achieving a test accuracy of **95.26%**. Leveraging pre-trained features from ImageNet significantly boosted the model's performance, especially in differentiating complex patterns.

Both models showed limitations in distinguishing between visually similar classes, such as cats and dogs. The ResNet50 model's deeper architecture and pre-trained weights provided superior feature extraction, resulting in better performance overall.

## Key Insights

1. **Transfer Learning**: Leveraging pre-trained models such as ResNet50 can significantly improve performance with limited training data.
2. **Data Augmentation**: Enhanced the custom CNN's generalization ability and helped reduce overfitting.
3. **Model Complexity vs. Performance**: While ResNet50 offered better accuracy, it came with increased complexity. For simpler applications, the custom CNN might still be preferred.

## Future Work

- **Architectural Exploration**: Investigate other models such as EfficientNet or Vision Transformers for improved performance on CIFAR-10.
- **Hyperparameter Optimization**: Conduct more exhaustive hyperparameter searches to improve custom CNN performance.
- **Ensemble Methods**: Explore combining different CNN architectures to further enhance performance.

## Repository Structure

- `main.py`: Holds model architectures and training for both the custom CNN and ResNet50.
- `presentation.pdf`: Summary presentation of the project highlighting key findings and visualizations
- `report.pdf`: Comprehensive project report containing detailed analysis, model architectures, training process, and evaluation metrics

## Getting Started

### Prerequisites

- Python 3.10+
- TensorFlow
- NumPy, Matplotlib, and other common ML libraries (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cifar10-cnn-analysis.git

# Change into the directory
cd cifar10-cnn-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Models

To run the entire pipeline, including data preparation, training both models, and evaluating the results, execute the following command:

```bash
python main.py
```

## Acknowledgments

- **CIFAR-10 Dataset**: Provided by the Canadian Institute For Advanced Research.
- **ResNet50 Model**: Pre-trained weights sourced from ImageNet.

