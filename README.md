# Crop Disease AI Classification for Rwanda

## Project Structure
```
CROP-DISEASE-AI-CLASSIFICATION/
│
├── Dataset/
│   └── dataset/
│       ├── test/
│       ├── train/
│       └── validation/
│
├── saved_models/
│   ├── optimized_model.keras
│   └── simple_model.keras
│
├── LICENSE
├── notebook.ipynb
└── README.md
```

## Project Overview
This project aims to develop a machine learning model for classifying crop diseases in Rwanda, focusing on key crops such as potato, tomato, and pepper bell. The project utilizes Convolutional Neural Networks (CNNs) to analyze images of plant leaves and identify various diseases.

## Dataset
The dataset is organized into three subsets:
- `train/`: Used for training the models
- `validation/`: Used for validating the models during training
- `test/`: Used for final evaluation of the models

## Models
Two models were developed and compared:
1. `simple_model.keras`: A basic CNN architecture
2. `optimized_model.keras`: An enhanced CNN with various optimization techniques

## Optimization Techniques

### 1. Data Augmentation
**Principle**: Artificially expanding the training dataset by applying various transformations to the existing images.

**Relevance**: Helps prevent overfitting and improves model generalization by exposing it to a wider variety of image variations.

**Parameters**:
- `rotation_range=20`: Randomly rotate images by up to 20 degrees
- `width_shift_range=0.2`: Randomly shift images horizontally by up to 20% of total width
- `height_shift_range=0.2`: Randomly shift images vertically by up to 20% of total height
- `shear_range=0.2`: Apply shearing transformations
- `zoom_range=0.2`: Randomly zoom images in or out by up to 20%
- `horizontal_flip=True`: Randomly flip images horizontally

**Tuning**: These values were chosen to provide a good balance between data diversity and maintaining the essential features of the crop diseases. The relatively modest ranges prevent excessive distortion that could obscure important disease indicators.

### 2. Batch Normalization
**Principle**: Normalizing the inputs of each layer to reduce internal covariate shift.

**Relevance**: Accelerates training by allowing higher learning rates and reducing the dependence on careful initialization.

**Parameters**: Batch normalization is applied after each convolutional layer with default parameters.

**Tuning**: The default parameters work well for most cases, so no specific tuning was necessary.

### 3. Dropout
**Principle**: Randomly setting a fraction of input units to 0 at each update during training.

**Relevance**: Helps prevent overfitting by reducing complex co-adaptations of neurons.

**Parameters**:
- `rate=0.5`: 50% of the neurons are randomly dropped out during training

**Tuning**: A dropout rate of 0.5 is a common starting point that often works well. It provides a good balance between retaining information and preventing overfitting.

### 4. Adam Optimizer
**Principle**: An adaptive learning rate optimization algorithm.

**Relevance**: Combines the advantages of AdaGrad and RMSProp algorithms to handle sparse gradients on noisy problems.

**Parameters**:
- `learning_rate=0.001`: Initial learning rate

**Tuning**: The default learning rate of 0.001 was used as a starting point. This value typically works well for many problems and can be adjusted if necessary based on the learning curves.

### 5. Early Stopping
**Principle**: Monitoring the model's performance on a validation set and stopping training when performance stops improving.

**Relevance**: Prevents overfitting by stopping the training process before the model starts to memorize the training data.

**Parameters**:
- `monitor='val_loss'`: Monitors the validation loss
- `patience=3`: Number of epochs with no improvement after which training will be stopped

**Tuning**: A patience of 3 epochs was chosen to allow for small fluctuations in validation loss while still stopping training relatively quickly if no improvements are seen.

### 6. Learning Rate Reduction
**Principle**: Reducing the learning rate when a metric has stopped improving.

**Relevance**: Allows the model to fine-tune its weights with smaller steps as it gets closer to the optimal solution.

**Parameters**:
- `monitor='val_loss'`: Monitors the validation loss
- `patience=3`: Number of epochs with no improvement after which learning rate will be reduced
- `factor=0.5`: Factor by which the learning rate will be reduced
- `min_lr=0.00001`: Minimum learning rate

**Tuning**: These values were chosen to allow the model to adapt its learning rate relatively quickly (every 3 epochs) while ensuring that the learning rate doesn't become too small too quickly.

## Results and Comparison

| Metric | Simple Model | Optimized Model |
|--------|--------------|-----------------|
| Training Accuracy | 93.75% | 93.75% |
| Validation Accuracy | 90.91% | 95.45% |
| Test Accuracy | 91.67% | 85.06% |
| Macro F1 Score | Not provided | 0.8363 |
| Micro F1 Score | Not provided | 0.8517 |
| Weighted F1 Score | Not provided | 0.8462 |

## Conclusion
While the optimized model showed improved performance on the validation set, its lower test accuracy suggests potential overfitting. The F1 scores indicate reasonably good performance across classes, but there's room for improvement in generalization and handling potential class imbalances.

## Future Work
- Expand the dataset and ensure balanced representation of all classes
- Experiment with more advanced architectures (e.g., ResNet, EfficientNet)
- Implement transfer learning using pre-trained models
- Conduct more detailed per-class performance analysis
- Evaluate model performance in real-world conditions

## License
See the `LICENSE` file for details.

## Notebook
For detailed implementation and analysis, refer to `notebook.ipynb`.