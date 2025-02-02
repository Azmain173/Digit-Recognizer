# Digit-Recognizer

## Large Files
The dataset is too large for GitHub. You can download train data from [Google Drive][https://drive.google.com/file/d/1PI52lEj6w5bJggyS0XXtnwjqdl5vJU-3/view?usp=sharing] and test data [https://drive.google.com/file/d/1BpEbKyrymu1LURs7k-e7kDN52KXX2Xx8/view?usp=sharing]


# Digit Recognizer - Kaggle Competition

## 📝 Project Overview
This project aims to build a **Digit Recognizer** using **Deep Learning (Neural Networks)** to classify handwritten digits from the famous **MNIST dataset**. The goal is to achieve high accuracy on the Kaggle competition dataset.

## 📂 Dataset
- **Train Data**: (42000, 785) → 42,000 images with 784 pixel values + 1 label.
- **Test Data**: (28000, 784) → 28,000 images with 784 pixel values.
- **Each image** is **28x28 pixels** in grayscale, flattened into a 784-dimensional vector.
- Dataset available on **Kaggle**: [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/)

## 🛠️ Tech Stack
- **Python**
- **NumPy, Pandas, Matplotlib, Seaborn** (Data Processing & Visualization)
- **TensorFlow, Keras** (Deep Learning Model)
- **Scikit-learn** (Evaluation & Metrics)

## 🏗️ Model Architecture
A deep neural network was implemented with the following layers:
1. **Input Layer**: 784 neurons (flattened 28x28 images)
2. **Hidden Layers**:
   - Dense(512, activation='relu')
   - Dropout(0.2)
   - Dense(256, activation='relu')
   - Dropout(0.2)
3. **Output Layer**: Dense(10, activation='softmax')

## 📌 Steps to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn
```

### 2️⃣ Load Dataset
```python
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```

### 3️⃣ Preprocess Data
```python
X_train = train_df.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df['label'].values
X_test = test_df.values.reshape(-1, 28, 28, 1) / 255.0
```

### 4️⃣ Train the Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

### 5️⃣ Evaluate the Model
```python
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Training Accuracy: {accuracy * 100:.2f}%')
```

### 6️⃣ Predict on Test Data & Submit to Kaggle
```python
y_pred = model.predict(X_test)
pred_labels = y_pred.argmax(axis=1)
submission = pd.DataFrame({'ImageId': range(1, len(pred_labels)+1), 'Label': pred_labels})
submission.to_csv('submission.csv', index=False)
```

## 📊 Model Evaluation
- **Accuracy**: Achieved **>98%** accuracy on the training set.
- **Confusion Matrix**:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## 🚀 Future Improvements
- **Use CNN (Convolutional Neural Networks) for better feature extraction**.
- **Hyperparameter tuning (batch size, learning rate, dropout rate)**.
- **Data Augmentation to enhance model generalization**.

## 📌 Contributors
- **[Your Name]** - kaggle.com/[Your Kaggle Profile]

## 📜 License
This project is open-source under the MIT License.

---

🔥 **Happy Coding! Let’s win this Kaggle challenge!** 🚀

# Digit Recognizer - Questions & Answers

## 1️⃣ What is the Digit Recognizer project?
This project focuses on classifying handwritten digits (0-9) using deep learning models, particularly neural networks, on the MNIST dataset provided by Kaggle.

## 2️⃣ What dataset is used in this project?
The dataset is from the Kaggle **Digit Recognizer** competition, which consists of:
- **Train Data**: 42,000 images with 784 pixel values + 1 label.
- **Test Data**: 28,000 images with 784 pixel values.

## 3️⃣ How are the images formatted?
Each image is **28x28 pixels**, represented as a **784-dimensional** vector in the dataset.

## 4️⃣ How is the dataset preprocessed?
- The pixel values are **normalized** to the range [0,1].
- The images are **reshaped** into 28x28x1 format for CNN models.
- Labels are extracted for supervised learning.

## 5️⃣ What deep learning model is used?
A **fully connected neural network (DNN)** with:
1. **Input Layer**: 784 neurons (flattened images)
2. **Hidden Layers**:
   - Dense(512, activation='relu')
   - Dropout(0.2)
   - Dense(256, activation='relu')
   - Dropout(0.2)
3. **Output Layer**: Dense(10, activation='softmax')

## 6️⃣ What optimizer is used?
Adam optimizer (`learning_rate=0.001`) is used for efficient training.

## 7️⃣ What loss function is used?
Sparse Categorical Crossentropy (`sparse_categorical_crossentropy`), as we have multi-class classification.

## 8️⃣ How is the model trained?
Using **TensorFlow/Keras** with:
```python
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

## 9️⃣ How is the model evaluated?
The model is evaluated using:
- **Training accuracy**
- **Validation accuracy**
- **Confusion matrix**

## 🔟 How to visualize the predictions?
```python
import matplotlib.pyplot as plt
import numpy as np

# Plot 10 images with predictions
def plot_predictions(images, labels, predictions):
    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].reshape(28,28), cmap='gray')
        plt.title(f'Pred: {np.argmax(predictions[i])}')
        plt.axis('off')
    plt.show()
```

## 1️⃣1️⃣ How can we improve accuracy?
- Use **Convolutional Neural Networks (CNNs)** instead of dense layers.
- **Hyperparameter tuning** (learning rate, dropout, batch size, etc.).
- **Data augmentation** (rotate, shift, scale, etc.).

## 1️⃣2️⃣ What is dropout and why is it used?
Dropout is a regularization technique that **randomly drops neurons** during training to prevent overfitting.

## 1️⃣3️⃣ How to save and load the trained model?
```python
model.save('digit_recognizer.h5')
model = tf.keras.models.load_model('digit_recognizer.h5')
```

## 1️⃣4️⃣ How to make predictions on test data?
```python
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)
```

## 1️⃣5️⃣ How to submit results to Kaggle?
```python
submission = pd.DataFrame({'ImageId': range(1, len(pred_labels)+1), 'Label': pred_labels})
submission.to_csv('submission.csv', index=False)
```

## 1️⃣6️⃣ How to implement a confusion matrix?
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_true, pred_labels)
sns.heatmap(cm, annot=True, fmt='d')
```

## 1️⃣7️⃣ What is the role of activation functions?
They introduce **non-linearity** into the model, enabling it to learn complex patterns.

## 1️⃣8️⃣ Why use Softmax in the output layer?
Softmax converts logits into **probabilities**, making it suitable for multi-class classification.

## 1️⃣9️⃣ What is batch size, and how does it affect training?
Batch size is the **number of samples** processed before updating the model weights. Larger batch sizes can improve stability but require more memory.

## 2️⃣0️⃣ How does learning rate affect training?
- **Too high**: Training diverges.
- **Too low**: Training becomes slow.
- **Optimal**: Fast convergence without instability.

## 2️⃣1️⃣ How can we use CNNs instead of DNNs?
Replace dense layers with **Conv2D, MaxPooling2D**:
```python
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
```

## 2️⃣2️⃣ How does early stopping help in training?
It stops training when validation loss stops improving, preventing overfitting.

## 2️⃣3️⃣ What is the difference between validation and test data?
- **Validation data**: Used during training to tune hyperparameters.
- **Test data**: Used only for final model evaluation.

## 2️⃣4️⃣ How to implement real-time digit recognition?
Use **OpenCV** to capture and preprocess handwritten digits from a webcam.

## 2️⃣5️⃣ What is the purpose of normalizing images?
Normalization scales pixel values to [0,1], making the model more stable and improving convergence.

## 2️⃣6️⃣ Why do we use one-hot encoding in classification?
It represents categorical labels in a **binary format**, making them easier to process by the model.

## 2️⃣7️⃣ What is data augmentation, and how can it help?
Data augmentation artificially increases the dataset size by adding **rotations, shifts, and flips**, improving generalization.

## 2️⃣8️⃣ How can we visualize feature maps in CNNs?
Use **Matplotlib** to display the outputs of convolutional layers.

## 2️⃣9️⃣ What libraries are used in this project?
- **NumPy, Pandas** (Data Handling)
- **Matplotlib, Seaborn** (Visualization)
- **TensorFlow, Keras** (Deep Learning)
- **Scikit-learn** (Evaluation Metrics)

## 3️⃣0️⃣ How to deploy this model?
- Convert the model to **TensorFlow Lite** or **ONNX**.
- Deploy using **Flask/FastAPI** for web-based inference.

---

✅ **End of Q&A**: This document covers all essential concepts of the **Digit Recognizer** project.

