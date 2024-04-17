import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create the sequential model
model = models.Sequential()

# Add the first convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Add a max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Add another convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Add a flatten layer to convert the 3D output of the previous layer to 1D
model.add(layers.Flatten())

# Add a dense layer
model.add(layers.Dense(64, activation='relu'))

# Add the output layer
model.add(layers.Dense(10))

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Print the summary of the model
model.summary()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)

# Plotting training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Function to unnormalize images
def unnormalize(images):
    return images * 255.0

# Function to display images
def display_images(images, labels, class_names, cols=3, cmap=None):
    plt.figure(figsize=(14, 14))
    plt.suptitle("CIFAR10 Dataset Images", fontsize=20)

    ax = plt.subplots(nrows=3, ncols=3)[1].ravel()
    for i in range(9):
        img = images[i]
        label = labels[i]
        # Ensure label is an integer
        label = int(label)
        ax[i].imshow(unnormalize(img), cmap=cmap)
        ax[i].set_title(class_names[label])
        ax[i].axis('off')
    plt.show()

# Load class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display 9 random images
sample_images = np.random.randint(0, len(train_images), 9)
display_images(train_images[sample_images], train_labels[sample_images], class_names)
