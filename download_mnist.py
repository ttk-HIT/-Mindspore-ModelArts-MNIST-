import os
import struct
import tensorflow as tf

# Create directories
os.makedirs('MNIST_Data/train', exist_ok=True)
os.makedirs('MNIST_Data/test', exist_ok=True)

# Download MNIST using TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def write_labels(filename, labels):
    """Write labels in IDX format"""
    with open(filename, 'wb') as f:
        # Magic number (2049 for labels), number of labels
        f.write(struct.pack('>II', 2049, len(labels)))
        f.write(labels.tobytes())

def write_images(filename, images):
    """Write images in IDX format"""
    with open(filename, 'wb') as f:
        # Magic number (2051 for images), number of images, rows, cols
        f.write(struct.pack('>IIII', 2051, len(images), 28, 28))
        f.write(images.tobytes())

# Write train files
print("Writing train images...")
write_images('MNIST_Data/train/train-images-idx3-ubyte', x_train)
print("Writing train labels...")
write_labels('MNIST_Data/train/train-labels-idx1-ubyte', y_train)

# Write test files
print("Writing test images...")
write_images('MNIST_Data/test/t10k-images-idx3-ubyte', x_test)
print("Writing test labels...")
write_labels('MNIST_Data/test/t10k-labels-idx1-ubyte', y_test)

print("MNIST dataset downloaded and saved successfully!")
