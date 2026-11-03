import os
import sys
import io
import time
import zipfile
import urllib.request
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow Version:", tf.__version__)
print("Python Version:", sys.version)

# 1. Test MNIST loading
mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
mnist_path = "mnist.npz"
if not os.path.exists(mnist_path):
    print("Downloading MNIST...")
    urllib.request.urlretrieve(mnist_url, mnist_path)

with np.load(mnist_path) as f:
    x_train_mnist, y_train_mnist = f['x_train'], f['y_train']
    x_test_mnist, y_test_mnist = f['x_test'], f['y_test']

x_train_mnist = x_train_mnist.astype(np.float32) / 255.0
x_test_mnist = x_test_mnist.astype(np.float32) / 255.0
x_train_mnist = np.expand_dims(x_train_mnist, -1)
x_test_mnist = np.expand_dims(x_test_mnist, -1)

print("MNIST Loaded:", x_train_mnist.shape, x_test_mnist.shape)

# 2. Test Tiny ImageNet 10 loading
ti_zip_path = "tiny-imagenet-200.zip"
ti_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
if not os.path.exists(ti_zip_path):
    print("Downloading Tiny ImageNet 200...")
    urllib.request.urlretrieve(ti_url, ti_zip_path)

classes_10 = ['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 
              'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750']
class_to_idx = {c: i for i, c in enumerate(classes_10)}

with zipfile.ZipFile(ti_zip_path) as z:
    words = z.read('tiny-imagenet-200/words.txt').decode('utf-8')
    class_names = {}
    for line in words.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            class_names[parts[0]] = parts[1].split(',')[0].strip()
            
    names = [n for n in z.namelist() if n.startswith('tiny-imagenet-200/train/') and n.endswith('.JPEG')]
    names_10 = [n for n in names if n.split('/')[2] in class_to_idx]
    
    images, labels = [], []
    for n in names_10:
        c = n.split('/')[2]
        img_bytes = z.read(n)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        images.append(np.array(img, dtype=np.uint8))
        labels.append(class_to_idx[c])

X_ti = np.stack(images).astype(np.float32) / 255.0
y_ti = np.array(labels, dtype=np.int64)

def stratified_split_np(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(X))
    train_idx, test_idx = [], []
    for c in np.unique(y):
        c_idx = indices[y == c]
        rng.shuffle(c_idx)
        n_test = int(len(c_idx) * test_size)
        test_idx.extend(c_idx[:n_test])
        train_idx.extend(c_idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train_ti, X_test_ti, y_train_ti, y_test_ti = stratified_split_np(X_ti, y_ti, test_size=0.2, random_state=42)

print(f"Tiny ImageNet 10 Loaded: Train={X_train_ti.shape}, Test={X_test_ti.shape}")
print("Test pipeline completed successfully!")
