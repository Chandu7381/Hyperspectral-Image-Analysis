import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def list_classes(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    return classes

def load_images_from_folder(folder, target_size=(128,128), max_files=None):
    X, y = [], []
    classes = list_classes(folder)
    for cls in classes:
        cls_folder = os.path.join(folder, cls)
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        if max_files:
            files = files[:max_files]
        for f in files:
            path = os.path.join(cls_folder, f)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            X.append(img)
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=object)
    return X, y

def normalize_images(X):
    return X / 255.0

def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_map = {i: c for i, c in enumerate(le.classes_)}
    return y_enc, le, class_map

def prepare_generators(train_dir, val_dir, target_size=(128,128), batch_size=32):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)
    train_flow = train_gen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical')
    val_flow = val_gen.flow_from_directory(
        val_dir, target_size=target_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
    return train_flow, val_flow
