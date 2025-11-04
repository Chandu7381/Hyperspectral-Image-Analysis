import numpy as np

def compute_exg(img):
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    R = arr[...,0]
    G = arr[...,1]
    B = arr[...,2]
    exg = 2*G - R - B
    return exg

def compute_exr(img):
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    R = arr[...,0]
    G = arr[...,1]
    exr = 1.4*R - G
    return exr

def extract_global_features(img):
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    features = []
    for ch in range(3):
        features.append(arr[...,ch].mean())
        features.append(arr[...,ch].std())
    exg = compute_exg(arr)
    exr = compute_exr(arr)
    features.append(exg.mean())
    features.append(exr.mean())
    return np.array(features, dtype=np.float32)
