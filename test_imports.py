"""Test script to verify all installed packages"""

print("Testing package imports...\n")

try:
    import flask
    print("[OK] Flask")
except Exception as e:
    print(f"[FAIL] Flask: {e}")

try:
    import cv2
    print("[OK] OpenCV")
except Exception as e:
    print(f"[FAIL] OpenCV: {e}")

try:
    import numpy
    print("[OK] NumPy")
except Exception as e:
    print(f"[FAIL] NumPy: {e}")

try:
    import pandas
    print("[OK] Pandas")
except Exception as e:
    print(f"[FAIL] Pandas: {e}")

try:
    import matplotlib
    print("[OK] Matplotlib")
except Exception as e:
    print(f"[FAIL] Matplotlib: {e}")

try:
    import sklearn
    print("[OK] Scikit-learn")
except Exception as e:
    print(f"[FAIL] Scikit-learn: {e}")

try:
    import seaborn
    print("[OK] Seaborn")
except Exception as e:
    print(f"[FAIL] Seaborn: {e}")

try:
    import tensorflow as tf
    print(f"[OK] TensorFlow {tf.__version__}")
except Exception as e:
    print(f"[FAIL] TensorFlow: {e}")

try:
    import keras
    print(f"[OK] Keras {keras.__version__}")
except Exception as e:
    print(f"[FAIL] Keras: {e}")

print("\n[OK] Package testing complete!")

