import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import os
import h5py

def load_weights(checkpoint, student):

    with h5py.File(checkpoint, "r") as f:
        student_group = f["student"]["layers"]
        for layer in student.layers:
            lname = layer.name
            if lname in student_group:
                vars_group = student_group[lname]["vars"]
                weights = []
                for k in sorted(vars_group.keys()):
                    weights.append(np.array(vars_group[k]))
                if weights:
                    try:
                        layer.set_weights(weights)
                        print(f"Loaded weights for {lname}")
                    except Exception as e:
                        print(f"Skipping {lname}: {e}")

def check_weights(model):
        all_ok = True
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:  # csak azok a rétegek, amiknek van súlya
                for i, w in enumerate(weights):
                    if np.all(w == 0):
                        print(f"⚠️ {layer.name} var[{i}] MIND NULLA!")
                        all_ok = False
                    elif np.any(np.isnan(w)):
                        print(f"❌ {layer.name} var[{i}] NaN értéket tartalmaz!")
                        all_ok = False
                    else:
                        print(f"✅ {layer.name} var[{i}] OK (shape={w.shape}, min={w.min():.4f}, max={w.max():.4f})")
        if all_ok:
            print("\n🎉 Minden súly rendben van, a modell ténylegesen betöltődött!")
        else:
            print("\n⚠️ Volt néhány gyanús súly, érdemes még utánanézni.")