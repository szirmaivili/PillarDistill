import pillar_distill_masked_hm as pillar
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import pickle
import math
import random
import os
import argparse
import torch
import h5py


# def make_streaming_dataset_pillars(samples, pillar_net, heads, batch_size=16, shuffle=True):
#     ch = {k: heads[k] for k in ['hm','reg','height','dim','rot','vel','iou']}

#     def gen():
#         idxs = np.arange(len(samples))
#         if shuffle:
#             np.random.shuffle(idxs)

#         for s in range(0, len(samples), batch_size):
#             sel = idxs[s:s+batch_size]

#             bevs, masks, tokens = [], [], []
#             teach = {k: [] for k in ch.keys()}
        
#             for j in sel:
#                 smp = samples[j]
#                 pts = smp['points']

#                 bev = pillar_net(pts[:, :3], pts[:, 3:5], training=False).numpy()

#                 bevs.append(bev)

#                 t = pillar.build_teacher_dict(smp)
#                 m = pillar.make_fg_mask_from_hm(t["hm"]).astype("float32")
#                 masks.append(m)
#                 tokens.append(smp['token'])

#                 for k in ch.keys():
#                     teach[k].append(t[k].astype("float32"))

#             # konvertálás RaggedTensor-ba közvetlenül
#             batch = {
#                 "bev": np.stack(bevs, axis=0).astype(np.float32),
#                 "teacher": {k: np.stack(teach[k], axis=0) for k in ch.keys()},
#                 "mask_fg": np.stack(masks, axis=0),
#                 "token": np.array(tokens).reshape(-1, 1)
#             }
#             yield batch

#     output_signature = {
#         "bev": tf.TensorSpec(shape=(None, None, None, 4), dtype=tf.float32),
#         "teacher": {
#             "hm": tf.TensorSpec(shape=(None, None, None, ch["hm"]), dtype=tf.float32),
#             "reg": tf.TensorSpec(shape=(None, None, None, ch["reg"]), dtype=tf.float32),
#             "height": tf.TensorSpec(shape=(None, None, None, ch["height"]), dtype=tf.float32),
#             "dim": tf.TensorSpec(shape=(None, None, None, ch["dim"]), dtype=tf.float32),
#             "rot": tf.TensorSpec(shape=(None, None, None, ch["rot"]), dtype=tf.float32),
#             "vel": tf.TensorSpec(shape=(None, None, None, ch["vel"]), dtype=tf.float32),
#             "iou": tf.TensorSpec(shape=(None, None, None, ch["iou"]), dtype=tf.float32),
#         },
#         "mask_fg": tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
#         "token": tf.TensorSpec(shape=(None, 1), dtype=tf.string)
#     }
    
#     return tf.data.Dataset.from_generator(
#         gen,
#         output_signature=output_signature
#     ).repeat().prefetch(tf.data.AUTOTUNE)

def argparser():

    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, help="Ahol a teszt adataink vannak")
    p.add_argument("--work-dir", type=str, help="Hová mentsük az eredményeket")
    p.add_argument("--checkpoint", type=str, help="Checkpoint file helye")
    return p.parse_args()

def main():

    args = argparser()

    # with open(args.dataroot, 'rb') as f:

    #     samples = pickle.load(f)

    # print("Samples read!")

    heads = {'hm':10, 'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}
    student = pillar.build_student_flatten_decoder_bigger((1440, 1440, 32), heads = heads)
    student.build((None,) + (1440, 1440, 32))

    #student.load_weights(args.checkpoint, by_name=True, skip_mismatch=True)
    #student.summary()

    # def print_h5_structure(file_path):
    #     def print_attrs(name, obj):
    #         if isinstance(obj, h5py.Dataset):
    #             print(f"{name} -> shape: {obj.shape}")
    #         elif isinstance(obj, h5py.Group):
    #             print(f"{name}/ (Group)")

    #     with h5py.File(file_path, "r") as f:
    #         f.visititems(print_attrs)

    # print_h5_structure(args.checkpoint)

    with h5py.File(args.checkpoint, "r") as f:
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

    check_weights(student)
    
    # all_h5_weights = {}
    # checkpoint_path = args.checkpoint
    # if not os.path.exists(checkpoint_path):
    #     print(f"Error: Checkpoint file not found at {checkpoint_path}")
    #     exit()

    # with h5py.File(checkpoint_path, 'r') as f:
    #     def _collect_weights(name, obj):
    #         if isinstance(obj, h5py.Dataset):
    #             all_h5_weights[name] = obj[()] # Kinyerjük a NumPy tömböt
    #     f.visititems(_collect_weights)

    # # 2. Iteráljunk végig a modell rétegein, és próbáljuk meg beállítani a súlyokat
    # weights_loaded_count = 0
    # weights_skipped_count = 0

    # for layer in student.layers:
    #     # A mi modellünkben csak a Conv2D, Dense rétegeknek van súlya. Activation, Flatten, Reshape, Cropping2D -nek nincs.
    #     # Az Activation rétegnek sincs weights attribútuma, ezért hagyjuk ki.
    #     if not layer.weights or isinstance(layer, (layers.InputLayer, layers.Flatten, layers.Reshape, layers.Cropping2D, layers.Activation)):
    #         continue

    #     # A Keras layer.name alapján keressük a súlyokat a h5 fájlban
    #     base_h5_path = f"layers/functional/layers/{layer.name}/vars"
        
    #     # A réteg által elvárt súlyok listája (pl. kernel, bias)
    #     current_layer_weights = []

    #     # Itt kellene tudnunk, hogy az adott réteg hány súlyt vár, és milyen sorrendben.
    #     # Konvolúciós és Dense rétegeknél: [kernel, bias (opcionális)]
        
    #     # Kernel súly
    #     kernel_h5_path = f"{base_h5_path}/0"
    #     if kernel_h5_path in all_h5_weights:
    #         current_layer_weights.append(all_h5_weights[kernel_h5_path])
    #     else:
    #         print(f"  WARNING: Kernel weight not found for {layer.name} at {kernel_h5_path}. Skipping layer.")
    #         weights_skipped_count += 1
    #         continue # Ha a kernelt nem találjuk, az egész réteget hagyjuk ki

    #     # Bias súly (ha van a rétegnek, és a h5 fájlban is van)
    #     # Ellenőrizzük a réteg konfigurációját, hogy van-e bias-a
    #     if hasattr(layer, 'use_bias') and layer.use_bias:
    #          bias_h5_path = f"{base_h5_path}/1"
    #          if bias_h5_path in all_h5_weights:
    #              current_layer_weights.append(all_h5_weights[bias_h5_path])
    #          else:
    #              print(f"  WARNING: Bias weight expected for {layer.name}, but not found at {bias_h5_path}. Skipping layer.")
    #              weights_skipped_count += 1
    #              continue # Ha a bias-t nem találjuk, az egész réteget hagyjuk ki


    #     try:
    #         # Csak akkor állítsuk be a súlyokat, ha a számuk megegyezik
    #         if len(current_layer_weights) == len(layer.weights):
    #             layer.set_weights(current_layer_weights)
    #             weights_loaded_count += 1
    #             print(f"  Successfully loaded weights for layer: {layer.name}")
    #         else:
    #             print(f"  WARNING: Mismatch in number of weights for {layer.name}. Expected {len(layer.weights)}, found {len(current_layer_weights)}. Skipping layer.")
    #             weights_skipped_count += 1
    #     except Exception as e:
    #         print(f"  ERROR setting weights for layer {layer.name}: {e}. Skipping layer.")
    #         weights_skipped_count += 1

    # print(f"Manual weight loading completed. Loaded {weights_loaded_count} layers, skipped {weights_skipped_count} layers.")

    print("Student checkpoint loaded!")

    # pillar_net = pillar.PillarFeatureNet(out_channels = 4)

    # ds = make_streaming_dataset_pillars(samples=samples, pillar_net=pillar_net, heads=heads, shuffle=False)

    # num_batches = math.ceil(len(samples)/16)

    # for batch_idx, batch_data in enumerate(ds):

    #     if batch_idx >= num_batches:

    #         break

    #     stufdent_input = batch_data["bev"]

    #     predictions = student(stufdent_input, training=False)

    #     tokens_raw = batch_data["token"].numpy().flatten()

    #     data = {}

    #     for i in range(len(tokens_raw)):
    #         current_token = tokens_raw[i].decode('utf-8') if isinstance(tokens_raw[i], bytes) else tokens_raw[i]
            
    #         # Készítünk egy szótárt az aktuális tokenhez tartozó eredményekkel
    #         token_predictions = {head_name: pred_array[i] for head_name, pred_array in predictions.items()}

    #         data.update({current_token: token_predictions})
            
    #     output_filepath = os.path.join(args.work_dir, f"batch_{batch_idx+1}.pkl")
    #     with open(output_filepath, 'wb') as f:
    #         pickle.dump(data, f)

    #     print(f"{batch_idx+1}/{num_batches} processed")

if __name__ == "__main__":

    main()