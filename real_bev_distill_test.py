import pillar_distill_masked_hm as pillar
import center_head
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
import raw_preds_2_preds_dicts
 

code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel': (2, 2), 'iou': (1, 2)}
pillar_size=0.075
pc_range=[-54, -54, -5.0, 54, 54, 3.0]

tasks = [
    dict(stride=8, class_names=["car"]),
    dict(stride=8, class_names=["truck", "construction_vehicle"]),
    dict(stride=8, class_names=["bus", "trailer"]),
    dict(stride=8, class_names=["barrier"]),
    dict(stride=8, class_names=["motorcycle", "bicycle"]),
    dict(stride=8, class_names=["pedestrian", "traffic_cone"]),
    ]

class config:

        def __init__(self, post_center_limit_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                        max_per_img = 500,
                        nms = dict(
                        use_rotate_nms=True,
                        use_multi_class_nms=False,
                        nms_pre_max_size=1000,
                        nms_post_max_size=83,
                        nms_iou_threshold=0.2,
                    ),
                    score_threshold = 0.1,
                    rectifier = 0.5,
                    pc_range = pc_range[:2],
                    out_size_factor = 8,
                    voxel_size=[pillar_size, pillar_size],
                    double_flip = False,
                    return_raw = False,
                    per_class_nms = False,
                    circular_nms = False):
            
            self.post_center_limit_range = post_center_limit_range
            self.max_per_img = max_per_img
            self.nms = nms
            self.score_threshold = score_threshold
            self.rectifier = rectifier
            self.pc_range = pc_range
            self.out_size_factor = out_size_factor
            self.voxel_size = voxel_size
            self.double_flip = double_flip
            self.return_raw = return_raw
            self.per_class_nms = per_class_nms
            self.circular_nms = circular_nms

test_cfg = config()

cent_head = center_head.CenterHead(tasks=tasks, 
                                   code_weights=code_weights,
                                   common_heads=common_heads,
                                   )

def argparser():

    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=str, help="Hová mentsük az eredményeket")
    p.add_argument("--checkpoint", type=str, help="Checkpoint file helye")
    return p.parse_args()

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

def main():

    args = argparser()

    os.makedirs(args.work_dir, exist_ok=True)

    point_folder = r'/PillarDistill/real_bevs_test'
    pickle_folder = r'/PillarDistill/test_data/teacher_dicts'

    tokens = [file[:-4] for file in os.listdir(point_folder) if file.endswith('.pkl')]

    N = len(tokens)

    num_batches = math.ceil(N/16)

    heads = {'hm':10, 'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}
    dimensions = {'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}

    ds = pillar.make_streaming_dataset_pillars(tokens=tokens, point_folder=point_folder, pickle_folder=pickle_folder, heads=heads, shuffle=False)

    student = pillar.build_student_flatten_decoder_bigger((1440, 1440, 32), heads=heads)

    student.build((None,) + (1440, 1440, 32))

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

    check_weights(student)

    for i, batch in enumerate(ds):

        if i >= num_batches:

            break
        
        batch_data = {}

        student_input = batch['bev']

        predictions = student(student_input, training=False)

        tokens_raw = batch['tokenss'].numpy().flatten()

        #preds_dicts = raw_preds_2_preds_dicts.raw_preds_2_preds_dicts(raw_preds=predictions, dimensions=dimensions)

        #ret_list = cent_head.predict(preds_dicts=preds_dicts, example={}, test_cfg=test_cfg)

        # for j, item in enumerate(ret_list):

        #     pred_dict.update({tokens_raw[j]: item})

        batch_data.update({'tokens': tokens_raw})
        batch_data.update({'preds': predictions})

        with open(os.path.join(args.work_dir, f'batch_{i+1}.pkl'), 'wb') as f:
            
            pickle.dump(batch_data, f)

        print(f'{i+1}/{num_batches} batch processed')

if __name__ == '__main__':

    main()