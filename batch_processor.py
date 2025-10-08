import center_head
import tensorflow as tf
tf.config.run_functions_eagerly(True)
import os
import pickle
import raw_preds_2_preds_dicts
import torch

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
                    double_flip = True,
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

cent_head = center_head.CenterHead(tasks=tasks, code_weights=code_weights, common_heads=common_heads)

dimensions = {'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}

def tf_to_torch(x):
        if isinstance(x, tf.Tensor):
            return torch.from_numpy(x.numpy())
        elif isinstance(x, dict):
            return {k: tf_to_torch(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [tf_to_torch(v) for v in x]
        else:
            return x

def main():

    folder = r'/PillarDistill/real_bev_distill_test_new'

    i = 0

    predictions = {}
    processed_files = []
    files = [file for file in os.listdir(folder) if 'batch' in file]
    faulty = []

    while i < 377:

        for file in files:

            if file not in processed_files:

                batch_preds = {}

                try:
                 
                    with open(os.path.join(folder, file), 'rb') as f:
                        
                        batch_data = pickle.load(f)

                except:
                     
                     processed_files.append(file)
                     i += 1
                     faulty.append(file)
                     continue

                try:

                    raw_preds = batch_data['preds']
                    tokens = batch_data['tokens']

                except:
                     
                     processed_files.append(file)
                     i += 1
                     faulty.append(file)
                     continue
                
                preds_dicts = raw_preds_2_preds_dicts.raw_preds_2_preds_dicts(raw_preds=raw_preds, dimensions=dimensions)
                ret_list = cent_head.predict(preds_dicts=preds_dicts, example={}, test_cfg=test_cfg)

                for key, value in zip(tokens, ret_list):
                     
                    batch_preds[key] = value

                processed_files.append(file)
                i += 1

                print(f'{i}/377 batch processed')

                with open(os.path.join(r'/PillarDistill/processed_batches_new', f'processed_{i}.pkl'), 'wb') as f:
                     
                    pickle.dump(batch_preds, f)

    print(faulty)
                     
if __name__ == '__main__':
     
     main()