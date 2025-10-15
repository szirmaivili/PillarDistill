import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import h5py
import pillar_distill_task_separated as pillar
import argparse
import tensorflow as tf
from tensorflow.keras import layers, Model
import pickle
import torch
import math

def argparser():

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, help="Checkpoint file")
    p.add_argument("--work-dir", type=str, help="Folder to save predictions")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    return p.parse_args()

def main():

    print("Elérhető GPU-k:", tf.config.list_physical_devices('GPU'))
    print("Elérhető CPU-k:", tf.config.list_physical_devices('CPU'))

    args = argparser()

    folder = r'/PillarDistill/real_bevs_test'

    tokens = [file[:-4] for file in os.listdir(folder) if file.endswith('.pkl')]

    num_batches = math.ceil(len(tokens)/args.batch_size)

    print("Number of batches: ", num_batches)

    dimensions = [{'hm':1, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':1, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1}]
    
    full_model, base_model, heads_model = pillar.build_student_flatten_decoder_bigger(input_shape=(1440, 1440, 32), dimensions = dimensions)

    full_model.load_weights(args.checkpoint)

    gen_fn = pillar.make_streaming_dataset_pillars(tokens=tokens, point_folder=folder, batch_size=args.batch_size, shuffle=False)

    gen = gen_fn()

    for i, batch in enumerate(gen):

        if i >= num_batches:
            break

        student_input = batch['bev']
        tokens_raw = [t.decode('utf-8') for t in batch['tokenss'].numpy().flatten()]

        predictions = full_model(student_input, training=False)

        batch_data = {
        'tokens': tokens_raw,
        'preds': predictions  # ha tf.Tensor
            }
        
        with open(os.path.join(args.work_dir, f'batch_{i+1}.pkl'), 'wb') as f:
            
            pickle.dump(batch_data, f)

        print(f'{i+1}/{num_batches} batch processed')

if __name__ == "__main__":

    main()