import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
tf.data.experimental.enable_debug_mode()
import pickle
import math
import random
import os
import torch

import pillar_distill_task_separated as pillar
import argparse
import time

def argparser():

    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=str, help="A könyvtár, ahová az adatokat mentjük")
    p.add_argument("--lr", type=float, default=1e-03, help="Learning rate")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lambda_1", type=float, default=0.5, help="Weight of teacher loss")
    p.add_argument("--lambda_2", type=float, default=0.5, help="Weight of label loss")
    return p.parse_args()

def main():

    tf.config.run_functions_eagerly(True)
    print("Eager execution enabled:", tf.executing_eagerly())

    args = argparser()

    pickle_folder = r'/PillarDistill/train_data/pickles'
    point_folder = r'/PillarDistill/real_bevs'
    example_folder = r'/PillarDistill/teacher_examples'

    tokens = [file[:-4] for file in os.listdir(point_folder) if file.endswith('.pkl')]
    print("Number of samples: ", len(tokens))
    
    # Adatok beolvasása

    heads = {'hm':10, 'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6} # Corrected 'vel' channel size to 84
    dimensions = [{'hm':1, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':1, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1},
                  {'hm':2, 'reg': 2, 'height':1,'dim':3,'rot':2,'vel':2,'iou':1}]
    
    full_model, base_model, heads_model = pillar.build_student_flatten_decoder_bigger(input_shape=(1440, 1440, 32), dimensions = dimensions)

    w = {'hm':1.0,'reg': 1.0, 'height':0.5,'dim':0.5,'rot':0.5,'vel':0.25,'iou':0.5}
    trainer = pillar.MapDistillTrainer(student=full_model, w=w)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(args.lr))

    gen_fn = pillar.make_streaming_dataset_pillars(tokens=tokens, point_folder=point_folder, batch_size=args.batch_size, shuffle=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.work_dir, 'epoch_{epoch}_loss_{loss:.4f}.weights.h5'),
    save_weights_only=True,   # csak a súlyokat menti
    monitor="loss",       # figyelt metrika
    mode="min",               # kisebb = jobb
    save_best_only=True,      # csak a legjobb modellt menti
    save_freq="epoch"
    )

    epochs = args.epochs

    steps = math.ceil(len(tokens)/args.batch_size)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        gen = gen_fn()  # újraindítjuk a generátort epochonként
        start_time = time.time()
        for step, batch in enumerate(gen):
            logs = trainer.train_step(batch)

            elapsed = time.time() - start_time
            avg_step_time = elapsed / (step + 1)
            remaining = avg_step_time * (steps - step - 1)

            eta_h, rem = divmod(remaining, 3600)
            eta_m, eta_s = divmod(rem, 60)
            eta_str = f"{int(eta_h):02d}:{int(eta_m):02d}:{int(eta_s):02d}"
            
            print(f"Step {step+1}/{steps} - loss: {logs['loss']:.4f} | ETA: {eta_str}", end="\r", flush=True)
            if step >= steps:
                break

        full_model.save_weights(os.path.join(args.work_dir, f"epoch_{epoch+1}_loss_{logs['loss']:.4f}.weights.h5"))
        

    base_model.save_weights(os.path.join(args.work_dir, 'base_model.weights.h5'))
    heads_model.save_weights(os.path.join(args.work_dir, 'heads_model.weights.h5'))

if __name__ == '__main__':

    main()