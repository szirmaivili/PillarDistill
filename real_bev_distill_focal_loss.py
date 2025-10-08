import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import pickle
import math
import random
import os
import torch

import pillar_distill_masked_hm as pillar
import argparse

def argparser():

    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=str, help="A könyvtár, ahová az adatokat mentjük")
    p.add_argument("--lr", type=float, default=1e-03, help="Learning rate")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pillar-out-size", type=int, default=4, help="PillarFeatureNet kimeneti csatornáinak a száma")
    p.add_argument("--use-tp-fp-fn", type=bool, default=False, help="True positive/false positive/false negative maszkolás")
    p.add_argument("--lambda_1", type=float, default=0.5, help="Weight of teacher loss")
    p.add_argument("--lambda_2", type=float, default=0.5, help="Weight of label loss")
    return p.parse_args()

def main():

    args = argparser()

    pickle_folder = r'/PillarDistill/train_data/pickles'
    point_folder = r'/PillarDistill/real_bevs'
    example_folder = r'/PillarDistill/teacher_examples'

    tokens = [file[:-4] for file in os.listdir(point_folder) if file.endswith('.pkl')]
    print(len(tokens))
    
    # Adatok beolvasása

    heads = {'hm':10, 'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6} # Corrected 'vel' channel size to 84
    student = pillar.build_student_flatten_decoder_bigger((1440, 1440, 32), heads)

    w = {'hm':1.0,'reg': 1.0, 'height':0.5,'dim':0.5,'rot':0.5,'vel':0.25,'iou':0.5}
    trainer = pillar.MapDistillTrainer(student=student, w=w, use_tp_fp_fn=args.use_tp_fp_fn, lambda_1=args.lambda_1, lambda_2=args.lambda_2)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(args.lr))

    ds = pillar.make_streaming_dataset_pillars(tokens=tokens, point_folder=point_folder, pickle_folder = pickle_folder,
                                                        heads=heads, batch_size=args.batch_size, shuffle=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.work_dir, 'epoch_{epoch}_loss_{loss:.4f}.weights.h5'),
    save_weights_only=True,   # csak a súlyokat menti
    monitor="loss",       # figyelt metrika
    mode="min",               # kisebb = jobb
    save_best_only=True,      # csak a legjobb modellt menti
    save_freq="epoch"
    )

    latest_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.work_dir, 'latest.weights.h5'),
    save_weights_only=True,   # csak a súlyokat menti
    save_best_only=False,      # csak a legjobb modellt menti
    save_freq=1
    )

    steps = math.ceil(len(tokens)/args.batch_size)

    trainer.fit(ds, epochs = args.epochs, verbose = 1, steps_per_epoch = steps, callbacks = [checkpoint_cb])

if __name__ == '__main__':

    main()
