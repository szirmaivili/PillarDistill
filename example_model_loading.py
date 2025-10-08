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
import model_loader

def argparser():

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, help="A checkpoint file elérési útvonala, ami a modell súlyait tartalmazza")
    p.add_argument("--work-dir", type=str, help="Hová mentsük ki a predikciókat")
    p.add_argument("--teacher-folder", type=str, help="Teacher inputok mappája (enélkül nem megy a dataloader)")
    p.add_argument("--input-folder", type=str, help="Bemeneti adatok mappája")
    return p.parse_args()

def main():

    args = argparser()

    '''
    Ebben a kis file-ban az van bemutatva, hogy hogyan kell betölteni egy ilyen self-made modellt. A ds (dataset) maker-ben a keyword argument-ek nem túl beszédesek, így legalább
    a parancssori argumentumokat olyanra állítottam. Akár le se kell futtatni, bár szerintem hasznos lehet.
    A checkpoint az a checkpoint file, ami fent van a branchban is
    A work-dir többé kevésbé opcionális ezesetben
    A teacher-folder a test_teacher névre elkeresztelt mappa legyen
    Az input-folder pedig a test_input nevű mappa
    '''

    tokens = [file[:-4] for file in os.listdir(args.input_folder) if file.endswith('.pkl')]
    N = len(tokens)

    num_batches = math.ceil(N/16)

    heads = {'hm':10, 'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}

    ds = pillar.make_streaming_dataset_pillars(tokens=tokens, point_folder=args.input_folder, pickle_folder=args.teacher_folder, heads=heads, shuffle=False)

    # Ez a student inicializálása:

    student = pillar.build_student_flatten_decoder_bigger((1440, 1440, 32), heads=heads)

    student.build((None,) + (1440, 1440, 32))

    # Student súlyainak betöltése:

    model_loader.load_weights(args.checkpoint, student)

    # Súlyok ellenőrzése: 

    model_loader.check_weights(student)

    for i, batch in enumerate(ds):

        if i >= num_batches:

            break
        
        batch_data = {}

        student_input = batch['bev']

        predictions = student(student_input, training=False)

        tokens_raw = batch['tokenss'].numpy().flatten()

        batch_data.update({'tokens': tokens_raw})
        batch_data.update({'preds': predictions})

        with open(os.path.join(args.work_dir, f'batch_{i+1}.pkl'), 'wb') as f:
            
            pickle.dump(batch_data, f)

        print(f'{i+1}/{num_batches} batch processed')

    '''
    Ez olyan típusú kimeneteket fog gyártani, mint amilyenek a test_raw_preds mappában vannak. Annyi, hogy valószínűleg máshogy fogja keverni a batch-eket, de a kimenetek biztos,
    hogy ugyanazokhoz a mintákhoz fognak tartozni.
    '''

if __name__ == "__main__":

    main()