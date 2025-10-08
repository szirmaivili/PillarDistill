import tensorflow as tf
import os
import pickle

def main():

    files = os.listdir(r'/PillarDistill/processed_batches_new')

    predictions = {}

    for i, file in enumerate(files):

        with open(os.path.join(r'/PillarDistill/processed_batches_new', file), 'rb') as f:

            data = pickle.load(f)

        predictions.update(data.items())

        # if i == 0:

        #     print(data.keys())

    predictions = {k.decode(): v for k,v in predictions.items()}

    print(len(predictions))

    with open(r'/PillarDistill/real_bev_distill_test_new/predictions.pkl', 'wb') as f:

        pickle.dump(predictions, f)

if __name__ == '__main__':

    main()