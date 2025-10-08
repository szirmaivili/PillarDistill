import tensorflow as tf

def raw_preds_2_preds_dicts(raw_preds: dict, dimensions: dict, tiling: bool = True):
    # Batch méretű objektumok: B, H, W, C. NEM KELL A DIMENZIÓKAT PERMUTÁLNI
    preds_dicts = []

    if tiling:

        expanded_preds = {k: tf.tile(v, [4, 1, 1, 1]) for k,v in raw_preds.items()}

    else:

        expanded_preds = raw_preds

    # hm hozzáadása:
    preds_dicts.append({'hm': expanded_preds['hm'][..., 0:1]})
    preds_dicts.append({'hm': expanded_preds['hm'][..., 1:3]})
    preds_dicts.append({'hm': expanded_preds['hm'][..., 3:5]})
    preds_dicts.append({'hm': expanded_preds['hm'][..., 5:6]})
    preds_dicts.append({'hm': expanded_preds['hm'][..., 6:8]})
    preds_dicts.append({'hm': expanded_preds['hm'][..., 8:10]})

    # Többi fej hozzáadása: 
    for i in range(6):

        for key in dimensions.keys():

            preds_dicts[i].update({key: expanded_preds[key][..., i*dimensions[key]//6 : (i+1)*dimensions[key]//6]})

    return preds_dicts
