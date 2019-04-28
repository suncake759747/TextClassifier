import sys
import pandas as pd
import pickle

from util import process_held_out_data
from nb_model_helpers import use_nb

held_out_file = sys.argv[1]
process_held_out_data(held_out_file)
    
nb_probs, true_classes, nb_preds, ids = use_nb('held_out_data.tsv')

with open ('nb_predictions.tsv', 'w') as f:
    for index in range(ids.shape[0]):
        f.write("%s\t%s\n" % (ids[index], nb_preds[index]))