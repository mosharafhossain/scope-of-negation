# -*- coding: utf-8 -*-
"""
"""

import scope_detector
import os
import random
import numpy as np
import tensorflow as tf
import argparse
import json 
import pickle
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)	
    np.random.seed(seed)
    tf.set_random_seed(seed) # set the graph-level seed.

def read_file(file_path):
    sent_list = []    
    file_obj = open(file_path, "r", encoding="utf8")
    for line in file_obj:
        sent_list.append(line.strip())
    file_obj.close()                
    return sent_list

def write_file(file_path, sent_list):
    file_obj = open(file_path, "w")
    for sent in sent_list:
        file_obj.write(sent)
        file_obj.write("\n")
    file_obj.close()  

if __name__ == "__main__":
    
    # python predict.py -c ./config/config.json -i ./data/sample-io/input_file.txt -o ./data/sample-io/ --cd_sco_eval True
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config_path", help="path to the configuration file", required=True)      
    argParser.add_argument("-i", "--input_file", help="path to the input file (one sentence per line)", required=True)
    argParser.add_argument("-o", "--output_dir", help="path to the output directory (each line in the output file contains token level prediction of negation scope (and cue))", required=True)
    argParser.add_argument("--cd_sco_eval", default=False, help="whether checking evaluation on cd-sco test corpus")
    
    args            = argParser.parse_args()
    config_path     = args.config_path    
    input_file_path = args.input_file
    output_dir      = args.output_dir
    cd_sco_eval     = args.cd_sco_eval
        
    
    # Read parameters from json file
    with open(config_path.strip()) as json_file_obj: 
        params = json.load(json_file_obj)
        
    # Set the seed    
    set_seed(params["seed"]) 


    # Load pre-trained model
    custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
    model = load_model(params["final_model_path"], custom_objects = custom_objects)

    
    # Load vocabs
    with open(params["vocab_loc"], "rb") as file_obj:
        vocab_dict = pickle.load(file_obj)  
    output_dict = {"model":model, "token_dict":vocab_dict["token_dict"], "index_dict":vocab_dict["index_dict"], "features_dict":vocab_dict["features_dict"], "max_len":vocab_dict["max_len"] }


    #Evaluate on cd-sco test corpus
    if cd_sco_eval:
        tr_model = scope_detector.train(params)
        tr_model.evaluate_model(params, output_dict)       
    
    
    
    # Test on given input file............................
    pred_obj        = scope_detector.scope_prediction() 
    orig_sent_list  = read_file(input_file_path)    
    new_token_sents, _, _ = pred_obj.predict_scope(orig_sent_list, output_dict, params)
    write_file(os.path.join(output_dir, "output_file.txt"),  new_token_sents)

    

    
    