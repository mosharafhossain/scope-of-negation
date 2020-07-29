# -*- coding: utf-8 -*-
"""
"""

import data_prep
import embeddings
import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import evaluation
import spacy
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import cue_detector
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model
import pickle



class train():
    def __init__(self, params):
        
        # Parameter Settings
        self.max_len           = params["max_len"]     # maximum allowed length of sentences
        self.batch_size        = params["batch_size"]
        self.training_file     = params["training_file"]
        self.embed_size        = params["embed_size"]  # embedding dimension
        self.embed_file        = params["embed_file"]
        self.features_dict     = params["features_dict"]
        self.num_epoch         = params["num_epoch"]
        self.model_path        = params["model_path"]
        self.final_model_path  = params["final_model_path"]
        self.patience          = params["patience"]
        self.isLower           = params["isLower"]
        self.isIncludeNonCue   = params["isIncludeNonCue"]
        self.label_name        = params.get("label_name", "scopes")
        
        # Network params
        self.cell_units        = params["cell_units"]   
        self.drpout            = params["drpout"] 
        self.rec_drpout        = params["rec_drpout"] 
        self.validation_split  = params["validation_split"] 
        
        self.vrbose            = params["vrbose"]        
        self.tr_obj            = data_prep.data_for_training() #object creation
        
        
        
    def prepare_tr_data(self):
        """
        Prepare training data.
        """
        training_obj = open(self.training_file, "r", encoding="utf8") #training + development          
        tr_proc_data, train_y, token_dict, index_dict = self.tr_obj.get_data_for_training(training_obj, self.max_len, self.isLower, self.isIncludeNonCue, self.label_name)                
        return tr_proc_data, train_y, token_dict, index_dict
    
    def prepare_embed_matrix(self, index_dict):
        """
        Prepare Embedding matrix.
        """
        word2index    = index_dict["word2index"]
        embed_obj     = embeddings.word_embed()
        embed_matrix  = embed_obj.glove_embedding(self.embed_file, self.embed_size, word2index)
        return embed_matrix


    def fit(self, ):
        """
        Fit a neural model.
        """        
        # Prepare data for training
        tr_refined_data, train_y, token_dict, index_dict = self.prepare_tr_data()
        embed_matrix = self.prepare_embed_matrix(index_dict)
        train_x, num_tokens, embed_dims = self.tr_obj.prepare_training_data(tr_refined_data, self.features_dict, index_dict, self.embed_size)
        num_labels = len(index_dict["scope2index"])        
                
        # Get the model     
        model_obj = nn.scope_models() 
        crf, model = model_obj.bilstm_glove_crf(self.max_len, self.features_dict, num_tokens, num_labels, embed_dims,  embed_matrix, self.cell_units, self.drpout, self.rec_drpout)
        early_stop = EarlyStopping(monitor='val_crf_viterbi_accuracy', mode='max', patience=self.patience)  #or monitor='val_loss', mode='min'
        model_check = ModelCheckpoint(self.model_path, monitor='val_crf_viterbi_accuracy', mode='max', save_best_only=True, verbose=self.vrbose) #or monitor='val_loss', mode='min'

        # Compile the model
        model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_viterbi_accuracy])        
        
        # Train the model
        history = model.fit(train_x, train_y,              
                    batch_size=self.batch_size, 
                    epochs=self.num_epoch, 
                    shuffle=True, 
                    validation_split=self.validation_split, 
                    callbacks=[early_stop, model_check], 
                    verbose=self.vrbose)
        
        #Load the best model
        model.load_weights(self.model_path) #load the weights of the best model
        
        # Save best model
        model.save(self.final_model_path)
        output_dict = {"history":history, "model": model, "token_dict":token_dict, "index_dict":index_dict, "max_len":self.max_len, "features_dict":self.features_dict}    
        return output_dict
    
    def evaluate_model(self, params, output_dict):
        """
        Evaluate the model on cd-sco test file.
        """
        
        # Load gold test file
        test_file    = open(params["cd-sco_test_file"], "r", encoding="utf8")
        dp_obj       = data_prep.data_preparation()
        prep_obj     = dp_obj.data_load(test_file)        
        newobj_list  = dp_obj.get_gold_cue_file_pp(prep_obj)
        
        
        # Extract data for scope resolution
        data_dict     = dp_obj.get_data_details(newobj_list)        
        eval_obj      = evaluation.evaluation()
        tr_obj        = data_prep.data_for_training()
        index_dict    = output_dict["index_dict"]
        token_dict    = output_dict["token_dict"]
        model         = output_dict["model"]
        max_len       = output_dict["max_len"]
        features_dict = output_dict["features_dict"]        
        index2label   = index_dict["index2scope"] 
        
        negation_dict = eval_obj.tag_negation_scopes(model, features_dict, dp_obj, tr_obj, newobj_list, data_dict, max_len, index_dict, token_dict, index2label, params["isIncludeNonCue"], params["isLower"])        
        new_obj_list = dp_obj.create_new_obj_list(newobj_list, negation_dict)        
        dp_obj.print_to_file(new_obj_list, params["cd-sco_prediction_file"]) 

        
        
        
class scope_prediction():        
    def tag_negation_cue(self, sent_list, params, token_dict):
        
        """
        params:
            sent_list: list of original sentences (not tokenized).
            params: parameters of scope model
        """
        
        # Load pre-trained cue model
        custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
        cue_model = load_model(params["cue-model"], custom_objects = custom_objects)
        
        # Load cue vocabs
        with open(params["cue-model-vocabs"], "rb") as file_obj:
            cue_vocab_dict = pickle.load(file_obj)  
        cue_output_dict = {"model":cue_model, "token_dict":cue_vocab_dict["token_dict"], "index_dict":cue_vocab_dict["index_dict"], "features_dict":cue_vocab_dict["features_dict"], "max_len":cue_vocab_dict["max_len"] }

        # Predict negation cues
        pred_obj      = cue_detector.cue_prediction()  
        cue_info_dict = pred_obj.generate_cue_info(sent_list, cue_output_dict)
        cue_pred = cue_info_dict["pred"]
        cue_pred = [["I_C" if tag in ["S_C", "PRE_C", "POST_C", "M_C"] else "O_C" for tag in tag_sent] for tag_sent in cue_pred]
        
        return cue_pred


    
    def prepare_data(self, orig_sent_list, model_dict, params):
        """
        prepare data for prediction.
        params:
            @orig_sent_list: (list), list of all original/raw sentences/segments.
            @model_dict: (dictionary), contains information such as pretrained model, index_dict, token_dict
        """

        
        # Collect information        
        index_dict     = model_dict["index_dict"]
        token_dict     = model_dict["token_dict"]
        features_dict  = model_dict["features_dict"]
        max_len        = model_dict["max_len"]
        
        nlp = spacy.load("en_core_web_sm")
        obj = data_prep.data_for_training()
        
        # Negation cue prediction
        sent_cue_list = self.tag_negation_cue(orig_sent_list, params, token_dict)
        #print("sent_cue_list: {}".format(sent_cue_list))
        
        # Tokenize the sentences
        sent_list = []
        upos_list = []
        for sent in orig_sent_list:
            spacy_doc = nlp(sent)
            word_tok_list   = [token.text for token in spacy_doc] # tokens of the sentence
            upos_tok_list   = [token.pos_ for token in spacy_doc] # universal POS of the sentence
            
            sent_list.append(word_tok_list)
            upos_list.append(upos_tok_list)
        
        
        # Prepare data for prediction
        pad_value = index_dict["word2index"]["PAD"]
        data_dict_scope = {}
        data_dict_scope["words"] = obj.get_sent_with_padding(sent_list, token_dict["words"], index_dict["word2index"], max_len, pad_value)
        data_dict_scope["cues"]  = obj.get_sent_with_padding(sent_cue_list, token_dict["cues"], index_dict["cue2index"], max_len, pad_value)
        data_dict_scope["upos"]  = obj.get_sent_with_padding(upos_list, token_dict["upos"], index_dict["upos2index"], max_len, pad_value)
        test_dict, _, _          = obj.prepare_training_data(data_dict_scope, features_dict, index_dict, params["embed_size"])
        
        return test_dict, sent_list, sent_cue_list
    
    def predict_scope(self, orig_sent_list, model_dict, params):
        
        # Prepare data for scope of negation prediction
        test_dict, sent_list, cue_sent_list = self.prepare_data(orig_sent_list, model_dict, params)
        
        # Predict scope of negation
        eval_obj   = evaluation.evaluation()
        model      = model_dict["model"]
        index_dict = model_dict["index_dict"]
        pred_scope = eval_obj.predict_test(model, test_dict, index_dict["index2scope"])
        pred_scope = [[p for p in pred_sent if p !='PAD'] for pred_sent in pred_scope]
        
        new_cue_pred   = []
        new_scope_pred = []
        new_token_sents = []
        for sent, cues, scopes in zip(sent_list, cue_sent_list, pred_scope):            
            num_tokens = len(sent)
            num_cues   = len(cues)
            num_scopes = len(scopes)
            #print("1. sent: {}, cues: {}, scopes: {}".format(len(sent), len(cues), len(scopes)))
            if num_cues > num_tokens:
                cues = cues[0:num_tokens] 
            else:
                cues = cues + ['O_C']*(num_tokens-num_cues)
            if num_scopes > num_tokens:
                scopes = scopes[0:num_tokens] 
            else:
                scopes = scopes + ['O_S']*(num_tokens-num_scopes)

            new_cue_pred.append(" ".join([c for c in cues]))
            new_scope_pred.append(" ".join([s for s in scopes]))
            #print("2. sent: {}, cues: {}, scopes: {}".format(len(sent), len(cues), len(scopes)))
            new_tokens = []
            for token, cue, scope in zip(sent, cues, scopes):
                if cue == "I_C":
                    new_tokens.append(token+"/"+cue)
                else:
                    new_tokens.append(token+"/"+scope)
            new_tokens = " ".join(token for token in new_tokens)
            new_token_sents.append(new_tokens)
        
        return new_token_sents, new_cue_pred, new_scope_pred
            
            
            
            
            
    