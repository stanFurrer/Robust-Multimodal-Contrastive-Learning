import torch
from copy import deepcopy
import sys
import nltk
from nltk.corpus import stopwords
from torch.nn import functional as F
# Need to do this one time nltk.download('stopwords')
from torch.nn import CosineSimilarity
import string
import numpy as np
import os
import torch.nn.functional as F
import torch.nn as nn
from nltk.corpus import wordnet

from torch.nn.utils.rnn import pad_sequence
# Uncomment if first type using script
#nltk.download('wordnet')

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


class GreedyAttack_cross_entropy:
    def __init__(self, args,n_candidates,max_loops, tokenizer):
        self.args              = args
        self.stopwords         = set(stopwords.words('english'))
        self.model             = None
        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)
        self.tokenizer         = tokenizer        
        self.device            = None
        self.words_to_sub_words= None
        self.max_length        = args["max_text_len"]     
        self.n_candidates      = n_candidates  
        self.max_loops         = max_loops       
        self.sim_thred         = args["sim_thred"]        
        self.word2id           = self.tokenizer.get_vocab()              
        self.id2word           = {v: k for k, v in self.word2id.items()}         
        self.cos_sim           = None                                    
        self.sim_word2id       = None 
        self.sim_id2word       = None
        self.synonym           = args["synonym"]
        self.cos_sim_dict      = None
        if self.args["cos_sim"]:                                              
            self.init_matrix(self.args["embedding_path"], self.args["sim_path"]) 

    def init_matrix(self, embedding_path, sim_path): 
        """Creat cos_sim_dict"""
        embeddings = []
        self.sim_id2word = {}
        self.sim_word2id = {}
        with open(embedding_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
                word = line.split()[0]
                if word not in self.sim_id2word:
                    self.sim_id2word[len(self.sim_id2word)] = word
                    self.sim_word2id[word] = len(self.sim_id2word) - 1
        # Load cos_sim if exit
        if os.path.exists(sim_path):
            self.cos_sim = np.load(sim_path,allow_pickle=True)
        # compute cos_sim if do not exit
        else:
            embeddings = np.array(embeddings)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.asarray(embeddings / norm, "float32")
            self.cos_sim = np.dot(embeddings, embeddings.T)
            np.save('cos_sim_counter_fitting.npy', self.cos_sim)
        if self.synonym == 'cos_sim':
            self.cos_sim_dict = {}
            for idx, word in self.sim_id2word.items():
                candidates = set()
                ### self.cos_sim is a matrix of 6e9 x 6e9
                indices = torch.topk(torch.tensor(self.cos_sim[idx]), k=self.n_candidates).indices
                for i in indices:
                    i = int(i)
                    if self.cos_sim[idx][i] < self.sim_thred:
                        break
                    if i == idx:
                        continue
                    candidates.add(self.sim_id2word[i])
                if len(candidates) == 0:
                    candidates = [word]
                self.cos_sim_dict[idx] = candidates
                
    def get_synonym_by_cos(self, word): 
        if not (word in self.sim_word2id):
            return [word]
        idx = self.sim_word2id[word]
        return self.cos_sim_dict[idx]
    
    def get_synonym(self, word):
        candidates = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name()
                if self.check_word(w):
                    continue
                if w in candidates:
                    continue
                candidates.add(w)
                
        if len(candidates) == 0:
            candidates = [word]

        return candidates[:self.n_candidates]    
    
    def get_important_scores(self,grads, words_to_sub_words):
        index_scores = [0.0] * len(words_to_sub_words)
        for i in range(len(words_to_sub_words)):
            matched_tokens  = words_to_sub_words[i]
            try :
                agg_grad        = np.mean(grads[matched_tokens], axis=0)
            except : 
                print("-------------------------------")
                print("HERE IS THE ERROR DESCRIPTION\n")
                print("This is words_to_sub_words", words_to_sub_words)
                print("This is the len of len(words_to_sub_words)", len(words_to_sub_words))
                print("This is the len of grads", grads.shape)
                print("This is i : ", i)
                print("This is matched_tokens_id : ", matched_tokens)
                sys.exit("Stop")
               
                # Get the norm of the gradient
                index_scores[i] = np.linalg.norm(agg_grad, ord=1)
        return index_scores
       
    def get_inputs(self,sentences, tokenizer, device):
        outputs        = tokenizer(sentences, truncation=True, padding=True,max_length = self.max_length)
        
        
        input_ids      = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]

        return  torch.tensor(input_ids).to(device)\
                ,torch.tensor(attention_mask).to(device)
         
    def check_word(self, word):
        return word == '[PAD]' or word == '[UNK]' or word == '[CLS]' or \
               word == '[SEP]' or word in self.stopwords or word in string.punctuation or \
               word in filter_words or word in '...' or word == '[MASK]'

    def get_grad(self,
                input_ids,
                text_masks,
                text,
                batch,
                device          
    ):
        embedding_layer = self.pl_module.get_input_embeddings()  # word_embeddings 
        projector_layer    = self.pl_module.nlvr2_classifier.linear2_nlvr2 #----

        original_state_pro = projector_layer.weight.requires_grad
        projector_layer.weight.requires_grad = True

        original_state_emb  = embedding_layer.weight.requires_grad 
        embedding_layer.weight.requires_grad = True                  

        emb_grads = []
        pro_grads = []

        def emb_grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
    
        def pro_grad_hook(module, grad_in, grad_out):
            pro_grads.append(grad_out[0])        
            
        emb_hook = embedding_layer.register_backward_hook(emb_grad_hook)
        pro_hook = projector_layer.register_backward_hook(pro_grad_hook)

        self.pl_module.zero_grad() # Isn't it a problem for accumulated gradient to zeros grad here----
        
        with torch.enable_grad(): 
            batch["text_ids"]   = input_ids
            batch["text_masks"] = text_masks
            batch["text"]       = text
            
            infer1 = self.pl_module.infer(
                batch, mask_text=False, mask_image=False, image_token_type_idx=1
            )
            infer2 = self.pl_module.infer(
                batch, mask_text=False, mask_image=False, image_token_type_idx=2
            )
            
            # NlVR2 output
            cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
            nlvr2_logits = self.pl_module.nlvr2_classifier(cls_feats)
            nlvr2_labels = batch["answers"]
            nlvr2_labels = torch.tensor(nlvr2_labels).to(self.pl_module.device).long()
            nlvr2_loss   = F.cross_entropy(nlvr2_logits, nlvr2_labels)
            nlvr2_loss.backward()
       
        grads   = emb_grads[0].cpu().numpy() ### To reflect if it's correct---
        # Shape is [batch_size,len_txt,768] # this is grads shape (8, 40, 768)
        
        grads_z = pro_grads[0].detach()
        
        embedding_layer.weight.requires_grad = original_state_emb
        projector_layer.weight.requires_grad = original_state_pro 
        emb_hook.remove()
        pro_hook.remove()

        all_representation = nlvr2_logits
        return grads_z, grads, all_representation

    def compute_word_importance  (self, 
                                  words, 
                                  input_ids,
                                  text_masks,
                                  text,
                                  batch,
                                  batch_size,
                                  device
                                 ):
        
        grads_z, grads, all_representation = self.get_grad(input_ids,
                                                            text_masks,
                                                            text,
                                                            batch,
                                                            device,
                                                           )
        
        sep_idx = (input_ids == self.tokenizer._convert_token_to_id('[SEP]')).nonzero()
        assert len(sep_idx)  == batch_size                                            
        
        replace_idx = [] # Word with biggest grad norm first for each sentence in the batch 
        for i in range(batch_size):
            temp_idx = None
            #grads[i][1:] because we don't want the [CLS] token (not include in dict words_to--) 
            norms = self.get_important_scores(grads[i][1:], self.words_to_sub_words[i])
            # Return the k biggest score in norm for each sentence (Only the indices)
            # eg. tensor([ 6, 12, 11, 10, 4,  5])
            indices = torch.topk(torch.tensor(norms), k=len(norms)).indices
            max_len = int(sep_idx[i][1] * 0.2)  # We change at most 20 % of the words
            
            for idx in indices:
                idx_int  = idx.item() 
                if self.check_word(words[i][idx_int].strip().lower()):
                    continue
                if (self.sim_word2id is not None) and not (words[i][idx_int].strip().lower() in self.sim_word2id):
                    continue
                if idx.item() in self.replace_history[i]:
                    continue
                if len(self.replace_history[i]) >= min(max_len, self.max_loops):
                    continue
                temp_idx = idx_int
                break
            if temp_idx is None:
                temp_idx = indices[0].item() 
                temp_idx = None 
            # After filtering : take word with highest salienty score :: Norm_grad
            replace_idx.append(temp_idx)
            if temp_idx is not None:  
                self.replace_history[i].add(temp_idx)
        
        return replace_idx, grads_z, all_representation

    def construct_new_samples(self,      
                              word_idx, 
                              words,  
                              batch_size): 
        """
        Creat all_num new sentence per original sentence with every candidat select by
        get_synonym_by_cos
        """
        ori_words = deepcopy(words)             
        all_new_text = []
        all_num = []
        
        if self.synonym == 'synonym':
            for i in range(batch_size):
                # print(i, ori_words[i][word_idx[i]])
                candidates = self.get_synonym(ori_words[i][word_idx[i]])
                for new_word in candidates:
                    # print(new_word)
                    ori_words[i][word_idx[i]] = new_word
                    all_new_text.append(' '.join(ori_words[i]))
                all_num.append(len(candidates))
            return all_new_text, all_num
        
        if self.synonym == 'cos_sim':
            for i in range(batch_size):
                if word_idx[i] is not None :        #<<<---- changed
                    candidates   = self.get_synonym_by_cos(ori_words[i][word_idx[i]])
                    nbr_candidat = len(candidates)  #<<<---- changed
                    
                    for new_word in candidates:
                        # print(new_word)
                        ori_words[i][word_idx[i]] = new_word
                        all_new_text.append(' '.join(ori_words[i]))
                else : #<<<---- changed
                    nbr_candidat = 1 #<<<---- changed
                    all_new_text.append(' '.join(ori_words[i])) #<<<---- changed
                all_num.append(nbr_candidat)#<<<---- changed
            return all_new_text, all_num
        raise ValueError("Only use wordnet of cos sim to find new words!")           
    
    def calc_words_to_sub_words(self, words, batch_size):
        """Creat a dictionary with the position of each words for each sentences"""
        self.words_to_sub_words = []
        for i in range(batch_size): 
            position = 0
            # Creat dictionary for each sentes
            self.words_to_sub_words.append({})
            # Loop for each words of a sentences
            for idx in range(len(words[i])):
                length = len(self.tokenizer.tokenize(words[i][idx]))               
                self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                position += length

    def split_forward(self,batch_c,device):
        """Do a Forward pass to get the Joint Representation"""
        with torch.no_grad():
            infer1 = self.pl_module.infer(
                batch_c, mask_text=False, mask_image=False, image_token_type_idx=1
            )
            infer2 = self.pl_module.infer(
                batch_c, mask_text=False, mask_image=False, image_token_type_idx=2
            )
            
            # NlVR2 output
            cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
            nlvr2_logits = self.pl_module.nlvr2_classifier(cls_feats)
            
        all_representation = nlvr2_logits
        return all_representation 
                                            
    def adv_attack_samples(self, 
                           pl_module,            
                           batch) : 

        self.device    = pl_module.device
        self.criterion = nn.CrossEntropyLoss().cuda(self.device)
        batch_size     = batch["text_ids"].size(0)
       
        txt_input_ids  = deepcopy(batch["text_ids"])     
        text_masks     = deepcopy(batch["text_masks"])   
        text           = deepcopy(batch["text"])                 
        original_words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).split(" ") 
                                                for ids in txt_input_ids]    
        cur_words = deepcopy(original_words)
        
        # Creat a dictionary with the position of each words for each sentences
        self.calc_words_to_sub_words(cur_words, batch_size)
        self.pl_module = pl_module

        self.replace_history = [set() for _ in range(batch_size)]
        for iter_idx in range(self.max_loops):
            # ori_z    : all_representation 
            # vector_z : gradient_projector (project.text.linear2)
            replace_idx, vector_z, ori_z = self.compute_word_importance(words = cur_words,
                                                                 input_ids    = txt_input_ids,
                                                                 text_masks   = text_masks,
                                                                 text         = text,
                                                                 batch        = batch,
                                                                 batch_size   = batch_size,
                                                                 device       = self.device
                                                                        )
            all_new_text, all_num = self.construct_new_samples(word_idx     = replace_idx,
                                                               words        = cur_words,
                                                               batch_size   = batch_size)             
            all_new_image0          = []
            all_new_image1          = []
            all_new_answers         = []
            all_new_table_name      = []
            all_new_txt_Labels      = []
            all_new_txt_ids_mlm     = []
            all_new_txt_labels_mlm  = []
            
            for idx,count  in enumerate (all_num) :
                all_new_image0.extend([batch['image_0'][0][idx]for _ in range(count)])
                all_new_image1.extend([batch['image_1'][0][idx]for _ in range(count)])
                all_new_answers.extend([batch['answers'][idx]for _ in range(count)])
                all_new_table_name.extend([batch['table_name'][idx]for _ in range(count)])
                all_new_txt_Labels.extend([batch['text_labels'][idx]for _ in range(count)])
                all_new_txt_ids_mlm.extend([batch['text_ids_mlm'][idx]for _ in range(count)])
                all_new_txt_labels_mlm.extend([batch['text_labels_mlm'][idx]for _ in range(count)])

            # Get the correct format
            all_new_image0          = [torch.stack(all_new_image0)]
            all_new_image1          = [torch.stack(all_new_image1)]
            all_new_txt_Labels      = torch.stack(all_new_txt_Labels)
            all_new_txt_ids_mlm     = torch.stack(all_new_txt_ids_mlm)
            all_new_txt_labels_mlm  = torch.stack(all_new_txt_labels_mlm)
            
            all_new_input_ids, all_new_text_masks = self.get_inputs(all_new_text,
                                                                   self.tokenizer,
                                                                   self.device)             
                
            batch_c                   = {}            
            batch_c['text']           = all_new_text
            batch_c['text_ids']       = all_new_input_ids
            batch_c['text_masks']     = all_new_text_masks
            batch_c['image_0']        = all_new_image0
            batch_c['image_1']        = all_new_image1
            batch_c['answers']        = all_new_answers
            batch_c['table_name']     = all_new_table_name
            batch_c['text_labels']    = all_new_txt_Labels
            batch_c['text_ids_mlm']   = all_new_txt_ids_mlm
            batch_c['text_labels_mlm']= all_new_txt_labels_mlm       
            outputs = self.split_forward(batch_c,self.device)  # torch.Size([sum(all_num), 768])
            
            outputs = torch.split(outputs, all_num)
            count = 0
            for i, cur_z in enumerate(outputs):
                # New-orginal all representation
                cur_z        = cur_z.float() - ori_z[i].float()
                # repeat grad_proj 
                z            = torch.repeat_interleave(vector_z[i].float().unsqueeze(0), 
                                                repeats=len(cur_z), dim=0)
                cur_z_norm   = cur_z.norm(dim=1)
                
                cosin_z      = self.cosine_similarity(cur_z, z)
                project_z    = torch.mul(cur_z_norm, cosin_z)
                selected_idx = torch.argmax(project_z)
                if project_z[selected_idx] > 0:
                    cur_words[i] = \
                    all_new_text[int(selected_idx)+count].split(' ')
                    self.words_to_sub_words[i] = {}
                    position = 0
                    for idx in range(len(cur_words[i])):
                        length = len(self.tokenizer.tokenize(cur_words[i][idx]))
                        if position + length >=  self.max_length : # Avoid to get sentence bigger than limit
                            continue 
                        self.words_to_sub_words[i][idx] =\
                        np.arange(position, position+length)
                        position += length
                            
                count += len(cur_z)            
            text = [' '.join(x) for x in cur_words]
            txt_input_ids, text_masks=\
            self.get_inputs(text,self.tokenizer,self.device)             
        return {'txt_input_ids' : txt_input_ids,
                'text_masks'    : text_masks ,
                'text'          : text}