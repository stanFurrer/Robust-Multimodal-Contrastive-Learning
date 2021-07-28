import sys
import nltk
import os
import string
from copy import deepcopy
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CosineSimilarity
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

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


class GreedyAttack:
    def __init__(self, config, contrastive_framework=None):
        self.pl_module         = None
        self.contrastive_framework = contrastive_framework
        self.stopwords         = set(stopwords.words('english'))
        self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)
        self.tokenizer         = BertTokenizer.from_pretrained(config["tokenizer"])
        self.device            = None
        self.words_to_sub_words= None
        self.max_length        = config["max_text_len"]
        self.n_candidates      = config["n_candidates"]
        self.max_loops         = config["max_loops"]
        self.sim_thred         = config["sim_thred"]
        self.word2id           = self.tokenizer.get_vocab()              
        self.id2word           = {v: k for k, v in self.word2id.items()}         
        self.cos_sim           = None                                    
        self.sim_word2id       = None 
        self.sim_id2word       = None
        self.synonym           = config["synonym"]
        self.cos_sim_dict      = None
        if config["cos_sim"]:
            self.init_matrix(config["embedding_path"], config["sim_path"])
        # a mini ViLTransformerSS
        self.max_image_len = config["max_image_len"]
        self.text_embeddings = None
        self.transformer = None
        self.token_type_embeddings = None
        self.pooler = None
        
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
        if os.path.exists(sim_path):
            self.cos_sim = np.load(sim_path,allow_pickle=True)
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

    def build_mini_vilt(self, pl_module):
        raise NotImplementedError(f"Build_mini_vilt of {self.contrastive_framework} isn't implemented.")
        
    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
    
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
    
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]  # [0] : Because it's a list of one element
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.max_image_len,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        # image_embeds.shape : [64 217 768] :: [batch,patch,hiddensize]
        # patch_index shape  : ([64 217 2]), (19,19)) (patch_index, (H,W))
    
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
    
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)
    
        x = co_embeds
    
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
    
        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1]:],
        )
        if self.pooler is not None:
            cls_feats = self.pooler(x)
        else:
            cls_feats = None
    
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
    
        return ret
    
    def vilt_zero_grad(self):
        raise NotImplementedError(f"vilt_zero_grad of {self.contrastive_framework} isn't implemented.")
    
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

        return list(candidates)[:self.n_candidates]
    
    def get_important_scores(self, grads, words_to_sub_words):
        index_scores = [0.0] * len(words_to_sub_words)
        for i in range(len(words_to_sub_words)):
            matched_tokens  = words_to_sub_words[i]
            agg_grad        = np.mean(grads[matched_tokens], axis=0)

            index_scores[i] = np.linalg.norm(agg_grad, ord=1)
        return index_scores
       
    def get_inputs(self,sentences, tokenizer, device):
        outputs        = tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length)
        input_ids      = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]

        return torch.tensor(input_ids).to(device),torch.tensor(attention_mask).to(device)
         
    def check_word(self, word):
        return word == '[PAD]' or word == '[UNK]' or word == '[CLS]' or \
               word == '[SEP]' or word in self.stopwords or word in string.punctuation or \
               word in filter_words or word in '...' or word == '[MASK]'

    def get_grad(self,
                input_ids,
                text_masks,
                text,
                batch,
                device,
                k_modality, # k_text
    ):
        raise NotImplementedError(f"get_grad of {self.contrastive_framework} isn't implemented.")   
    

    def compute_word_importance  (self,
                                  words, 
                                  input_ids,
                                  text_masks,
                                  text,
                                  batch,
                                  batch_size,
                                  device,
                                  k_modality=None # k_text=None
                                 ):
        
        loss_z, grads, text_representation = self.get_grad(input_ids,
                                                            text_masks,
                                                            text,
                                                            batch,
                                                            device,
                                                            k_modality # k_text
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
        
        return replace_idx, loss_z, text_representation

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
        changed = []
        
        if self.synonym == 'synonym':
            for i in range(batch_size):
                if word_idx[i] is not None:
                    candidates = self.get_synonym(ori_words[i][word_idx[i]])
                    for new_word in candidates:
                        # print(new_word)
                        ori_words[i][word_idx[i]] = new_word
                        all_new_text.append(' '.join(ori_words[i]))
                    all_num.append(len(candidates))
                    changed.append(True)
                else:
                    all_new_text.append(' '.join(ori_words[i]))
                    all_num.append(1)
                    changed.append(False)
            return all_new_text, all_num
        
        if self.synonym == 'cos_sim':
            for i in range(batch_size):
                if word_idx[i] is not None :        
                    candidates   = self.get_synonym_by_cos(ori_words[i][word_idx[i]])
                    nbr_candidat = len(candidates)
                    
                    for new_word in candidates:
                        # print(new_word)
                        ori_words[i][word_idx[i]] = new_word
                        all_new_text.append(' '.join(ori_words[i]))
                    changed.append(True)
                else :
                    nbr_candidat = 1 
                    all_new_text.append(' '.join(ori_words[i]))
                    changed.append(False)
                all_num.append(nbr_candidat)
            return all_new_text, all_num, changed
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
                if position + length >= self.max_length:  # if Sentence too big
                    break
                self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                position += length
                
    def split_forward(self,batch, all_num, ori_z, k_image):
        """Do a Forward pass to get the text Representation"""
        raise NotImplementedError(f"split_forward of {self.contrastive_framework} isn't implemented.")
        
    def adv_attack_samples(self, 
                           pl_module,            
                           batch,           
                           k_image, # k_text
                          ):
        raise NotImplementedError(f"adv_attack_samples of {self.contrastive_framework} isn't implemented.")  
              
        
class GreedyAttack_moco(GreedyAttack):
    def __init__(self, config):
        super().__init__(config, "moco")
        # a mini ViLTransformerSS
        self.moco_head = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.moco_head = deepcopy(pl_module.moco_head)
        
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.moco_head.zero_grad()
    
    def get_grad(self,
                 input_ids,
                 text_masks,
                 text,
                 batch,
                 device,
                 k_modality, # k_text
                 ):
        embedding_layer = self.text_embeddings.word_embeddings  # word_embeddings
        #projector_layer = self.moco_head.txt_model.linear2 ##
        #projector_layer = self.moco_head.model.linear2
        
        #original_state_pro = projector_layer.weight.requires_grad
        #projector_layer.weight.requires_grad = True
        
        original_state_emb = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True
        
        emb_grads = []
        #pro_grads = []
        
        def emb_grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        
        #def pro_grad_hook(module, grad_in, grad_out):
        #    pro_grads.append(grad_out[0])
        
        emb_hook = embedding_layer.register_full_backward_hook(emb_grad_hook)
        #pro_hook = projector_layer.register_full_backward_hook(pro_grad_hook)
        
        self.vilt_zero_grad()
        
        with torch.enable_grad():
            batch["text_ids"] = input_ids
            batch["text_masks"] = text_masks
            batch["text"] = text
            
            infer = self.infer(batch, mask_text=False, mask_image=False)
            image_representation_q, text_representation_q = self.moco_head(infer['image_feats'], infer['text_feats'])
            q_attacked = nn.functional.normalize(text_representation_q, dim=1)
            
            ################ RMCL #################
            l_pos = torch.einsum('nc,nc->n', [q_attacked, k_modality]).unsqueeze(-1) # k_text
            l_neg = torch.einsum('nc,ck->nk', [q_attacked, self.pl_module.text_queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.pl_module.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            labels = labels.type_as(logits)
            loss = self.criterion(logits.float(), labels.long())
            # print("loss", loss)
            loss.backward()
            ################ RMCL #################
        
        grads = emb_grads[0].cpu().numpy()
        # Shape is [batch_size,len_txt,768]
        #grads_z = pro_grads[0].detach()
        
        embedding_layer.weight.requires_grad = original_state_emb
        #projector_layer.weight.requires_grad = original_state_pro
        emb_hook.remove()
        #pro_hook.remove()
        
        #text_representation = text_representation_q
        #return grads_z, grads, text_representation
        return loss, grads, q_attacked
    
    def split_forward(self, batch, all_num, ori_z, k_modality):
        """Do a Forward pass to get the text Representation"""
        with torch.no_grad():
            infer = self.infer(batch, mask_text=False, mask_image=False)
            _, text_representation_q = self.moco_head(infer['image_feats'], infer['text_feats'])
            q_attacked = nn.functional.normalize(text_representation_q, dim=1)
            text_representation = torch.split(q_attacked, all_num)
            
            all_loss = []
            for i, txt_split in enumerate(text_representation):
                cur_loss = []
                cur_max_loss, cur_max_loss_idx = -1, -1
                t_save = ori_z[i]
                for j, txt in enumerate(txt_split):
                    ori_z[i] = txt
                    ################ RMCL #################
                    l_pos = torch.einsum('nc,nc->n', [ori_z, k_modality]).unsqueeze(-1)
                    l_neg = torch.einsum('nc,ck->nk', [ori_z, self.pl_module.text_queue.clone().detach()])
                    logits = torch.cat([l_pos, l_neg], dim=1)
                    logits /= self.pl_module.temperature
                    labels = torch.zeros(logits.shape[0], dtype=torch.long)
                    labels = labels.type_as(logits)
                    loss = self.criterion(logits.float(), labels.long())
                    cur_loss.append(loss)
                    if loss > cur_max_loss:
                        cur_max_loss, cur_max_loss_idx = loss, j
                    ################ RMCL #################
                all_loss.append((cur_loss, cur_max_loss_idx))
                ori_z[i] = t_save

        # print(all_num)
        # print([len(x[0]) for x in all_loss])
        return all_loss    
     
    #def split_forward(self, batch):
    #    """Do a Forward pass to get the text Representation"""
    #    with torch.no_grad():
    #        infer = self.infer(batch, mask_text=False, mask_image=False)
    #        image_representation_q, text_representation_q = self.moco_head(infer['image_feats'], infer['text_feats'])
        
    #    text_representation = text_representation_q
    #    return text_representation  
    
    def adv_attack_samples(self,
                           pl_module,
                           batch,
                           k_modality,
                           ):
        
        self.device = pl_module.device
        self.criterion = nn.CrossEntropyLoss().cuda(self.device)
        batch_size = batch["text_ids"].size(0)
        
        txt_input_ids = deepcopy(batch["text_ids"])
        text_masks = deepcopy(batch["text_masks"])
        text = deepcopy(batch["text"])
        original_words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).split(" ")
                          for ids in txt_input_ids]
        cur_words = deepcopy(original_words)
        
        # Creat a dictionary with the position of each words for each sentences
        self.calc_words_to_sub_words(cur_words, batch_size)
        self.build_mini_vilt(pl_module)
        
        self.replace_history = [set() for _ in range(batch_size)]
        # Test
        changes_verification = [0] * batch_size #
        
        
        for iter_idx in range(self.max_loops):
            # ori_z    : text_representation
            # vector_z : gradient_projector (project.text.linear2)
            # loss_z   : loss
            # print(cur_words)
            replace_idx, loss_z, ori_z = self.compute_word_importance(words=cur_words,
                                                                        input_ids=txt_input_ids,
                                                                        text_masks=text_masks,
                                                                        text=text,
                                                                        batch=batch,
                                                                        batch_size=batch_size,
                                                                        device=self.device,
                                                                        k_modality=k_modality, # k_text=k_text,
                                                                        )
            
            all_new_text, all_num,changed = self.construct_new_samples(word_idx=replace_idx,
                                                               words=cur_words,
                                                               batch_size=batch_size)
            
            #print("This is all_num",all_num)
            #all_new_false_image_0 = []
            #all_new_replica = []
            #all_new_raw_index = []
            #all_new_cap_index = []
            #all_new_img_index = []
            all_new_image = []
            # all_new_iid             = []
            all_new_text_labels = []
            #all_new_text_ids_mlm = []
            #all_new_text_labels_mlm = []

            for idx, count in enumerate(all_num):
                #all_new_false_image_0.extend([batch['false_image_0'][0][idx] for _ in range(count)])
                #all_new_cap_index.extend([batch['cap_index'][idx] for _ in range(count)])
                all_new_image.extend([batch['image'][0][idx] for _ in range(count)])
                #all_new_replica.extend([batch['replica'][idx] for _ in range(count)])
                #all_new_img_index.extend([batch['img_index'][idx] for _ in range(count)])
                # all_new_iid.extend([batch['iid'][idx]for _ in range(count)])
                #all_new_raw_index.extend([batch['raw_index'][idx] for _ in range(count)])
                all_new_text_labels.extend([batch['text_labels'][idx] for _ in range(count)])
                #all_new_text_ids_mlm.extend([batch['text_ids_mlm'][idx] for _ in range(count)])
                #all_new_text_labels_mlm.extend([batch['text_labels_mlm'][idx] for _ in range(count)])
                
            # Get the correct format
            #all_new_false_image_0 = [torch.stack(all_new_false_image_0)]
            all_new_image = [torch.stack(all_new_image)]
            all_new_text_labels = torch.stack(all_new_text_labels)
            #all_new_text_ids_mlm = torch.stack(all_new_text_ids_mlm)
            #all_new_text_labels_mlm = torch.stack(all_new_text_labels_mlm)
            
            # Get the inputs_ids
            all_new_text_ids, all_new_text_masks = self.get_inputs(all_new_text,
                                                                   self.tokenizer,
                                                                   self.device)   
            batch_c = {}
            #batch_c['false_image_0'] = all_new_false_image_0
            #batch_c['cap_index'] = all_new_cap_index
            batch_c['image'] = all_new_image
            #batch_c['replica'] = all_new_replica
            #batch_c['img_index'] = all_new_img_index
            # batch_c['iid']             = all_new_iid
            #batch_c['raw_index'] = all_new_raw_index
            batch_c['text_labels'] = all_new_text_labels
            #batch_c['text_ids_mlm'] = all_new_text_ids_mlm
            #batch_c['text_labels_mlm'] = all_new_text_labels_mlm
            batch_c['text'] = all_new_text
            batch_c['text_ids'] = all_new_text_ids
            batch_c['text_masks'] = all_new_text_masks
            
            outputs = self.split_forward(batch_c, all_num, ori_z, k_modality)
            
            #outputs = self.split_forward(batch_c)
            #outputs = torch.split(outputs, all_num)
            count   = 0
            
            for i, (cur_z, selected_idx) in enumerate(outputs):
                if changed[i] == False :
                    count += len(cur_z)
                    continue
                    
                if cur_z[selected_idx] > 0:
                    changes_verification[i]+=1 #
                    cur_words[i] = all_new_text[int(selected_idx) + count].split(' ')
                    self.words_to_sub_words[i] = {}
                    position = 0
                    for idx in range(len(cur_words[i])):
                        length = len(self.tokenizer.tokenize(cur_words[i][idx]))
                        if position + length >= self.max_length:  # if Sentence too big
                            break
                        self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                        position += length
                count += len(cur_z)
            text = [' '.join(x) for x in cur_words]
            txt_input_ids, text_masks = \
                self.get_inputs(text, self.tokenizer, self.device)
        
        num_changes = []
        change_rate = []
        Problem     = False
        for old_words, new_words in zip(original_words, cur_words):
            changes = sum(~(np.array(old_words) == np.array(new_words)))
            if changes ==0 :
                Problem = True
            num_changes.append(changes)
            change_rate.append(changes / len(old_words))
            
        #print("\n-----------This is the np.mean(num_changes)",np.mean(num_changes))            
        return {'txt_input_ids' : txt_input_ids,
                'text_masks'    : text_masks ,
                'text'          : text,
                'num_changes'   : np.mean(num_changes),
                'change_rate'   : np.mean(change_rate),
                'Problem'       : Problem,
                'changes_verification'       : changes_verification} 

class GreedyAttack_barlowtwins(GreedyAttack):
    def __init__(self, config):
        super().__init__(config, "barlowtwins")
        # a mini ViLTransformerSS
        self.barlowtwins_head = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.barlowtwins_head = deepcopy(pl_module.barlowtwins_head)
    
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.barlowtwins_head.zero_grad()
    
    def get_grad(self,
                 input_ids,
                 text_masks,
                 text,
                 batch,
                 device,
                 k_modality=None # k_text=None
                 ):
        
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        embedding_layer = self.text_embeddings.word_embeddings  # word_embeddings
        #projector_layer = self.barlowtwins_head.projector.linear3
        
        #original_state_pro = projector_layer.weight.requires_grad
        #projector_layer.weight.requires_grad = True
        
        original_state_emb = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True
        
        emb_grads = []
        #pro_grads = []
        
        def emb_grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        
        #def pro_grad_hook(module, grad_in, grad_out):
        #    pro_grads.append(grad_out[0])
        
        emb_hook = embedding_layer.register_full_backward_hook(emb_grad_hook)
        #pro_hook = projector_layer.register_full_backward_hook(pro_grad_hook)
        
        self.vilt_zero_grad()
        
        with torch.enable_grad():
            batch["text_ids"] = input_ids
            batch["text_masks"] = text_masks
            batch["text"] = text
            
            infer = self.infer(batch, mask_text=False, mask_image=False)
            image_representation, text_representation = self.barlowtwins_head(infer['image_feats'], infer['text_feats'])
            
            ################ RMCL #################
            # c = text_representation.T @ k_text
            c = torch.mm(text_representation.T, k_modality) / text_representation.shape[0] #k_text

            # c.div_(pl_module.per_step_bs)
            # torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()

            loss = on_diag + self.pl_module.adv_lr * off_diag  # / self.pl_module.loss_weight
            # print(loss)
            loss.backward()
            ################ RMCL #################
        
        grads = emb_grads[0].cpu().numpy()
        # Shape is [batch_size,len_txt,768]
        # grads_z = pro_grads[1].detach()
        
        embedding_layer.weight.requires_grad = original_state_emb
        # projector_layer.weight.requires_grad = original_state_pro
        emb_hook.remove()
        # pro_hook.remove()
        
        #text_representation = text_representation
        return loss, grads, (image_representation, text_representation)
    
    def split_forward(self, batch, all_num, ori_z, k_modality):
        """Do a Forward pass to get the text Representation"""
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        all_loss = []
        with torch.no_grad():
            infer = self.infer(batch, mask_text=False, mask_image=False)
            image_representation, text_representation = self.barlowtwins_head(infer['image_feats'], infer['text_feats'])
            image_representation = torch.split(image_representation, all_num)
            text_representation = torch.split(text_representation, all_num)
            # print(all_num, len(image_splie), len(text_splie))
            for i, (img_split, txt_split) in enumerate(zip(image_representation, text_representation)):
                cur_loss = []
                cur_max_loss, cur_max_loss_idx = -1, -1
                t_save = (ori_z[0][i], ori_z[1][i])
                for j, (img, txt) in enumerate(zip(img_split, txt_split)):
                    ori_z[0][i], ori_z[1][i] = img, txt
                    ################ RMCL #################
                    # c = ori_z[1].T @ k_text
                    c = torch.mm(ori_z[1].T, k_modality) / ori_z[1].shape[0] # Yiran k_text

                    # c.div_(pl_module.per_step_bs)
                    # torch.distributed.all_reduce(c)

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).pow_(2).sum()

                    loss = on_diag + self.pl_module.adv_lr * off_diag  # / self.pl_module.loss_weight
                    cur_loss.append(loss)
                    if loss > cur_max_loss:
                        cur_max_loss, cur_max_loss_idx = loss, j
                    ################ RMCL #################
                all_loss.append((cur_loss, cur_max_loss_idx))
                ori_z[0][i], ori_z[1][i] = t_save[0], t_save[1]
            
        # print(all_num)
        # print([len(x[0]) for x in all_loss])
        return all_loss
    
    def adv_attack_samples(self, pl_module, batch, k_modality=None): #k_text
        
        self.device = pl_module.device
        self.criterion = nn.CrossEntropyLoss().cuda(self.device)
        batch_size = batch["text_ids"].size(0)
        
        txt_input_ids = deepcopy(batch["text_ids"])
        text_masks = deepcopy(batch["text_masks"])
        text = deepcopy(batch["text"])
        original_words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).split(" ")
                          for ids in txt_input_ids]
        cur_words = deepcopy(original_words)
        
        # Creat a dictionary with the position of each words for each sentences
        self.calc_words_to_sub_words(cur_words, batch_size)
        self.build_mini_vilt(pl_module)
        
        self.replace_history = [set() for _ in range(batch_size)]
        changes_verification = [0] * batch_size  #
        for iter_idx in range(self.max_loops):
            # ori_z    : text_representation
            # vector_z : gradient_projector (project.text.linear2)
            replace_idx, loss_z, ori_z = self.compute_word_importance(words=cur_words,
                                                                        input_ids=txt_input_ids,
                                                                        text_masks=text_masks,
                                                                        text=text,
                                                                        batch=batch,
                                                                        batch_size=batch_size,
                                                                        device=self.device,
                                                                        k_modality=k_modality #k_text
                                                                        )
            
            all_new_text, all_num,changed  = self.construct_new_samples(word_idx=replace_idx,
                                                               words=cur_words,
                                                               batch_size=batch_size)
            
            all_new_false_image_0 = []
            all_new_replica = []
            all_new_raw_index = []
            all_new_cap_index = []
            all_new_img_index = []
            all_new_image = []
            # all_new_iid             = []
            all_new_text_labels = []
            all_new_text_ids_mlm = []
            all_new_text_labels_mlm = []
            
            for idx, count in enumerate(all_num):
                all_new_false_image_0.extend([batch['false_image_0'][0][idx] for _ in range(count)])
                all_new_cap_index.extend([batch['cap_index'][idx] for _ in range(count)])
                all_new_image.extend([batch['image'][0][idx] for _ in range(count)])
                all_new_replica.extend([batch['replica'][idx] for _ in range(count)])
                all_new_img_index.extend([batch['img_index'][idx] for _ in range(count)])
                # all_new_iid.extend([batch['iid'][idx]for _ in range(count)])
                all_new_raw_index.extend([batch['raw_index'][idx] for _ in range(count)])
                all_new_text_labels.extend([batch['text_labels'][idx] for _ in range(count)])
                all_new_text_ids_mlm.extend([batch['text_ids_mlm'][idx] for _ in range(count)])
                all_new_text_labels_mlm.extend([batch['text_labels_mlm'][idx] for _ in range(count)])
            
            # Get the correct format
            all_new_false_image_0 = [torch.stack(all_new_false_image_0)]
            all_new_image = [torch.stack(all_new_image)]
            all_new_text_labels = torch.stack(all_new_text_labels)
            all_new_text_ids_mlm = torch.stack(all_new_text_ids_mlm)
            all_new_text_labels_mlm = torch.stack(all_new_text_labels_mlm)
            
            # Get the inputs_ids
            all_new_text_ids, all_new_text_masks = self.get_inputs(all_new_text,
                                                                   self.tokenizer,
                                                                   self.device)
            batch_c = {}
            batch_c['false_image_0'] = all_new_false_image_0
            batch_c['cap_index'] = all_new_cap_index
            batch_c['image'] = all_new_image
            batch_c['replica'] = all_new_replica
            batch_c['img_index'] = all_new_img_index
            # batch_c['iid']             = all_new_iid
            batch_c['raw_index'] = all_new_raw_index
            batch_c['text_labels'] = all_new_text_labels
            batch_c['text_ids_mlm'] = all_new_text_ids_mlm
            batch_c['text_labels_mlm'] = all_new_text_labels_mlm
            batch_c['text'] = all_new_text
            batch_c['text_ids'] = all_new_text_ids
            batch_c['text_masks'] = all_new_text_masks
            
            outputs = self.split_forward(batch_c, all_num, ori_z, k_modality)
            #outputs = torch.split(outputs, all_num)
            count = 0
            
            for i, (cur_z, selected_idx) in enumerate(outputs):
                if changed[i] == False:
                    count += len(cur_z)
                    continue
                    
                if cur_z[selected_idx] > loss_z:
                    changes_verification[i] += 1
                    cur_words[i] = all_new_text[int(selected_idx) + count].split(' ')
                    self.words_to_sub_words[i] = {}
                    position = 0
                    for idx in range(len(cur_words[i])):
                        length = len(self.tokenizer.tokenize(cur_words[i][idx]))
                        if position + length >= self.max_length:  # if Sentence too big
                            break
                        self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                        position += length
                
                count += len(cur_z)
            text = [' '.join(x) for x in cur_words]
            txt_input_ids, text_masks = self.get_inputs(text, self.tokenizer, self.device)
        
        num_changes = []
        change_rate = []
        Problem     = False
        for old_words, new_words in zip(original_words, cur_words):
            changes = sum(~(np.array(old_words) == np.array(new_words)))
            if changes ==0 :
                Problem = True
            num_changes.append(changes)
            change_rate.append(changes / len(old_words))
            
        # print(num_changes)
            
        return {'txt_input_ids' : txt_input_ids,
                'text_masks'    : text_masks ,
                'text'          : text,
                'num_changes'   : np.mean(num_changes),
                'change_rate'   : np.mean(change_rate),
                'Problem'       : Problem,
                'changes_verification': changes_verification}
    
class GreedyAttack_nlvr2(GreedyAttack):
    def __init__(self, config):
        super().__init__(config, "nlvr2")
        # a mini ViLTransformerSS
        self.nlvr2_classifier = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.pooler = deepcopy(pl_module.pooler)
        self.nlvr2_classifier = deepcopy(pl_module.nlvr2_classifier)
    
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.pooler.zero_grad()
        self.nlvr2_classifier.zero_grad()
    
    def get_grad(self,
                 input_ids,
                 text_masks,
                 text,
                 batch,
                 device,
                 k_modality,#k_text
                 ):
        embedding_layer = self.text_embeddings.word_embeddings  # word_embeddings
        # projector_layer = self.nlvr2_classifier.linear2_nlvr2
        
        # original_state_pro = projector_layer.weight.requires_grad
        # projector_layer.weight.requires_grad = True
        
        original_state_emb = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True
        
        emb_grads = []
        # pro_grads = []
        
        def emb_grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])
        
        # def pro_grad_hook(module, grad_in, grad_out):
        #     pro_grads.append(grad_out[0])
        
        emb_hook = embedding_layer.register_full_backward_hook(emb_grad_hook)
        # pro_hook = projector_layer.register_full_backward_hook(pro_grad_hook)
        
        self.vilt_zero_grad()
        
        with torch.enable_grad():
            batch["text_ids"] = input_ids
            batch["text_masks"] = text_masks
            batch["text"] = text

            infer1 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
            infer2 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)
            # NlVR2 output
            cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
            nlvr2_logits = self.nlvr2_classifier(cls_feats)
            # Compute the cross-entropy
            nlvr2_labels = batch["answers"]
            nlvr2_labels = torch.tensor(nlvr2_labels).to(self.pl_module.device).long()
            loss = self.criterion(nlvr2_logits, nlvr2_labels)
            loss.backward()
        
        grads = emb_grads[0].cpu().numpy()
        # Shape is [batch_size,len_txt,768]
        # grads_z = pro_grads[0].detach()
        
        embedding_layer.weight.requires_grad = original_state_emb
        # projector_layer.weight.requires_grad = original_state_pro
        emb_hook.remove()
        # pro_hook.remove()
        
        return loss, grads, (infer1["cls_feats"], infer2["cls_feats"])
    
    def split_forward(self, batch, all_num, ori_z):
        """Do a Forward pass to get the text Representation"""
        with torch.no_grad():
            infer1 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
            infer2 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)
            infer1_cls_feats = torch.split(infer1["cls_feats"], all_num)
            infer2_cls_feats = torch.split(infer2["cls_feats"], all_num)

            all_loss = []
            for i in range(len(all_num)):
                cur_loss = []
                cur_max_loss, cur_max_loss_idx = -1, -1
                t_save = (ori_z[0][i], ori_z[1][i])
                for j in range(all_num[i]):
                    ori_z[0][i] = infer1_cls_feats[i][j]
                    ori_z[1][i] = infer2_cls_feats[i][j]
                    
                    cls_feats = torch.cat([ori_z[0], ori_z[1]], dim=-1)
                    nlvr2_logits = self.nlvr2_classifier(cls_feats)
                    # Compute the cross-entropy
                    nlvr2_labels = batch["answers"]
                    nlvr2_labels = torch.tensor(nlvr2_labels).to(self.pl_module.device).long()
                    loss = self.criterion(nlvr2_logits, nlvr2_labels)
                    
                    cur_loss.append(loss)
                    if loss > cur_max_loss:
                        cur_max_loss, cur_max_loss_idx = loss, j
                        
                all_loss.append((cur_loss, cur_max_loss_idx))
                ori_z[0][i], ori_z[1][i] = t_save[0], t_save[1]
        
        # print(all_num)
        # print([len(x[0]) for x in all_loss])
        return all_loss
    
    def adv_attack_samples(self,
                           pl_module,
                           batch,
                           k_modality, #k_text
                           ):
        
        self.device = pl_module.device
        self.criterion = nn.CrossEntropyLoss().cuda(self.device)
        batch_size = batch["text_ids"].size(0)
        
        txt_input_ids = deepcopy(batch["text_ids"])
        text_masks = deepcopy(batch["text_masks"])
        text = deepcopy(batch["text"])
        original_words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False).split(" ")
                          for ids in txt_input_ids]
        cur_words = deepcopy(original_words)
        
        # Creat a dictionary with the position of each words for each sentences
        self.calc_words_to_sub_words(cur_words, batch_size)
        self.build_mini_vilt(pl_module)
        
        self.replace_history = [set() for _ in range(batch_size)]
        # Test
        changes_verification = [0] * batch_size  #
        
        for iter_idx in range(self.max_loops):
            # ori_z    : text_representation
            # vector_z : gradient_projector (project.text.linear2)
            # print(iter_idx)
            replace_idx, loss_z, ori_z = self.compute_word_importance(words=cur_words,
                                                                      input_ids=txt_input_ids,
                                                                      text_masks=text_masks,
                                                                      text=text,
                                                                      batch=batch,
                                                                      batch_size=batch_size,
                                                                      device=self.device,
                                                                      k_modality=k_modality, #k_text
                                                                      )
            
            all_new_text, all_num, changed = self.construct_new_samples(word_idx=replace_idx,
                                                                        words=cur_words,
                                                                        batch_size=batch_size)
            
            # print("This is all_num",all_num)
            # all_new_false_image_0 = []
            # all_new_replica = []
            # all_new_raw_index = []
            # all_new_cap_index = []
            # all_new_img_index = []
            all_new_image_0 = []
            all_new_image_1 = []
            # all_new_iid             = []
            all_new_text_labels = []
            # all_new_text_ids_mlm = []
            # all_new_text_labels_mlm = []
            
            for idx, count in enumerate(all_num):
                # all_new_false_image_0.extend([batch['false_image_0'][0][idx] for _ in range(count)])
                # all_new_cap_index.extend([batch['cap_index'][idx] for _ in range(count)])
                all_new_image_0.extend([batch['image_0'][0][idx] for _ in range(count)])
                all_new_image_1.extend([batch['image_1'][0][idx] for _ in range(count)])
                # all_new_replica.extend([batch['replica'][idx] for _ in range(count)])
                # all_new_img_index.extend([batch['img_index'][idx] for _ in range(count)])
                # all_new_iid.extend([batch['iid'][idx]for _ in range(count)])
                # all_new_raw_index.extend([batch['raw_index'][idx] for _ in range(count)])
                all_new_text_labels.extend([batch['text_labels'][idx] for _ in range(count)])
                # all_new_text_ids_mlm.extend([batch['text_ids_mlm'][idx] for _ in range(count)])
                # all_new_text_labels_mlm.extend([batch['text_labels_mlm'][idx] for _ in range(count)])
            
            # Get the correct format
            # all_new_false_image_0 = [torch.stack(all_new_false_image_0)]
            all_new_image_0 = [torch.stack(all_new_image_0)]
            all_new_image_1 = [torch.stack(all_new_image_1)]
            all_new_text_labels = torch.stack(all_new_text_labels)
            # all_new_text_ids_mlm = torch.stack(all_new_text_ids_mlm)
            # all_new_text_labels_mlm = torch.stack(all_new_text_labels_mlm)
            
            # Get the inputs_ids
            all_new_text_ids, all_new_text_masks = self.get_inputs(all_new_text,
                                                                   self.tokenizer,
                                                                   self.device)
            batch_c = {}
            # batch_c['false_image_0'] = all_new_false_image_0
            # batch_c['cap_index'] = all_new_cap_index
            batch_c['image_0'] = all_new_image_0
            batch_c['image_1'] = all_new_image_1
            batch_c['answers'] = batch['answers']
            # batch_c['replica'] = all_new_replica
            # batch_c['img_index'] = all_new_img_index
            # batch_c['iid']             = all_new_iid
            # batch_c['raw_index'] = all_new_raw_index
            batch_c['text_labels'] = all_new_text_labels
            # batch_c['text_ids_mlm'] = all_new_text_ids_mlm
            # batch_c['text_labels_mlm'] = all_new_text_labels_mlm
            batch_c['text'] = all_new_text
            batch_c['text_ids'] = all_new_text_ids
            batch_c['text_masks'] = all_new_text_masks
            
            outputs = self.split_forward(batch_c, all_num, ori_z)
            count = 0
            
            for i, (cur_z, selected_idx) in enumerate(outputs):
                if changed[i] == False:
                    count += len(cur_z)
                    continue
                
                if cur_z[selected_idx] > 0:
                    changes_verification[i] += 1  #
                    cur_words[i] = all_new_text[int(selected_idx) + count].split(' ')
                    self.words_to_sub_words[i] = {}
                    position = 0
                    for idx in range(len(cur_words[i])):
                        length = len(self.tokenizer.tokenize(cur_words[i][idx]))
                        if position + length >= self.max_length:  # if Sentence too big
                            break
                        self.words_to_sub_words[i][idx] = np.arange(position, position + length)
                        position += length
                count += len(cur_z)
            text = [' '.join(x) for x in cur_words]
            txt_input_ids, text_masks = \
                self.get_inputs(text, self.tokenizer, self.device)
        
        num_changes = []
        change_rate = []
        Problem = False
        for old_words, new_words in zip(original_words, cur_words):
            changes = sum(~(np.array(old_words) == np.array(new_words)))
            if changes == 0:
                Problem = True
            num_changes.append(changes)
            change_rate.append(changes / len(old_words))
        
        # print(num_changes)
        return {'txt_input_ids': txt_input_ids,
                'text_masks': text_masks,
                'text': text,
                'num_changes': np.mean(num_changes),
                'change_rate': np.mean(change_rate),
                'Problem': Problem,
                'changes_verification': changes_verification}          