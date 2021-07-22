from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BertTokenizer#
from augmentation.eda import eda
from sentence_transformers import SentenceTransformer,util
import torch
class TextAugmentation : 
    def __init__(self, config):
        super().__init__()
        self.num_beams = config["num_beams"]
        self.num_return_sequences = config["num_return_sequences"]
        self.max_length = config["max_text_len"]
        self.type_txt_augm = config["type_txt_augm"]
        self.tokenizer =  BertTokenizer.from_pretrained(config["tokenizer"])
        self.tokenizer_pegasus = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
        self.pegasus = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase')  
        self.model_sentence_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')        
    
    def augmentation(self,pl_module, batch):
        epoch =  pl_module.current_epoch
        self.pegasus = self.pegasus.to(pl_module.device)
        txt_input  = []
        text_masks = []
        final_sentences = []
        for sentence in batch["text"] :
            if "PEGASUS" in self.type_txt_augm  :
                batch_pegasus = self.tokenizer_pegasus(sentence, 
                                                         truncation=True, padding='longest', return_tensors="pt").to(pl_module.device)
                translated = self.pegasus.generate(**batch_pegasus,max_length=self.max_length,
                                                   num_beams=self.num_beams, num_return_sequences=self.num_return_sequences, 
                                                   temperature=1.5).to(pl_module.device)
                augmented_text = self.tokenizer_pegasus.batch_decode(translated, skip_special_tokens=True)            
            if "EDA" in self.type_txt_augm :
                augmented_text.extend(eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=self.num_return_sequences))             
            augmented_text_embeddings = self.model_sentence_embedding.encode(augmented_text,show_progress_bar=False)
            original_text_embeddings  = self.model_sentence_embedding.encode(sentence,show_progress_bar=False)
            cosine_scores = util.pytorch_cos_sim(original_text_embeddings, augmented_text_embeddings)            
            values, indices = torch.sort(cosine_scores,descending =True) 
            final_sentences.append(augmented_text[int(indices[0][epoch])])

        #for i,sentence in enumerate(batch["text"]) : 
        #    print("Original sentence :::", sentence)
        #    print("Augmented sentence:::", final_sentences[i]) 
        #    print("\n")
       
        outputs = self.tokenizer(final_sentences, truncation=True, padding=True, max_length=self.max_length)
        batch["text"] = augmented_text
        batch["text_ids"]= torch.tensor(outputs["input_ids"]).to(pl_module.device)
        batch["text_masks"]= torch.tensor(outputs["attention_mask"]).to(pl_module.device)  
        return batch