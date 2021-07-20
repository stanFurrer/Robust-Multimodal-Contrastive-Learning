from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BertTokenizer#
from augmentation.eda import eda
import torch
class TextAugmentation : 
    def __init__(self, config):
        super().__init__()
        self.max_length = config["max_text_len"]
        self.type_txt_augm = config["type_txt_augm"]
        self.tokenizer =  BertTokenizer.from_pretrained(config["tokenizer"])
        self.tokenizer_pegasus = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
        self.pegasus = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')        
    
    def augmentation(self,pl_module, batch):
        self.pegasus = self.pegasus.to(pl_module.device)
        txt_input  = []
        text_masks = []
        if self.type_txt_augm == "PEGASUS" : 
            batch_pegasus = self.tokenizer_pegasus(batch["text"], 
                                                     truncation=True, padding='longest', return_tensors="pt").to(pl_module.device)
            translated = self.pegasus.generate(**batch_pegasus).to(pl_module.device)
            augmented_text = self.tokenizer_pegasus.batch_decode(translated, skip_special_tokens=True)

        if self.type_txt_augm == "EDA" : 
            augmented_text = []
            for i,sentence in enumerate(batch["text"]) : 
                augmented_text.append(eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1))
 
        for i,sentence in enumerate(batch["text"]) : 
            print("Original sentence :::", sentence)
            print("Augmented sentence:::", augmented_text[i]) 
            print("\n")
        
        outputs = self.tokenizer(augmented_text, truncation=True, padding=True, max_length=self.max_length)
        batch["text"] = augmented_text
        batch["text_ids"]= torch.tensor(outputs["input_ids"]).to(pl_module.device)
        batch["text_masks"]= torch.tensor(outputs["attention_mask"]).to(pl_module.device)  
        return batch