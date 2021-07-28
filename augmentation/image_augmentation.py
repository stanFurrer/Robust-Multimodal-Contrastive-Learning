import random
import torch
import io
import pyarrow as pa
import os
import copy
from PIL import Image
from vilt.transforms import keys_to_transforms
from vilt.config import ex

class ImageAugmentation : 
    """
    load_path : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
    """
    def __init__(self, config):
        super().__init__()
        self.data_dir = config["data_root"]
        self.names = ["coco_caption_karpathy_train"]
        self.text_column_name ="caption"
        self.image_size = config["image_size"]
        remove_duplicate = True
        self.transforms = keys_to_transforms(["pixelbert_randaug"], size=self.image_size)
        
        if len(self.names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{self.data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in self.names
                if os.path.isfile(f"{self.data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(self.names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if self.text_column_name != "":
                self.all_texts = self.table[self.text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if self.text_column_name != "":
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(table)):
                self.index_mapper[i] = (i, None)
    
    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        ### Save Original
        #path_save = "/itet-stor/sfurrer/net_scratch/UNITER/ViLT/attacks_analysis/image_augmentation"
        #image.save(os.path.join(path_save,"image{}.jpg".format(index)),"JPEG")
        ### 
        image_tensor = [tr(image) for tr in self.transforms]
        return image_tensor       

    def collate(self, images):
        batch_size = len(images)
        # Creat the tensor matrix with max_height and max_width
        img_sizes = [ii.shape for i in images if i is not None for ii in i]
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

        new_images = [torch.zeros(batch_size, 3, max_height, max_width)]
        
        for bi in range(batch_size):
            orig_batch = images[bi]
            if orig_batch is None:
                new_images[0][bi] = None
            else:
                orig = images[bi][0]
                new_images[0][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        return new_images        

    def augmentation(self, batch):
        images_tensor = []
        for index in batch["img_index"] :
            images_tensor.append(self.get_image(index))
        new_image = self.collate(images_tensor) 
        return new_image     