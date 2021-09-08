import random
import torch
import io
import pyarrow as pa
import os
import copy
from PIL import Image
from vilt.config import ex
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as T
import sys

# To save the image 
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# To save the image 
def show(imgs_clean,imgs_augm):
    """Open an tensor and save it in PNG"""
    save_exemple_augm = "/itet-stor/sfurrer/net_scratch/UNITER/ViLT/attacks_analysis_vilt/AUGM/exemple"
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  

    imgs_clean = imgs_clean.to('cpu')
    imgs_augm= imgs_augm.to('cpu') 
    for i,(img_clean,img_augm) in enumerate(zip(imgs_clean,imgs_augm)):     
        img_clean  = unorm(img_clean)    
        img_augm = unorm(img_augm)     
        img_clean = T.ToPILImage()(img_clean)
        img_augm = T.ToPILImage()(img_augm)   
        img_clean.save(os.path.join(save_exemple_augm,"img_clean{}.png".format(i)),"PNG",dpi=(1000, 1000))
        img_augm.save(os.path.join(save_exemple_augm,"img_augm{}.png".format(i)),"PNG",dpi=(1000, 1000))            
    sys.exit("Stop")

class MinMaxResize:
    def __init__(self, shorter=800, longer=1333):
        self.min = shorter
        self.max = longer

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
class Transform:
    def __init__(self,size=800):
        longer = int((1333 / 800) * size)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.2),
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            # This is simple maximum entropy normalization performed in ViLT paper. 
            # It is different than original BarlowTwins
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])  
        ])

class ImageAugmentation : 
    """
    load_path : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
    """
    def __init__(self, config):
        super().__init__()
        self.data_dir = config["data_root"]
        self.names = ['coco_caption_karpathy_train', 'coco_caption_karpathy_restval']#["coco_caption_karpathy_train"]#coco_caption_karpathy_train
        self.text_column_name ="caption"
        self.image_size = config["image_size"]
        remove_duplicate = True
        self.transforms = Transform(size=self.image_size)
        
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
        path_save = "/itet-stor/sfurrer/net_scratch/UNITER/ViLT/attacks_analysis_vilt/AUGM/exemple"
        image.save(os.path.join(path_save,"img{}.png".format(index)),"PNG")
        ### 
        image_tensor = [self.transforms.transform(image)]
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
        show(batch["image"][0],new_image[0])
        return new_image     