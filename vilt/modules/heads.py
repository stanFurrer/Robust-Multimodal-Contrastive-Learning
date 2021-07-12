import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BarlowTwinsHead(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim):
        super().__init__()
        layer_dim = [input_dim] + inner_dim + [output_dim]
        self.projector = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(layer_dim[0], layer_dim[1], bias=False)),
            ('BatchNorm', nn.BatchNorm1d(layer_dim[1])),
            ('ReLu', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(layer_dim[1], layer_dim[2], bias=False)),
            ('BatchNorm', nn.BatchNorm1d(layer_dim[2])),
            ('ReLu', nn.ReLU(inplace=True)),
            ('linear3', nn.Linear(layer_dim[2], layer_dim[3], bias=False)),
        ]))
        '''
        self.txt_projector = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(layer_dim[0], layer_dim[1], bias=False)),
            ('BatchNorm', nn.BatchNorm1d(layer_dim[1])),
            ('ReLu', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(layer_dim[1], layer_dim[2], bias=False)),
            ('BatchNorm', nn.BatchNorm1d(layer_dim[2])),
            ('ReLu', nn.ReLU(inplace=True)),
            ('linear3', nn.Linear(layer_dim[2], layer_dim[3], bias=False)),
        ]))
        '''
        self.norm = nn.BatchNorm1d(layer_dim[-1], affine=False)
    
    def forward(self, y1, y2):
        y1 = y1[:, 0]
        y2 = y2[:, 0]
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        return self.norm(z1), self.norm(z2)
        

class MOCOHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.img_model  = nn.Sequential(OrderedDict([
            ('linear1_img', nn.Linear(self.input_dim, self.hidden_dim)),
            ('LayerNorm',   nn.LayerNorm(self.hidden_dim)),
            ('LayerNorm',   nn.ReLU()),
            ('linear2_img', nn.Linear(self.hidden_dim, self.output_dim, bias=False))
        ]))

        self.txt_model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.input_dim, self.hidden_dim)),
            ('LayerNorm', nn.LayerNorm(self.hidden_dim)),
            ('relu', nn.ReLU()),
            ('linear2', nn.Linear(self.hidden_dim, self.output_dim, bias=False))
        ]))

    def forward(self, img, txt):
        first_image_tensor = img[:, 0]
        first_text_tensor = txt[:, 0]
        img = self.img_model(first_image_tensor)
        txt = self.txt_model(first_text_tensor)
        return img, txt

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
