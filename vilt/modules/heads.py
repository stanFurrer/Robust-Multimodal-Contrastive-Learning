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


class MOCOHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.img_model  = nn.Sequential(OrderedDict([
            ('linear1_img',nn.Linear(self.input_dim, self.hidden_dim)),
            ('norm_img',   nn.LayerNorm(self.hidden_dim)),
            ('relu_img',   nn.ReLU()),
            ('linear2_img',nn.Linear(self.hidden_dim, self.output_dim, bias=False))
        ]))

        self.txt_model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.input_dim, self.hidden_dim)),
            ('norm',    nn.LayerNorm(self.hidden_dim)),
            ('relu',    nn.ReLU()),
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
