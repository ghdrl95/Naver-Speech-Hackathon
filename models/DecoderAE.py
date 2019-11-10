

import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModel(nn.Module):
    def __init__(self,input_size=820, output_size = 820, n_layers=2, PAD_token = 0):
        input_list = []
        before_size = input_size
        for i in range(2,n_layers+3):
            input_list.append(nn.Linear(before_size,output_size//i))
            before_size = output_size//i
        for _ in range(n_layers+1):
            i -= 1
            input_list.append(nn.Linear(before_size, output_size // i))
            before_size = output_size // i

        self.autoencoder = nn.Sequential( *input_list )
    def forward(self, x):
        #미니배치 사이즈 추출
        minibatch_size = x.size(0)
        #reshape (미니배치*문장, 사전글자수)
        x = x.view(x.size(0) *x.size(1), x.size(-1))
        #오토인코더 연산
        y = self.autoencoder(x)
        #다시 복구(미니배치, 문장, 사전글자수)
        y = y.view(minibatch_size, -1, y.size(-1))
        #출력
        return y
