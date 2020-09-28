"""
@author: Shana
@file: Models.py
@time: 9/13/20 3:23 PM
"""

import torch
import torch.nn as nn

MAX_LENGTH = 10

torch.manual_seed(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DisEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch=True, drop=0):
        super(DisEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch,
                           dropout=drop)

    def forward(self, x):
        out, (h_n, c_n) = self.rnn(x)
        # Batch first
        # [batch_size,num_layers,hidden_size]
        dynamic = out
        static = c_n[-1, :, :]
        return dynamic, static


class DisDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch=True, drop=0):
        # input_size:h1+h2
        super(DisDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch,
                           dropout=drop)

    def forward(self, x):
        out = self.rnn(x)
        # Batch first
        # [batch_size,num_layers,hidden_size]
        return out


class DyEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch=True, drop=0):
        super(DyEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.dropout(x)
        out, h_n = self.rnn(x)
        # Batch first
        # [batch size, num_layers, hidden_size]
        return h_n


class DyDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch=True, drop=0):
        super(DyDecoder, self).__init__()
        self.input_size = input_size#context
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=input_size + output_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=batch )
        self.dropout = nn.Dropout(drop)
        self.out = nn.Linear( input_size +hidden_size + output_size, output_size)
        self.mask = nn.Linear(output_size,2)

    def forward(self, x, hidden, context):
        # context (output of encoder)  [batchsize,1,]
        # hidden [batchsize,1,h2]
        # x [batchsize,out_put_size] yt-1
        x = x.unsqueeze(1)
        input = torch.cat((x, context), dim=2)
        output, hid = self.rnn(input, hidden)
        output = torch.cat((x.squeeze(1), hid.squeeze(1), context.squeeze(1)), dim=1)
        pre = self.out(output)
        pro = self.mask(pre)
        pro = nn.functional.softmax(pro,dim=1)
        return pre, hidden, pro


class DyModule(nn.Module):
    def __init__(self, dyencoder, dydecoder, max_len):
        super(DyModule, self).__init__()
        self.encoder = dyencoder
        self.decoder = dydecoder
        self.max_len = max_len

    def forward(self, x):
        # context = dynamic feature [batch,1,h2]
        # x [batchsize,T,h1]
        batch_size = x.shape[0]
        context = self.encoder(x)
        hidden = context  # first hidden will be dynamic feature
        # 初始化yt 存疑
        inputs = torch.zeros(batch_size, self.decoder.output_size).to(torch.device('cuda'))
        # print("yt-1,output_size: "+str(input.shape))
        # print("context, input_size: "+ str(context.shape))
        outputs = []
        masks = []
        for t in range(0, self.max_len):
            output, hidden, ma = self.decoder(inputs, hidden, context)
            outputs.append(output)
            masks.append(ma)
            inputs = output
        # output [batchsize,h1]
        result = torch.zeros(batch_size, 
                             self.max_len, 
                             self.decoder.output_size, 
                             device=torch.device('cuda'))
        mask = torch.zeros(batch_size, self.max_len, 2)
        for i in range(0, self.max_len):
            for j in range(0, batch_size):
                result[j][i] = outputs[i][j]
                mask[j][i] = masks[i][j]


        return result, mask, context
