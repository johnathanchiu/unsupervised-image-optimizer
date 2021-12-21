import torch.nn as nn
import torch


# MODEL

def zz_encode_model(block):
    zz = [0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42, 3,  8,  
          12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 
          39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 
          59, 61, 35, 36, 48, 49, 57, 58, 62, 63]
    n, s, c, d, _ = block.shape
    encoded = torch.zeros((n, s, c, d * d), dtype=block.dtype, device=block.device)
    c = 0
    for i in range(d):
        for j in range(d):
            encoded[...,zz[c]] = block[...,i,j]
            c += 1
    return encoded


def zz_decode_model(encoded):
    zz = [0,  1,  5,  6,  14, 15, 27, 28, 2,  4,  7,  13, 16, 26, 29, 42, 3,  8,  
          12, 17, 25, 30, 41, 43, 9,  11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 
          39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 
          59, 61, 35, 36, 48, 49, 57, 58, 62, 63]
    n, s, c, d = encoded.shape
    d = int(d ** 0.5)
    decoded = torch.zeros((n, s, c, d, d), dtype=encoded.dtype, device=encoded.device)
    c = 0
    for i in range(d):
        for j in range(d):
            decoded[...,i,j] = encoded[...,zz[c]]
            c += 1
    return decoded

class QTableOptimizer(nn.Module):
    def __init__(self, max_q, input_channels=1, n_qtables=1, samples=32):
        super(QTableOptimizer, self).__init__()
        self.output_activation = nn.Sigmoid()
        self.input_activation = nn.Tanh()
        self.max_q = max_q
        self.qtables_out = n_qtables
        self.sample_learning = nn.Sequential(
            nn.Conv2d(samples, int(samples//2), kernel_size=1, stride=(1, 2)),
            self.input_activation,
            nn.BatchNorm2d(int(samples//2)),
            # nn.Dropout(p=0.3),
            nn.Conv2d(int(samples//2), 1, kernel_size=1, stride=(1, 2)),
            self.output_activation,
            nn.BatchNorm2d(1),
            # nn.Dropout(p=0.3),
        )
        self.channel_embedding = nn.Sequential(
            nn.Conv2d(input_channels, n_qtables, kernel_size=1, stride=1),
            # nn.ReLU(),
            self.output_activation,
            nn.BatchNorm2d(n_qtables),
            # nn.Dropout(p=0.3),
            # nn.Conv2d(input_channels * 4, n_qtables, kernel_size=1, stride=1),
            # self.output_activation,
            # nn.BatchNorm2d(n_qtables),
            # # nn.Dropout(p=0.3),
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(64, 256),
            self.output_activation,
            nn.BatchNorm2d(samples), 
        )
        
    def forward(self, x):
        # embed input over the sample
        x = zz_encode_model(x) # (b, s, c, p: 64)
        x = self.embedding_layer(x) # (b, s, c, p: 256)
        x = self.sample_learning(x) # (b, s: 1, c, p: 64)
        x = zz_decode_model(x) # (b, c, x: 8, y: 8)
        x = torch.squeeze(x, dim=1) # (b, c, x: 8, y: 8)
        if self.qtables_out > 1:
            x = self.channel_embedding(x) # (b, c: n_tables, x: 8, y: 8)
        y = self.output_activation(x) * self.max_q
        return y