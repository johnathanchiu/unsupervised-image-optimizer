from .transforms import *

# inplace partition of image
def partition_inplace(img, block_size=8, crop=True):
    if crop:
        _, x, y = img.shape
        x_crop, y_crop = x - x % block_size, y - y % block_size
        img = img[...,:x_crop,:y_crop]
    else:
        assert img.shape[-1] % block_size == 0
        assert img.shape[-2] % block_size == 0
    c, x, y = img.shape
    img_partitions = np.zeros((x//block_size, y//block_size, c, block_size, block_size), dtype=img.dtype)
    for i in range(0, x, block_size):
        for j in range(0, y, block_size):
            block = img[...,i:i+block_size,j:j+block_size]
            img_partitions[i//block_size,j//block_size,...] = block
    return img_partitions

# inplace undo partition
def reduce_inplace(img_partitions):
    n, xb, yb, c, b, _ = img_partitions.shape
    img_reduced = torch.zeros((n, c, xb * b, yb * b), dtype=img_partitions.dtype, device=img_partitions.device)
    x, y = xb * b, yb * b
    for i in range(0, x, b):
        for j in range(0, y, b):
            img_reduced[...,i:i+b,j:j+b] = img_partitions[:,i//b,j//b,...]
    return img_reduced

# partitions block in 1d list of 8 by 8 in order
def partition(img, block_size=8, crop=True):
    if crop:
        _, x, y = img.shape
        x_crop, y_crop = x - x % block_size, y - y % block_size
        img = img[...,:x_crop,:y_crop]
    b = block_size
    _, x, y = img.shape
    return [img[...,i:i+b,j:j+b] for i in range(0, x, b) for j in range(0, y, b)]

# zig-zag encoder
def zz_encode(block):
    i = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 
         3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 
         7, 7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7]
    j = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 
         3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 
         2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7]
    idx = list(range(64))
    n, c, d, _ = block.shape
    encoded = torch.zeros((n, c, d * d), dtype=block.dtype, device=block.device)
    encoded[...,idx] = block[...,i,j]
    return encoded

# zig-zag decoder
def zz_decode(encoded):
    i = [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 
         3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 
         7, 7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7]
    j = [0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 
         3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 
         2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7]
    idx = list(range(64))
    n, c, d = encoded.shape
    d = int(d ** 0.5)
    decoded = torch.zeros((n, c, d, d), dtype=encoded.dtype, device=encoded.device)
    decoded[...,i,j] = encoded[...,idx]
    return decoded



def reconstruct_img(x, table):
    # batch, channels, x, y
    if table.shape[1] == 2:
        table = torch.cat((table, table[:,-1:,:,:]), axis=1)
    table = torch.unsqueeze(torch.unsqueeze(table, dim=1), dim=1)
    x_quantize = x / table
    x_reconstruct = idct2(torch.round(x_quantize) * table) + 128
    x_reconstruct[x_reconstruct < 0] = 0.0
    x_reconstruct[x_reconstruct > 255] = 255.0
    x_reconstruct = reduce_inplace(x_reconstruct)
    zz_quantized = zz_batch_encode(x_quantize)
    return zz_quantized, x_reconstruct


def zz_batch_encode(tensor_input):
    n, xb, yb, c, b, _ = tensor_input.shape
    encoded = torch.zeros((n, xb * yb, c, b * b), dtype=tensor_input.dtype, device=tensor_input.device)
    t = 0
    for i in range(xb):
        for j in range(yb):
            encoded[:,t,...] = zz_encode(tensor_input[:,i,j,...])
            t += 1
    return encoded