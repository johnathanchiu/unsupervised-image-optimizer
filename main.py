from image.loader import *

from model.loss import *
from model.optimizer import *
from model.small import *

from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torch

import numpy as np
import math

import argparse


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


SAMPLES = 256
CROP = (384, 384)
EPOCHS = 70
MAX_Q_VALUE = 170

def main(img_path, save_path):

    dataset = ImageCompressionDataset(img_path, device=device, crop=CROP, samples=SAMPLES)
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)

    model = QTableOptimizer(MAX_Q_VALUE, input_channels=3, n_qtables=2, samples=SAMPLES).double().to(device)

    blur_fn = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 2.0))
    criterion = QuantizationLoss(rate_weight=1, distortion_weight=1).to(device)
    rate_criterion = RateLoss(band_scales=(1e-4, 1e2, 1e3), device=device).to(device)
    distortion_criterion = DistortionLoss(win_size=3, 
                                        blur_kernel=21, 
                                        ssim_scale=1e6, 
                                        alpha=0.85,
                                        n_channels=3,
                                        blur_fn=blur_fn).to(device)
    annealer = AnnealingOptimizer(1, 1, t=1e5, beta=0.65)
    optimizer = optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.999), weight_decay=0, amsgrad=True)

    best_in_bin = [(float('inf'), None)] * 10

    best_loss = float('inf')
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        print('EPOCH: %d\n%s' % (epoch + 1, '-'*40))

        frequency_data, freq_data_partition, spatial_data, image_path = next(iter(dataset))
        frequency_data = frequency_data.to(device)
        freq_data_partition = freq_data_partition.to(device)
        spatial_data = spatial_data.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        frequency_data_input = normalize(frequency_data)
        qtables = model(frequency_data_input)
        # print(qtables)

        zz_quantized, reconstruction = reconstruct_img(freq_data_partition, qtables)
        rate_loss = rate_criterion(zz_quantized)
        ssim_value, distortion_loss = distortion_criterion(reconstruction, spatial_data)
        loss = criterion(rate_loss, distortion_loss)
        measure_loss = rate_loss + distortion_loss

        if epoch == 0:
            annealer.set_original_entropy(rate_loss)

        print('ssim:', ssim_value.item())

        loss.backward()
        optimizer.step()
        
        rate_update, distortion_update = annealer.forward(ssim_value, rate_loss, epoch)
        criterion.rate_weight = rate_update
        criterion.distortion_weight = distortion_update

        print('rate weight:', criterion.rate_weight)
        print('distortion weight:', criterion.distortion_weight)
        
        # store top images in bins
        if math.floor(ssim_value * 10) == 9:
            bin_index = math.floor(ssim_value * 100) % 10
            if rate_loss / criterion.rate_weight < best_in_bin[bin_index][0]:
                best_in_bin[bin_index] = (rate_loss / criterion.rate_weight, qtables)
            
        # print statistics
        print('epoch %3d loss: %.3f' % (epoch + 1, abs(measure_loss)))
        print(best_in_bin)

    print('Finished Training')

    for i, item in enumerate(best_in_bin):
        if item is not None:
            _, qtables = item
            # save top images
            save_test_image_color(qtables, image_path[0], save_path, i)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='path to input image')
    parser.add_argument('-o', '--output', required=True, help='folder output for images')
    args = parser.parse_args()

    main(args.input, args.output)


