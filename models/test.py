import argparse

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from deblur_demo import DeblurDataset
from generator import Generator
import torch
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser()
parser.add_argument('--testroot', default='/Users/tl/Downloads/part_train/test/', help='path to Places test set')
parser.add_argument('--testroot_c', default='/Users/tl/Downloads/part_train_c/test/', help='path to CelebA test set')
parser.add_argument('--output', default='../test_result/', help='path to test_result folder')

parser.add_argument('--gen', default='../checkpoint/generator_epoch_90_240Places2.pth', help="path to generator")
parser.add_argument('--batchSize', default=2, help="path to generator")
parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='')
opt = parser.parse_args()

# initialize generator
gen = Generator().to(opt.device)
# Load the generator
gen.load_state_dict(torch.load(opt.gen, map_location=opt.device)['state_dict'])
gen.eval()


test_dataset = DeblurDataset(root_dir=opt.testroot)
print("Number of Images: ", len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize)

index = 0   # the no. of batch
psnr_list = []
ssim_list = []

for x, y in iter(test_loader):
    input_image, target_image = x.to(opt.device), y.to(opt.device)
    with torch.no_grad():
        fake = gen(input_image)
        fake = fake * 0.5 + 0.5  # remove normalization
        save_image(fake, opt.output + f"generated_" + str(index)+".png")
        save_image(input_image * 0.5 + 0.5, opt.output + f"input_" + str(index)+".png")

        output = Image.open(opt.output + f"generated_" + str(index)+".png").convert('RGB')
        input = Image.open(opt.output  + f"input_" + str(index)+".png").convert('RGB')
        img1 = np.array(input)
        img2 = np.array(output)  # (256, 256, 3)

        psnr_list.append(psnr(img1, img2))
        ssim_list.append(sk_cpt_ssim(img1, img2, multichannel=True))
    gen.train()
    index = index + 1

print(psnr_list)
print(sum(psnr_list)/len(psnr_list))
print(ssim_list)
print(sum(ssim_list)/len(ssim_list))
