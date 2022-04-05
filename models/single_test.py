"""
demo for repair blurred image
input: one blurred image
output: clear image
"""
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from generator import Generator
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser()
parser.add_argument('--testroot', default='/Users/tl/Downloads/part_train/test/', help='path to Places test set')
parser.add_argument('--testroot_c', default='/Users/tl/Downloads/part_train_c/test/', help='path to CelebA test set')

parser.add_argument('--gen', default='../checkpoint/generator_epoch_90_240Places2.pth', help="path to generator")

parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='')
opt = parser.parse_args()



def single_test(img_path):
    # initialize generator and load pre-trained weights
    gen = Generator().to(opt.device)
    gen.load_state_dict(torch.load(opt.gen, map_location=opt.device)['state_dict'])

    # load the image
    input_image = Image.open(img_path).convert('RGB')
    x = input_image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    blur = transform(input_image)
    input_image = torch.unsqueeze(blur, 0)
    input_image = input_image.to(opt.device)
    gen.eval()

    with torch.no_grad():
        generated_image = gen(input_image)

        # Undo Normalization
        generated_image = generated_image * 0.5 + 0.5

        # Save the output image
        out_path = img_path.replace('.jpg', '-deblurred.jpg')
        save_image(generated_image, out_path)

    output = Image.open(out_path).convert('RGB')
    input = Image.open(img_path).convert('RGB')
    img1 = np.array(input)
    img2 = np.array(output)   # (256, 256, 3)

    print(psnr(img1, img2))  # 52.26561389555605 numpy.ndarray
    print(sk_cpt_ssim(img1, img2, multichannel=True))


if __name__ == "__main__":
    path = '../demo/test.jpg'
    single_test(path)

