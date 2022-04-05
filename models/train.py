import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from deblur_demo import DeblurDataset  # 读取数据集
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../train', help='path to Places dataset')
parser.add_argument('--dataroot_c', default='/Users/tl/Downloads/part_train_c/', help='path to CelebA dataset')
parser.add_argument('--testroot', default='/Users/tl/Downloads/part_train/test/', help='path to Places test set')
parser.add_argument('--testroot_c', default='/Users/tl/Downloads/part_train_c/test/', help='path to CelebA test set')
parser.add_argument('--checkpoint', default='../checkpoint/Places/', help='path to checkpoint for Places dataset')
parser.add_argument('--checkpoint_c', default='../checkpoint/CelebA', help='path to checkpoint for Places dataset')

parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--Epoch', type=int, default=2, help='number of epochs to train for')  # 25
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--L1_LAMBDA', default=100)

parser.add_argument('--gen', default='', help="path to generator (to continue training)")
parser.add_argument('--dis', default='', help="path to discriminator (to continue training)")

parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='')
opt = parser.parse_args()


def main():
    # initialize the discriminator, generator and their optimizers
    netD = Discriminator().to(opt.device)
    netG = Generator().to(opt.device)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # initialize loss objects
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = nn.L1Loss()

    # load previously trained model
    if opt.gen != '':
        netG.load_state_dict(torch.load(opt.gen, map_location=opt.device)['state_dict'])
        netD.load_state_dict(torch.load(opt.dis, map_location=opt.device)['state_dict'])
        optimizerG.load_state_dict(torch.load(opt.gen, map_location=opt.device)['optimizer'])
        optimizerD.load_state_dict(torch.load(opt.dis, map_location=opt.device)['optimizer'])
        start_epoch = netG.load_state_dict(torch.load(opt.gen)['epoch'])
    else:
        start_epoch = 0

    # Load the training set
    train_dataset = DeblurDataset(root_dir=opt.dataroot)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)

    # training loop
    for epoch in range(start_epoch, opt.Epoch):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)  # 只有1个False

        for idx, (x, y) in loop:
            blur, real = x.to(opt.device), y.to(opt.device)
            '''
            将模型中的参数的梯度设为0
            model.zero_grad()
            optimizer.zero_grad() 将模型的参数梯度初始化为0
            loss.backward()   反向传播计算梯度，当网络参量进行反馈时，梯度是累积计算而不是被替换，要对每个batch调用一次zero_grad()
            optimizer.step()  更新所有参数
            '''

            # Train Discriminator
            fake = netG(blur)  # generator生成的假图
            D_real = netD(blur, real)  # hd和ld的判别结果
            D_fake = netD(blur, fake.detach())  # ld和生成的
            D_real_loss = adversarial_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = adversarial_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = D_real_loss + D_fake_loss
            optimizerD.zero_grad()
            D_loss.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            D_fake = netD(blur, fake)
            G_D_loss = adversarial_loss(D_fake, torch.ones_like(D_fake))
            G_fake_loss = content_loss(fake, real) * opt.L1_LAMBDA

            G_loss = G_fake_loss + G_D_loss
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()

            loop.set_description(f'Epoch [{epoch + 1}/{opt.Epoch}][{idx}/{len(train_loader)}]')
            loop.set_postfix(gloss=G_loss.item(), dloss=D_loss.item())

            if idx % 100 == 0:
                save_image(real * 0.5 + 0.5, '../result/train/real/' + str(idx) + '-' + str(epoch) + '.png')
                save_image(blur * 0.5 + 0.5, '../result/train/blur/' + str(idx) + '-' + str(epoch) + '.png')
                save_image(fake * 0.5 + 0.5, '../result/train/deblur/' + str(idx) + '-' + str(epoch) + '.png')

        torch.save({'epoch': epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint + 'discriminator/dis-' + str(epoch) + '.pth')
        torch.save({'epoch': epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint + 'generator/gen-' + str(epoch) + '.pth')

        # do checkpointing
    torch.save({'epoch': opt.Epoch, 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict()}, opt.checkpoint + 'generator/generator.pth')
    torch.save({'epoch': opt.Epoch, 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict()}, opt.checkpoint + 'discriminator/discriminator.pth')


if __name__ == "__main__":
    main()
