import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import  PCDNet
from utils import AverageMeter
import copy
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from loss import LossFunction

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='PCDNet')
    parser.add_argument('--train_dir', type=str, default='data/real/train')
    parser.add_argument('--val_dir', type=str, default='data/real/val')
    parser.add_argument('--outputs_dir', type=str, default='weights/test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epoch_save_num', type=int, default=1)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'PCDNet':
        model = PCDNet()

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print(f"=> loading checkpoint '{opt.resume}'")
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(f"=> no checkpoint found at '{opt.resume}'")

    train_noisy_dir = os.path.join(opt.train_dir, 'noisy')
    train_clean_dir = os.path.join(opt.train_dir, 'clean')
    val_noisy_dir = os.path.join(opt.val_dir, 'noisy')
    val_clean_dir = os.path.join(opt.val_dir, 'clean')

    train_dataset = PairedImageDataset(train_noisy_dir, train_clean_dir)
    val_dataset = PairedEvalDataset(val_noisy_dir, val_clean_dir)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.threads, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    criterion = LossFunction(
        lambda_l1=0.6,
        lambda_edge=0.2,
        lambda_percep=0.2
    ).to(device)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    results = {'loss': [], 'psnr': [], 'ssim': []}

    for epoch in range(opt.start_epoch, opt.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=len(train_loader.dataset) - len(train_loader.dataset) % opt.batch_size) as _tqdm:
            _tqdm.set_description(f"Epoch [{epoch + 1}/{opt.num_epochs}]")
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.update(loss.item(), len(inputs))
                _tqdm.set_postfix(loss="{:.6f}".format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        model.eval()
        val_psnr = AverageMeter()
        val_ssim_total = 0
        val_batch_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                pred_np = preds.squeeze().mul(255.0).clamp(0, 255).byte().cpu().numpy()
                label_np = labels.squeeze().mul(255.0).clamp(0, 255).byte().cpu().numpy()

                pred_np = np.clip(pred_np, 0, 255)
                label_np = np.clip(label_np, 0, 255)

                psnr = compare_psnr(label_np, pred_np, data_range=255.0)
                ssim = compare_ssim(label_np, pred_np, data_range=255.0)

                val_psnr.update(psnr)
                val_ssim_total += ssim
                val_batch_total += 1

        avg_ssim = val_ssim_total / val_batch_total
        print(f'===> Val PSNR: {val_psnr.avg:.2f}, SSIM: {avg_ssim:.4f}')

        if val_psnr.avg > best_psnr:
            best_psnr = val_psnr.avg
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            print(f"=> New best PSNR: {best_psnr:.2f} at epoch {epoch}")

        results['loss'].append(epoch_losses.avg)
        results['psnr'].append(val_psnr.avg)
        results['ssim'].append(avg_ssim)

        if (epoch + 1) % opt.epoch_save_num == 0:
            df = pd.DataFrame({
                'Loss': results['loss'],
                'PSNR': results['psnr'],
                'SSIM': results['ssim']
            }, index=range(opt.start_epoch, epoch + 1))
            df.to_csv(os.path.join(opt.outputs_dir, f'results_{opt.arch}.csv'), index_label='Epoch')

    # print(f"=> Best Epoch: {best_epoch}, PSNR: {best_psnr:.2f}")
    # torch.save(best_weights, os.path.join(opt.outputs_dir, f'best_{opt.arch}.pth'))
    print(f"=> Training complete. Best Epoch: {best_epoch}, PSNR: {best_psnr:.2f}")
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_weights,
        'psnr': best_psnr,
        'ssim': results['ssim'][best_epoch]
    }, os.path.join(opt.outputs_dir, f'best_{opt.arch}.pth'))