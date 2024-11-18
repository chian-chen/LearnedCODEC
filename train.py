import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

torch.autograd.set_detect_anomaly(True)

def train(model, train_loader, args, device):
    
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * 2, eta_min=0)

    print_step = 5000

    for epoch in range(args.epochs):
        
        print(f"Epoch: {epoch}")

        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = batch

            input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = \
                input_image.to(device), ref_image.to(device), quant_noise_feature.to(device), quant_noise_z.to(device), quant_noise_mv.to(device)
            
            _, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = model(input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv)
            
            mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
                torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv), torch.mean(bpp)
            
            distribution_loss = bpp
            distortion = mse_loss + 0.1 * (warploss + interloss)
            rd_loss = 2048 * distortion + distribution_loss

            optimizer.zero_grad()
            rd_loss.backward()

            clip_gradient(optimizer, 0.5)

            optimizer.step()
            scheduler.step()

            if batch_idx % print_step == 0:
                if mse_loss > 0:
                    psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
                else:
                    psnr = 100
                if warploss > 0:
                    warppsnr = 10 * (torch.log(1 * 1 / warploss) / np.log(10)).cpu().detach().numpy()
                else:
                    warppsnr = 100
                if interloss > 0:
                    interpsnr = 10 * (torch.log(1 * 1 / interloss) / np.log(10)).cpu().detach().numpy()
                else:
                    interpsnr = 100

                print(f'Train Epoch: {epoch}, psnr:{psnr}, warppsnr: {warppsnr}, interpsnr: {interpsnr}')

                torch.save(model.state_dict(), os.path.join(args.output_weight, f'{batch_idx}.ckpt'))
                with open(os.path.join(args.output_weight, f'experiment.log'), "a") as f:
                    f.write(f'Train Epoch: {epoch}, Batch_index: {batch_idx}, psnr:{psnr}, warppsnr: {warppsnr}, interpsnr: {interpsnr}\n')


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)