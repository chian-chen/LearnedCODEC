import torch
from tqdm import tqdm
import numpy as np
import os

from utils.metric import ms_ssim

def inference(model, inference_loader, args, device):
    
    with torch.no_grad():
        model.eval()
        model.to(device)

        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch in tqdm(inference_loader, total=len(inference_loader), desc="Inference"):
            
            input_images, ref_image, ref_bpp, ref_psnr, ref_msssim = batch
            input_images, ref_image = input_images.to(device), ref_image.to(device)

            sumbpp += torch.mean(ref_bpp).detach().numpy()
            sumpsnr += torch.mean(ref_psnr).detach().numpy()
            summsssim += torch.mean(ref_msssim).detach().numpy()

            seqlen = input_images.shape[1]
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]
    
                clipped_recon_image, mse_loss, _, _, _, _, _, bpp = model(input_image, ref_image)

                sumbpp += torch.mean(bpp).cpu().detach().numpy()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
                summsssim += ms_ssim(clipped_recon_image.cpu().detach(), input_image.cpu().detach(), data_range=1.0, size_average=True).numpy()
                cnt += 1
                ref_image = clipped_recon_image

        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        print(f"UVGdataset")
        print(f"Average bpp: {sumbpp:.4f}, Average psnr: {sumpsnr:.4f}, Average msssim: {summsssim:.4f}")
        with open(os.path.join(args.output_weight, f'experiment.log'), "a") as f:
            f.write(f"Average bpp: {sumbpp:.4f}, Average psnr: {sumpsnr:.4f}, Average msssim: {summsssim:.4f}\n")
