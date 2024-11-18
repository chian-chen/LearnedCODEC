import os 
import torch
import imageio
import numpy as np
from torch.utils.data import Dataset

from utils.metric import CalcuPSNR, ms_ssim
from utils.transform import random_flip, random_crop_and_pad_image_and_labels

out_channel_M = 96
out_channel_N = 64
out_channel_mv = 128

# This dataset only use for testing, pre-processing: Use x.265 to generate I frame, store bpp
class UVGDataset(Dataset):
    def __init__(self, root="data/UVG/images/", filelist="data/UVG/video_list.txt", refdir="H265L20", testfull = False):
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []

        AllIbpp = self.getbpp(refdir)
        ii = 0

        for ii, folder in enumerate(folders):

            seq = folder.rstrip()
            seqIbpp = AllIbpp[ii]

            imlist = [im for im in os.listdir(os.path.join(root, seq)) if im.endswith('.png')]
            framerange = len(imlist) // 12 if testfull else 1

            for i in range(framerange):
                refpath = os.path.join(root, seq, refdir, f'im{(i * 12 + 1):04}.png')
                inputpath = [os.path.join(root, seq, f'im{(i * 12 + 1 + j):03}.png') for j in range(12)]

                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)

    def getbpp(self, ref_i_dir):    # Write it as a json file later
        Ibpp = None
        if ref_i_dir == 'H265L20':
            print('use H265L20')
            Ibpp = [1.292902, 0.675868, 0.940059, 0.677053, 0.75437, 0.864065, 0.692403] 
            print(Ibpp)
        elif ref_i_dir == 'H265L23':
            print('use H265L23')
            Ibpp = [0.724385, 0.471212, 0.567216, 0.360455, 0.550235, 0.580513, 0.500595]
        elif ref_i_dir == 'H265L26':
            print('use H265L26')
            Ibpp = [0.341165, 0.331902, 0.368315, 0.205473, 0.406156, 0.403127, 0.362514]
        elif ref_i_dir == 'H265L29':
            print('use H265L29')
            Ibpp = [0.141959, 0.230965, 0.259548, 0.135205, 0.30208, 0.289264, 0.262448]
        else:
            print('cannot find ref : ', ref_i_dir)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp

    def __len__(self):
        return len(self.ref)
    
    def __getitem__(self, index):

        ref_image = imageio.v2.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])

        input_images = []
        refpsnr = None
        refmsssim = None

        for filename in self.input[index]:
            input_image = (imageio.v2.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0

            if refpsnr is None:
                refpsnr = CalcuPSNR(torch.from_numpy(input_image), torch.from_numpy(ref_image)).item()
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = torch.from_numpy(np.array(input_images))
        ref_image = torch.from_numpy(ref_image)

        # ref_image: the image after compression by H.265 still image, self.refbpp, refpsnr, refmssim: I frame data, input_images: 12 lossless png frames
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim
    

class DataSet(Dataset):
    def __init__(self, root="data/Vimeo/vimeo_septuplet/sequences", filelist="data/Vimeo/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(root, filelist)
        self.im_height = im_height
        self.im_width = im_width
        
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])

    def get_vimeo(self, root="data/vimeo_septuplet/sequences", filelist="data/vimeo_septuplet/test.txt"):
        with open(filelist) as f:
            data = f.readlines()
        
        input_paths = []
        ref_paths = []

        for line in data:
            path = os.path.join(root, line.rstrip())
            input_paths.append(path)

            ref_number = int(path[-5:-4]) - 2   # 2 or 1 may be the same?
            ref_path = f'{path[:-5]}{ref_number}.png'
            ref_paths.append(ref_path)

        return input_paths, ref_paths

    def __len__(self):
        return len(self.image_input_list)
    
    def __getitem__(self, index):
        input_image = imageio.v2.imread(self.image_input_list[index]).astype(np.float32) / 255.0
        ref_image = imageio.v2.imread(self.image_ref_list[index]).astype(np.float32) / 255.0

        input_image = torch.from_numpy(input_image).permute(2, 0, 1)
        ref_image = torch.from_numpy(ref_image).permute(2, 0, 1)

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)

        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5)
        quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)

        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv


if __name__ == "__main__":

    dataset = UVGDataset(root='./data/UVG/images', 
                         filelist='./data/UVG/video_list.txt', 
                         refdir='H265L20', testfull=False)
    input_images, ref_image, refbpp, refpsnr, refmsssim = dataset.__getitem__(0)

    print(input_images.shape)
    print(ref_image.shape)
    print(refbpp)
    print(refpsnr)
    print(refmsssim)
    
    train_dataset = DataSet(root = 'data/Vimeo/vimeo_septuplet/sequences',
                            filelist='data/Vimeo/vimeo_septuplet/test.txt')
    
    input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv = train_dataset.__getitem__(0)

    print(input_image.shape)
    print(ref_image.shape)
    print(quant_noise_feature.shape)
    print(quant_noise_mv.shape)
    print(quant_noise_z.shape)
    print(train_dataset.__len__())