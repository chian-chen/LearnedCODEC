import numpy as np
import torch
import torchac
from concurrent.futures import ThreadPoolExecutor

from net import VideoCompressor
from enc_dec import enc, dcd

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0


model = VideoCompressor(training=False).to('cuda')
load_model(model=model, f='./weights/2048.model')
mxrange = 150
test_num = 1

x = torch.from_numpy(np.load(f"./npys/npy_x/x_{test_num}.npy")).to('cuda')
sigma = torch.from_numpy(np.load(f"./npys/npy_sigma/sigma_{test_num}.npy")).to('cuda')
mv = torch.from_numpy(np.load(f"./npys/npy_mv/mv_{test_num}.npy")).to('cuda')
z = torch.from_numpy(np.load(f"./npys/npy_z/z_{test_num}.npy")).to('cuda')

print(f'residual data: {x.shape}')
# print(sigma.shape)
print(f'mv data: {mv.shape}')
print(f'z data: {z.shape}')

bpp = 0

# Residual 

mu = torch.zeros_like(sigma)
sigma = sigma.clamp(1e-5, 1e10)
gaussian = torch.distributions.laplace.Laplace(mu, sigma)

cdfs = []

x = x + mxrange
n, c, h, w = x.shape

for i in range(-mxrange, mxrange):
    cdfs.append(gaussian.cdf(torch.tensor(i - 0.5)).view(n,c,h,w,1))
cdfs = torch.cat(cdfs, 4).cpu().detach()

byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)
real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8]))).float().cuda()

sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
re_x = sym_out - mxrange

print(real_bits)
bpp += real_bits

# MV

cdfs = []
mv = mv + mxrange
n,c,h,w = mv.shape
for i in range(-mxrange, mxrange):
    cdfs.append(model.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
cdfs = torch.cat(cdfs, 4).cpu().detach()
byte_stream = torchac.encode_float_cdf(cdfs, mv.cpu().detach().to(torch.int16), check_input_bounds=True)

real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
re_mv = sym_out - mxrange

print(real_bits)
bpp += real_bits

# z

cdfs = []
z = z + mxrange
n,c,h,w = z.shape
for i in range(-mxrange, mxrange):
    cdfs.append(model.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
cdfs = torch.cat(cdfs, 4).cpu().detach()
byte_stream = torchac.encode_float_cdf(cdfs, z.cpu().detach().to(torch.int16), check_input_bounds=True)

real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

z = sym_out - mxrange

print(real_bits)
bpp += real_bits

print(bpp / 1920 / 1024)