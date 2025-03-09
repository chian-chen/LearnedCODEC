import torch
import torch.nn as nn
import torchac
import numpy as np
import math

from subnet.SpyNet import ME_Spynet
from subnet.mvEncoder import Analysis_mv_net
from subnet.mvDecoder import Synthesis_mv_net
from subnet.WarpNet import Warp_net
from subnet.resEncoder import Analysis_net
from subnet.resDecoder import Synthesis_net
from subnet.respriorEncoder import Analysis_prior_net
from subnet.respriorDecoder import Synthesis_prior_net
from subnet.BitEstimator import BitEstimator

from subnet.utils.spynet import flow_warp
from subnet.utils.dimension import out_channel_N, out_channel_mv



class VideoCompressor(nn.Module):
    def __init__(self, training=True):
        super(VideoCompressor, self).__init__()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)

        self.mxrange = 150
        self.calrealbits = not training
        self.training = training

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None, cnt=0):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
        quant_mv_upsample = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = input_image - prediction

        feature = self.resEncoder(input_residual)
        batch_size = feature.size()[0]

        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)


        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))
        

        # bit per pixel

        def feature_probs_based_sigma(feature, sigma):
            
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(torch.tensor(i - 0.5)).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)
                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits


            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))

            np.save(f"./npys/npy_x/x_{cnt}.npy", feature.cpu().numpy())
            np.save(f"./npys/npy_sigma/sigma_{cnt}.npy", sigma.cpu().numpy())

            # if self.calrealbits and not self.training:
            #     decodedx, real_bits = getrealbitsg(feature, gaussian)
            #     total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))

            np.save(f"./npys/npy_z/z_{cnt}.npy", z.cpu().numpy())
            # if self.calrealbits and not self.training:
            #     decodedx, real_bits = getrealbits(z)
            #     total_bits = real_bits

            return total_bits, prob


        def iclr18_estrate_bits_mv(mv):

            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            # if self.calrealbits and not self.training:
            #     decodedx, real_bits = getrealbits(mv)
            #     total_bits = real_bits
            np.save(f"./npys/npy_mv/mv_{cnt}.npy", mv.cpu().numpy())

            return total_bits, prob

        compressed_feature_renorm = torch.nan_to_num(compressed_feature_renorm, nan=0.0)
        recon_sigma = torch.nan_to_num(recon_sigma, nan=1.0)

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv
        
        return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp

def build_model():
    out_channel_M = 96
    out_channel_N = 64
    out_channel_mv = 128

    featurenoise = torch.zeros([out_channel_M, 256 // 16, 256 // 16])
    znoise = torch.zeros([out_channel_N, 256 // 64, 256 // 64])
    mvnois = torch.zeros([out_channel_mv, 256 // 16, 256 // 16])
    quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(featurenoise), -0.5, 0.5).to('cuda')
    quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(znoise), -0.5, 0.5).to('cuda')
    quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(mvnois), -0.5, 0.5).to('cuda')


    frame = torch.rand((1, 3, 256, 256)).to('cuda')
    ref = torch.rand((1, 3, 256, 256)).to('cuda')
    net = VideoCompressor().to('cuda')

    clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(frame, ref, quant_noise_feature, quant_noise_z, quant_noise_mv)

    print(f'Input Data: {frame.shape}')
    print(f'Output Result: {clipped_recon_image.shape}')
    print(mse_loss)
    print(warploss)
    print(interloss)
    print(bpp_feature)
    print(bpp_z)
    print(bpp_mv)
    print(bpp)
    


if __name__ == '__main__':
  build_model()
