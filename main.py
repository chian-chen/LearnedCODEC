import torch
import argparse
from torch.utils.data import DataLoader

from net import VideoCompressor
from dataset import UVGDataset, DataSet
from train import train
from inference import inference


def fixed_seed():
    myseed = 2222222
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def arg_parse():
    parser = argparse.ArgumentParser(description="My Version of DVC")
    
    # ------------------------------------------------------------------------------

    parser.add_argument('--training', action="store_true", help='training')
    parser.add_argument('--pretrain_weight', action="store_true", help='path to pre-train weight')
    parser.add_argument('--pretrain_weight_path', default='weights/best.model', type=str, help='path to pre-train weight')
    parser.add_argument('--ref_dir', default='H265L20', type=str, help='different lambda, different setting')
    parser.add_argument('--output_weight', default='experiment_weights', type=str, help="experiments_weight")

    parser.add_argument('--batch_size', default=8, type=int, help="hyper-parameter: batch_size")
    parser.add_argument('--epochs', default=1, type=int, help="hyper-parameter: epochs")
    parser.add_argument('--lambda', default=2048, type=int, help="weights for mse-loss -> larger: highr psnr and bpp")

    # ------------------------------------------------------------------------------

    args = parser.parse_args()
    return args

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


if __name__ == "__main__":

    fixed_seed()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = arg_parse()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.training:
        model = VideoCompressor(training=args.training).to(device)
        train_set = DataSet()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        train(model, train_loader, args, device)
    else:
        model = VideoCompressor(training=args.training).to(device)
        if args.pretrain_weight:
            load_model(model=model, f=args.pretrain_weight_path)
        inference_set = UVGDataset(refdir=args.ref_dir, testfull=True)
        inference_loader = DataLoader(inference_set, batch_size=args.batch_size, shuffle=False)
        inference(model, inference_loader, args, device)
