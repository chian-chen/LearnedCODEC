# parser.add_argument('--training', action="store_true", help='training')
# parser.add_argument('--pretrain_weight', action="store_true", help='path to pre-train weight')
# parser.add_argument('--pretrain_weight_path', default='weights/best.model', type=str, help='path to pre-train weight')
# parser.add_argument('--ref_dir', default='H265L20', type=str, help='different lambda, different setting')
# parser.add_argument('--output_weight', default='experiment_weights', type=str, help="experiments_weight")

# parser.add_argument('--batch_size', default=8, type=int, help="hyper-parameter: batch_size")
# parser.add_argument('--epochs', default=1, type=int, help="hyper-parameter: epochs")
# parser.add_argument('--lambda', default=2048, type=int, help="weights for mse-loss -> larger: highr psnr and bpp")

python3 main.py --training --batch_size 8 --epochs 10