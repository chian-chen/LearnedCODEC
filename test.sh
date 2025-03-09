# parser.add_argument('--training', default=True, type=bool, help='training')
# parser.add_argument('--pretrain_weight', default=False, type=bool, help='path to pre-train weight')
# parser.add_argument('--pretrain_weight_path', default='weights/best.model', type=str, help='path to pre-train weight')
# parser.add_argument('--output_weight', default='experiment_weights', type=str, help="experiments_weight")

# parser.add_argument('--batch_size', default=8, type=int, help="hyper-parameter: batch_size")
# parser.add_argument('--epochs', default=1, type=int, help="hyper-parameter: epochs")

python3 main.py --pretrain_weight --ref_dir 'H265L20' --pretrain_weight_path './weights/2048.model' --batch_size 1
# python3 main.py --pretrain_weight --ref_dir 'H265L23' --pretrain_weight_path './weights/1024.model' --batch_size 1
# python3 main.py --pretrain_weight --ref_dir 'H265L26' --pretrain_weight_path './weights/512.model' --batch_size 1
# python3 main.py --pretrain_weight --ref_dir 'H265L29' --pretrain_weight_path './weights/256.model' --batch_size 1

# python3 main.py --pretrain_weight --pretrain_weight_path './experiment_weights/40000.ckpt' --batch_size 1
