import argparse
from tqdm import tqdm

import torch

from dataset import get_loader
from model import CifarNet
import utils

parser = argparse.ArgumentParser()

parser.add_argument(
    '--fbs',
    type=utils.str2bool,
    default=False
)
parser.add_argument(
    '--sparsity_ratio',
    type=float,
    default=1.0
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=256
)

parser.add_argument(
    '--num_worker',
    type=int,
    default=4
)
parser.add_argument(
    '--ckpt_path',
    type=str,
    default='checkpoints'
)

args = parser.parse_args()

train_loader, test_loader = get_loader(args.batch_size, args.num_worker)
model = CifarNet(fbs=args.fbs, sparsity_ratio=args.sparsity_ratio, test=True).cuda()

# state_dict = torch.load(f'{args.ckpt_path}/backup/best_{args.fbs}_{args.sparsity_ratio}.pt')
state_dict = torch.load('{}/best_{}_{}.pt'.format(args.ckpt_path, args.fbs, args.sparsity_ratio))
model.load_state_dict(state_dict)

with torch.no_grad():
    total_num = 0
    correct_num = 0

    model.eval()
    MACs_total = 0
    for img_batch, lb_batch in tqdm(test_loader, total=len(test_loader)):
        img_batch = img_batch.cuda()
        lb_batch = lb_batch.cuda()
        if args.fbs:
            pred_batch, _, MACs = model(img_batch)
            MACs_total += MACs.sum().item()
        else:
            pred_batch, MACs = model(img_batch)
            #             MACs_total += MACs * img_batch.size(0)
            MACs_total += MACs.sum().item()

        _, pred_lb_batch = pred_batch.max(dim=1)
        total_num += lb_batch.shape[0]
        correct_num += pred_lb_batch.eq(lb_batch).sum().item()

    test_acc = 100. * correct_num / total_num
    MACs_avg = MACs_total / total_num

print('Test accuracy: {}% MACs: {:.2f}'.format(test_acc, MACs_avg / 1e+8))

with open('fig3a/results.tsv', 'a') as f:
    f.write('{}\t{}\t{}\t{}\n'.format(args.fbs, args.sparsity_ratio, test_acc, MACs_avg))