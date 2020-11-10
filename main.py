import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

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
    '--lasso_lambda',
    type=float,
    default=1e-8
)

parser.add_argument(
    '--epochs',
    type=int,
    default=500
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=256
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-3
)

parser.add_argument(
    '--seed',
    type=int,
    default=1
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

print(args.fbs, args.sparsity_ratio)

os.makedirs(args.ckpt_path, exist_ok=True)
# with open(f'{args.ckpt_path}/train_log_{args.fbs}_{args.sparsity_ratio}.tsv', 'w') as log_file:
#     log_file.write('epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\tbest_acc\n')

with open('{}/train_log_{}_{}.tsv'.format(args.ckpt_path, args.fbs, args.sparsity_ratio), 'w') as log_file:
    log_file.write('epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\tbest_acc\n')

utils.set_seed(args.seed)

train_loader, test_loader = get_loader(args.batch_size, args.num_worker)
model = CifarNet(fbs=args.fbs, sparsity_ratio=args.sparsity_ratio).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# TODO: initialize current model parameters with previous model parameters


if args.fbs:
    if args.sparsity_ratio == 1.:
        # original CNN -> FBS
        state_dict = torch.load('./checkpoints/best_False_1.0.pt')
        for i in range(8):
            l = 'layer{}'.format(str(i))
            model.state_dict()[l + '.conv.weight'].copy_(state_dict[l + '.conv.weight'])
            model.state_dict()[l + '.conv.bias'].copy_(state_dict[l + '.conv.bias'])
    else:
        # FBS -> FBS
        state_dict = torch.load('./checkpoints/best_True_{}.pt'.format(round(args.sparsity_ratio + 0.1, 1)))
        for i in range(8):
            l = 'layer{}'.format(str(i))
            model.state_dict()[l + '.conv.weight'].copy_(state_dict[l + '.conv.weight'])
            model.state_dict()[l + '.conv.bias'].copy_(state_dict[l + '.conv.bias'])

            model.state_dict()[l + '.bn.weight'].copy_(state_dict[l + '.bn.weight'])
            model.state_dict()[l + '.bn.bias'].copy_(state_dict[l + '.bn.bias'])

            model.state_dict()[l + '.weights'].copy_(state_dict[l + '.weights'])
            model.state_dict()[l + '.bias'].copy_(state_dict[l + '.bias'])

        model.state_dict()['fc.weight'].copy_(state_dict['fc.weight'])
        model.state_dict()['fc.bias'].copy_(state_dict['fc.bias'])

best_acc = 0.
step = 0
patience = 50
for epoch in range(1, args.epochs + 1):
    # print('Epoch: {}'.format(epoch))

    train_loss = 0
    total_num = 0
    correct_num = 0
    total_step = len(train_loader)

    model.train()
    pbar = tqdm(train_loader, total=total_step)
    for img_batch, lb_batch in pbar:
        img_batch = img_batch.cuda()
        lb_batch = lb_batch.cuda()

        if args.fbs:
            pred_batch, lasso = model(img_batch)
            loss = criterion(pred_batch, lb_batch) + args.lasso_lambda * lasso
        else:
            pred_batch = model(img_batch)
            loss = criterion(pred_batch, lb_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred_lb_batch = pred_batch.max(dim=1)
        total_num += lb_batch.shape[0]
        correct_num += pred_lb_batch.eq(lb_batch).sum().item()

    train_loss = train_loss / total_step
    train_acc = 100. * correct_num / total_num
    print('E:{:3d} L:{:.5f} A:{:.1f}'.format(epoch + 1, train_loss, train_acc))

    with torch.no_grad():
        test_loss = 0
        total_num = 0
        correct_num = 0
        total_step = len(test_loader)

        model.eval()
        for img_batch, lb_batch in tqdm(test_loader, total=len(test_loader)):
            img_batch = img_batch.cuda()
            lb_batch = lb_batch.cuda()
            if args.fbs:
                pred_batch, lasso = model(img_batch)
                loss = criterion(pred_batch, lb_batch) + args.lasso_lambda * lasso
            else:
                pred_batch = model(img_batch)
                loss = criterion(pred_batch, lb_batch)

            test_loss += loss.item()
            _, pred_lb_batch = pred_batch.max(dim=1)
            total_num += lb_batch.shape[0]
            correct_num += pred_lb_batch.eq(lb_batch).sum().item()

        test_loss = test_loss / total_step
        test_acc = 100. * correct_num / total_num
        print('L: {:.5f} Test ACC:{:.1f}'.format(test_loss, test_acc))

    if test_acc > best_acc:
        step = 0
        best_acc = test_acc
        torch.save(model.state_dict(), '{}/best_{}_{}.pt'.format(args.ckpt_path, args.fbs, args.sparsity_ratio))
    else:
        step += 1
        if step > patience:
            print('Early stopped...')
            break

    with open('{}/train_log_{}_{}.tsv'.format(args.ckpt_path, args.fbs, args.sparsity_ratio), 'a') as log_file:
        log_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(epoch, train_loss, test_loss, train_acc, test_acc, best_acc))