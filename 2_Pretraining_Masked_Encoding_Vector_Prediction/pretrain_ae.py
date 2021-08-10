import torch.optim as optim
from models.ResNet_Model import Multiview_MEP
from losses import *
from helpers import *
import os
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="4")
parser.add_argument("--pre_dataset", type=str, default='ixi_camcan_abide')
parser.add_argument("--model", type=str, default='7_Multiview_MEP_CN_ResNet_freeze')
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--is_pool", type=int, default=1)

parser.add_argument("--is_finetune_resnet", type=int, default=1)

parser.add_argument("--mask_ratio", type=float, default=0.1)
# parser.add_argument("--sample_ratio", type=float, default=0.5)

parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=5e-4)  # 5e-4
parser.add_argument("--batch_size", type=int, default=4)
# parser.add_argument("--lambda1", type=float, default=0.0001)
parser.add_argument("--lambda2", type=float, default=0.0000)

parser.add_argument("--depth", type=int, default=18)
parser.add_argument("--inplanes", type=int, default=16)
parser.add_argument("--d_f", type=int, default=64)

# Transformer
# parser.add_argument("--max_slicelen", type=int, default=229)
# parser.add_argument("--axial_slicelen", type=int, default=193)
# parser.add_argument("--coronal_slicelen", type=int, default=229)

parser.add_argument("--max_slicelen", type=int, default=114)
parser.add_argument("--axial_slicelen", type=int, default=96)
parser.add_argument("--coronal_slicelen", type=int, default=114)


parser.add_argument("--d_ff", type=int, default=128)
parser.add_argument("--num_stack", type=int, default=1)
parser.add_argument("--num_heads", type=int, default=4)
# parser.add_argument("--slice_len", type=int, default=193)

# parser.add_argument("--class_scenario", type=str, default='cn_mci_ad')
# parser.add_argument("--class_scenario", type=str, default='mci_ad')
# parser.add_argument("--class_scenario", type=str, default='cn_mci')
parser.add_argument("--class_scenario", type=str, default='cn_ad')
args = parser.parse_args()

# GPU Configuration
# gpu_id = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Dataset
images = np.load('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/data_norm_min_max.npy', mmap_mode="r", allow_pickle=True)  # (566, 193, 229, 193)
indexes = np.load('/DataCommon/ejjun/dataset/IXI_camcan_ABIDE/output/10idx.npy', allow_pickle=True)  # (509,), (57,)
labels = []

# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
directory = './log/%s/%s_batch_%d_lr_%f_lambda2_%f_ResNet_%d_inplanes_%d/' % (args.pre_dataset, date_str, args.batch_size, args.lr, args.lambda2, args.depth, args.inplanes)

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + 'img/')
    os.makedirs(directory + 'img/train/')
    os.makedirs(directory + 'img/valid/')
    os.makedirs(directory + 'img/test/')
    os.makedirs(directory + 'tflog/')
    os.makedirs(directory + 'model/')

# Text Logging
f = open(directory + 'setting.log', 'a')
writelog(f, '======================')
writelog(f, 'Model: %s' % (args.model))
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, '----------------------')
# writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '======================')
f.close()

f = open(directory + 'log.log', 'a')
# Tensorboard Logging
# tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/train_')
# tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/valid_')
# tfw_test = tf.compat.v1.summary.FileWriter(directory + 'tflog/kfold_' + str(args.fold) + '/test_')
tfw_train = tf.compat.v1.summary.FileWriter(directory + 'tflog//train_')
tfw_valid = tf.compat.v1.summary.FileWriter(directory + 'tflog//valid_')

# Tensor Seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

# Define Loaders
writelog(f, 'Load training data')
train_loader = sample_loader('train', images, labels, indexes, args, drop_last=True)
writelog(f, 'Load validation data')
valid_loader = sample_loader('valid', images, labels, indexes, args, shuffle=False)
# writelog(f, 'Load test data')
# test_loader = sample_loader('test', images, labels, indexes, args, shuffle=False)
dataloaders = {'train': train_loader,
               'valid': valid_loader}
               # 'test': test_loader}


def train(dataloader, dir='.'):
    # Set mode as training
    model.train()

    # Define training variables
    train_loss = 0
    n_batches = 0

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):
        x = Variable(xbatch['data'].float(), requires_grad=True).cuda()  # [4, 96, 114, 96]

        if args.is_finetune_resnet == 1:
            for param in model.module.encoding.parameters():
                # print(param)
                param.requires_grad = False
        else:
            for param in model.module.encoding.parameters():
                param.requires_grad = True

        # Zero Grad
        optimizer.zero_grad()

        emb = model(x)  # [1, 193, 193]

        # Calculate Loss
        loss = creterion(emb)
        loss.backward()
        optimizer.step()
        print('Training loss = (%.5f)' % loss)

        # Backpropagation & Update the weights
        train_loss += (loss.item() * x.size(0))
        n_batches += 1

    # Take average
    # train_loss = train_loss / n_batches
    train_loss = train_loss / len(dataloader.dataset)
    writelog(f, 'Train Loss: %.8f' % train_loss)

    # Tensorboard Logging
    info = {
            'loss': train_loss,
           }
    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        tfw_train.add_summary(summary, epoch)


def evaluate(phase, dataloader, dir='.'):
    # Set mode as training
    model.eval()

    # Define training variables
    test_loss = 0
    n_batches = 0

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):
            x = xbatch['data'].float().cuda()

            emb = model(x)  # [1, 193, 193]

            # Calculate Loss
            loss = creterion(emb)
            print('(%s) loss = (%.5f)' % (phase, loss))

            test_loss += (loss.item() * x.size(0))
            n_batches += 1

    # Take average
    # test_loss = test_loss / n_batches
    test_loss = test_loss / len(dataloader.dataset)
    writelog(f, '%s Loss: %.8f' % (phase, test_loss))

    # Tensorboard Logging
    info = {'loss': test_loss}

    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        if phase == 'Validation':
            tfw_valid.add_summary(summary, epoch)
        # else:
        #     tfw_test.add_summary(summary, epoch)

    return test_loss


# Define Model
model = nn.DataParallel(Multiview_MEP(args)).to(device)

pretrain_directory = '/DataCommon/ejjun/MedBERT/experiment/0_prepretrain/log/ixi_camcan_abide/20210129.19.22.57_batch_8_lr_0.000100_lambda2_0.001000_ResNet_18_inplanes_16/model/0_ResNet_prepretrain.pt'

pretrained_dict = torch.load(pretrain_directory)
model_dict = model.state_dict()
for k, v in pretrained_dict.items():
    if k in model_dict:
        print(k)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


# loss function
creterion = MultiviewMSELoss(args)
# creterion = nn.L1Loss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lambda2)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, last_epoch=-1)

# Define data type
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Best epoch checking
valid = {
    'epoch': 0,
    'auc': 0,
}

# Train Epoch
ES = EarlyStopping(delta=0, patience=30, verbose=True)
for epoch in range(args.epoch):
    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, 'Training')
    train(dataloaders['train'], dir=directory)

    writelog(f, 'Validation')
    loss_val = evaluate('Validation', dataloaders['valid'], dir=directory)

    if epoch == 0:
        valid['loss'] = loss_val

    # Save Model
    if loss_val < valid['loss']:
        torch.save(model.state_dict(), directory + '/model/%s_%d.pt' % (args.model, epoch))
        writelog(f, 'Best validation loss is found! Validation loss : %f' % loss_val)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss_val
        valid['epoch'] = epoch
        ES(loss_val, None)

    # loss_test = evaluate('Test', dataloaders['test'], dir=directory)

    scheduler.step()
    if ES.early_stop == True:
        break

writelog(f, 'Best model for testing: epoch %d-th' % valid['epoch'])

del model

# Define Model
model = nn.DataParallel(Multiview_MEP(args)).to(device)
model.load_state_dict(torch.load(directory + '/model/%s_%d.pt' % (args.model, valid['epoch'])))

# writelog(f, 'Testing')
# loss = evaluate('test', dataloaders['test'], dir=directory)

writelog(f, 'END OF TRAINING')
f.close()