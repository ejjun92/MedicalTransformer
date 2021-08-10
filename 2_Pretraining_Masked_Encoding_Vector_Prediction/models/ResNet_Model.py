import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import math
import copy
import numpy as np
import torch.nn.functional as F


class Multiview_MEP(nn.Module):
    def __init__(self, args):
        super(Multiview_MEP, self).__init__()

        self.args = args

        # nonlineraity
        self.relu = nn.ReLU()
        self.act_f = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.encoding = ResNet(self.args.inplanes, self.args.depth, self.args.d_f, self.args, bottleneck=False)

        # transformer
        self.TF = Transformer(self.args)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # encode
        # x [B, 193, 229, 193]
        x = x.unsqueeze(1)  # [B, 1, 193, 229, 193]

        # axial
        x_axial = x.clone()  # [B, 1, 193, 229, 193]
        mask_ind = np.random.choice(x_axial.size(4), int(x_axial.size(4) * self.args.mask_ratio), replace=False)
        masked_x = Variable((-1) * torch.ones(x_axial.size(0), x_axial.size(2), x_axial.size(3)).unsqueeze(1).unsqueeze(-1), requires_grad=True).cuda()  # [B, 1, 193, 229]
        masked_x = masked_x.expand([masked_x.size(0), masked_x.size(1), masked_x.size(2), masked_x.size(3), len(mask_ind)])  # [4, 1, 193, 229, 19]
        # print("axial", mask_ind)
        encoding_axial = self.encoding(x_axial)  # [B, 256, 193]
        mask_encoding_axial = self.encoding(masked_x)  # [B, 256, 19]

        emb_axial = encoding_axial.clone()  # [1, 256, 193]
        emb_gt_axial = emb_axial[:, :, mask_ind].clone()  # [1, 256, 1]
        emb_axial[:, :, mask_ind] = mask_encoding_axial.clone()  # [MASK]

        attn_mask = torch.zeros(encoding_axial.size(0), encoding_axial.size(2))  # [B, 193]
        tf_axial = self.TF(emb_axial.permute(0, 2, 1), attn_mask.cuda(), [0]).permute(0, 2, 1)  # [B, 256, 193]

        hidden = tf_axial

        emb_pred_axial = hidden[:, :, mask_ind]
        del mask_ind, masked_x, attn_mask, hidden


        # sagittal
        x_sagittal = x.clone().permute(0, 1, 4, 3, 2)  # [B, 1, 193, 229, 193]
        mask_ind = np.random.choice(x_sagittal.size(4), int(x_sagittal.size(4) * self.args.mask_ratio), replace=False)
        masked_x = Variable((-1) * torch.ones(x_axial.size(0), x_axial.size(2), x_axial.size(3)).unsqueeze(1).unsqueeze(-1), requires_grad=True).cuda()  # [B, 1, 193, 229]
        masked_x = masked_x.expand([masked_x.size(0), masked_x.size(1), masked_x.size(2), masked_x.size(3), len(mask_ind)])  # [4, 1, 193, 229, 19]
        # print("sagittal", mask_ind)
        encoding_sagittal = self.encoding(x_sagittal)  # [B, 256, 193]
        mask_encoding_sagittal = self.encoding(masked_x)  # [B, 256, 19]

        emb_sagittal = encoding_sagittal.clone()  # [1, 256, 193]
        emb_gt_sagittal = emb_sagittal[:, :, mask_ind].clone()  # [1, 256, 1]
        emb_sagittal[:, :, mask_ind] = mask_encoding_sagittal.clone()  # [MASK]

        attn_mask = torch.zeros(encoding_sagittal.size(0), encoding_sagittal.size(2))  # [B, 193]
        tf_sagittal = self.TF(emb_sagittal.permute(0, 2, 1), attn_mask.cuda(), [1]).permute(0, 2, 1)  # [B, 256, 193]

        hidden = tf_sagittal

        emb_pred_sagittal = hidden[:, :, mask_ind]

        del mask_ind, masked_x, attn_mask, hidden


        # coronal
        x_coronal = x.clone().permute(0, 1, 2, 4, 3)  # [B, 1, 193, 193, 229]
        mask_ind = np.random.choice(x_coronal.size(4), int(x_coronal.size(4) * self.args.mask_ratio), replace=False)
        masked_x = Variable((-1) * torch.ones(x_coronal.size(0), x_coronal.size(2), x_coronal.size(3)).unsqueeze(1).unsqueeze(-1), requires_grad=True).cuda()  # [B, 1, 193, 193]
        masked_x = masked_x.expand([masked_x.size(0), masked_x.size(1), masked_x.size(2), masked_x.size(3), len(mask_ind)])  # [4, 1, 193, 193, 19]
        # print("coronal", mask_ind)
        encoding_coronal = self.encoding(x_coronal)  # [B, 256, 193]
        mask_encoding_coronal = self.encoding(masked_x)  # [B, 256, 19]

        emb_coronal = encoding_coronal.clone()  # [1, 256, 193]
        emb_gt_coronal = emb_coronal[:, :, mask_ind].clone()  # [1, 256, 1]
        emb_coronal[:, :, mask_ind] = mask_encoding_coronal.clone()  # [MASK]

        attn_mask = torch.zeros(encoding_coronal.size(0), encoding_coronal.size(2))  # [B, 193]
        tf_coronal = self.TF(emb_coronal.permute(0, 2, 1), attn_mask.cuda(), [2]).permute(0, 2, 1)  # [B, 256, 193]

        hidden = tf_coronal

        emb_pred_coronal = hidden[:, :, mask_ind]
        del mask_ind, masked_x, attn_mask, hidden

        emb_matrix = [emb_gt_axial.squeeze(-1), emb_pred_axial.squeeze(-1),
                      emb_gt_sagittal.squeeze(-1), emb_pred_sagittal.squeeze(-1),
                      emb_gt_coronal.squeeze(-1), emb_pred_coronal.squeeze(-1)]

        return emb_matrix


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=None, padding=0, dilation=1):
        super(ConvLayer, self).__init__()

        if padding != 0:
            self.replication_pad = nn.ReplicationPad3d((0, 0, padding, padding, padding, padding))  # TODO : replicate
        else:
            self.replication_pad = None
        kernel_size = (kernel, kernel, 1)
        stride_size = (stride, stride, 1)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride_size, bias=False)

    def forward(self, x):
        if self.replication_pad is not None :
            x = self.replication_pad(x)
        out = self.conv(x)
        return out


class TransposeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dilation=1):
        super(TransposeConvLayer, self).__init__()
        # if pad is not None:
        #     self.reflection_pad = nn.ReplicationPad3d((0, 0, pad[0], pad[1], pad[2], pad[3]))
        # else:
        #     self.reflection_pad = None

        kernel_size = (kernel, kernel, 1)
        stride_size = (stride, stride, 1)
        stride_1_size = (1, 1, 1)

        self.deconv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size, stride_size, bias=False)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), stride=stride_1_size,
                              padding=(1, 1, 0), bias=False)


    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        # if self.reflection_pad is not None:
        #     x1 = self.reflection_pad(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [0, 0, 0, diffX, 0, diffY], mode='replicate')
        x1 = self.conv(x1)
        return x1


# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, f_in, f_out):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(f_in, f_out, kernel=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm3d(f_out, affine=True)
        # self.in1 = nn.BatchNorm3d(f_out, affine=True)
        self.act_f = nn.LeakyReLU(inplace=True)
        self.conv2 = ConvLayer(f_out, f_out, kernel=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm3d(f_out, affine=True)
        # self.in2 = nn.BatchNorm3d(f_out, affine=True)
        self.down = ConvLayer(f_in, f_out, kernel=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        out = self.act_f(self.in1(self.conv1(x)))
        # print(x.mean())
        # print(out.mean())
        out = self.in2(self.conv2(out))
        out = out + self.down(residual)
        out = self.act_f(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, f_in, f_out):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(f_in, f_out, kernel=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm3d(f_out, affine=True)
        # self.in1 = nn.BatchNorm3d(f_out, affine=True)
        self.act_f = nn.LeakyReLU(inplace=True)
        self.conv2 = ConvLayer(f_out, f_out, kernel=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm3d(f_out, affine=True)
        # self.in2 = nn.BatchNorm3d(f_out, affine=True)

    def forward(self, x):
        out = self.act_f(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.act_f(out)
        return out


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # self.embed = Embedder(self.args.input_dim, self.args.d_model)
        self.pos_embed = PositionalEncoder(self.args)
        # self.pos_embed = nn.Embedding(self.args.slice_len, self.args.d_f)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.args.slice_len + 1, d_model), freeze=True)
        # self.seg_embed = nn.Embedding(2, self.args.d_f)

        self.encoding_block = Encoding_Block(self.args.d_f, self.args.num_heads, self.args.d_ff, self.args.num_stack)

    def forward(self, data, attn_m, view_idx):
        """
        :param src: Batch x Max_seq_len x Variable
        :param mask: Batch x Max_seq_len x Max_seq_len
        """
        batch_size, seq_len, var_num = data.size()  # [1, 193, 1152]

        # attention mask
        attn_mask = attn_m.unsqueeze(1)  # [1, 1, 193, 1152]
        attn_mask = attn_mask.expand(batch_size, seq_len, seq_len)

        # positional embedding
        # pos = torch.arange(seq_len, dtype=torch.long).cuda()
        # e = data + self.pos_embed(pos)
        e = self.pos_embed(data, view_idx)  # [3, 193, 32]

        x = self.encoding_block(e, mask=attn_mask)  # [3, 193, 32]
        return x


class Encoding_Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_stack):
        super().__init__()

        self.N = num_stack

        self.layers = get_clones(EncoderLayer(d_model, num_heads, d_ff), num_stack)
        self.norm = Norm(d_model)

    def forward(self, q, mask):
        # MHA Encoding
        for i in range(self.N):
            q = self.layers[i](q, mask)

        # Normalize
        encoded_data = self.norm(q)
        return encoded_data


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.embed = nn.Linear(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(self.args.max_slicelen, self.args.d_f)
        for pos in range(self.args.max_slicelen):
            for i in range(0, self.args.d_f, 2):
                pe[pos, i] = math.sin(pos / (self.args.max_slicelen ** ((2 * i) / self.args.d_f)))
                pe[pos, i + 1] = math.cos(pos / (self.args.max_slicelen ** ((2 * (i + 1)) / self.args.d_f)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.seg_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.args.d_f)

    def forward(self, x, view_idx):
        # make embeddings relatively larger
        x = x * math.sqrt(self.args.d_f)

        # add constant to embedding
        seq_len = x.size(1)

        pos_emb = Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        pos_emb = pos_emb.expand_as(x)

        seq = torch.tensor(view_idx * (seq_len) , dtype=torch.long).cuda()
        seq = seq.unsqueeze(0).expand([x.size(0), x.size(1)])
        seq_emb = self.seg_embedding(seq)

        x = x + pos_emb + seq_emb  # [4, 193, 16]

        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_k = 256
        self.h = heads
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, self.d_k * self.h)
        self.v_linear = nn.Linear(d_model, self.d_k * self.h)
        self.k_linear = nn.Linear(d_model, self.d_k * self.h)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.d_k * self.h, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)  # [batch_size * len_q * n_heads * hidden_dim]

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]
        q = q.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]
        v = v.transpose(1, 2)  # [batch_size * n_heads * len_q * hidden_dim]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)  # [batch_size x n_heads x len_q x len_k]

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # if tf == 1:
        #     scores = scores.transpose(2, 3)

        # concatenate heads and put through final linear layer

        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_k * self.h)

        output = self.out(concat)

        return output # [b, 193, 1152]


def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    :param q: Batch x n_head x max_seq_len x variable
    :param k: Batch x n_head x max_seq_len x variable
    :param v: Batch x n_head x max_seq_len x variable
    :param d_k:
    :param mask: Batch x n_had x max_seq_len x max_seq_len
    :param dropout:
    :return:
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [Batch x n_head x max_seq_len x max_seq_len]

    if mask is not None:
        # scores = scores.masked_fill(mask, -1e9)
        # scores = scores.masked_fill(torch.tensor(mask, dtype=torch.uint8).cuda(), -1e9)
        scores = scores.masked_fill(mask.bool(), -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.attn = MultiHeadAttention(heads, d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x2 = self.norm_1(x)
        # x = self.dropout_2(self.ff(self.dropout_1(self.attn(x2, x2, x2, mask))))

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x



#############
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=(stride, stride, 1),
                     padding=(1, 1, 0), bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 1), stride=(stride, stride, 1), padding=(1, 1, 0), bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * Bottleneck.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, inplanes, depth, num_classes, args, bottleneck=False):
        super(ResNet, self).__init__()
        # self.dataset = dataset
        #
        # if self.dataset.startswith('cifar'):
        #     self.inplanes = 16
        #     print(bottleneck)
        #
        #     if bottleneck == True:
        #         n = int((depth - 2) / 9)
        #         block = Bottleneck
        #     else:
        #         n = int((depth - 2) / 6)
        #         block = BasicBlock
        #     self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        #     self.bn1 = nn.BatchNorm2d(self.inplanes)
        #     self.relu = nn.ReLU(inplace=True)
        #     self.layer1 = self._make_layer(block, 16, n)
        #     self.layer2 = self._make_layer(block, 32, n, stride=2)
        #     self.layer3 = self._make_layer(block, 64, n, stride=2)
        #     self.avgpool = nn.AvgPool2d(8)
        #     self.fc = nn.Linear(64 * block.expansion, num_classes)

        # elif dataset == 'imagenet':

        self.args = args
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
        self.inplanes = inplanes
        # self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.f_out = [inplanes, inplanes*2, inplanes*4, inplanes*8]

        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=(7, 7, 1), stride=(2, 2, 1), padding=(3, 3, 0), bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.InstanceNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.layer1 = self._make_layer(blocks[depth], self.f_out[0], layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], self.f_out[1], layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], self.f_out[2], layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], self.f_out[3], layers[depth][3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        if self.args.is_pool == 1:
            self.avgpool = nn.AvgPool3d((3, 3, 1))
        elif self.args.is_pool == 0:
            self.avgpool = nn.AvgPool3d((7, 7, 1))
        else:
            pass
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.InstanceNorm3d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1, 1), stride=(stride, stride, 1), bias=False),
                nn.InstanceNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # if self.dataset == 'cifar10' or self.dataset == 'cifar100':
        #     x = self.conv1(x)
        #     x = self.bn1(x)
        #     x = self.relu(x)
        #     x = self.layer1(x)
        #     x = self.layer2(x)
        #     x = self.layer3(x)
        #     x = self.avgpool(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)
        # elif self.dataset == 'imagenet':
        # [1, 1, 193, 229, 193]
        x = self.conv1(x)  # [1, 4, 97, 115, 193]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [1, 4, 49, 58, 193]
        x = self.layer1(x)  # [1, 4, 49, 58, 193]
        x = self.layer2(x)  # [1, 64, 25, 29, 193]
        x = self.layer3(x)  # [1, 256, 13, 15, 193]
        x = self.layer4(x)  # [1, 1024, 7, 8, 193]
        x = self.avgpool(x)  # [1, 1024, 1, 1, 193]
        x = x.view(x.size(0), -1, x.size(-1))  # [1, 1024, 193]
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)  # [1, 256, 193)
        # print('hi')
        return x