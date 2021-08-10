import torch
from torch.distributions.normal import Normal
from torch.nn import functional as F
import torch.nn as nn

# from pytorch_ssim import SSIM
# from pytorch_ssim import ssim
from torch.autograd import Variable

# Multi-view MSE Loss
class MultiviewMSELoss(torch.nn.Module):
    def __init__(self, args):
        super(MultiviewMSELoss, self).__init__()
        self.args = args
        self.loss = nn.MSELoss()

    def forward(self, emb):
        total_loss = self.loss(emb[1], emb[0]) + self.loss(emb[3], emb[2]) + self.loss(emb[5], emb[4])

        # L2 Regularization
        # l2_regularization = torch.tensor(0).float().cuda()
        # for name, param in model.named_parameters():
        #    if 'bias' not in name:
        #        l2_regularization += self.args.lambda2 * torch.norm(param.cuda(), 2)

        # Take the average
        # loss = torch.mean(pred_loss) + l2_regularization
        # loss = torch.mean(total_loss)

        return total_loss


# SAELOSS
class AELoss(torch.nn.Module):

    def __init__(self, args):
        super(AELoss, self).__init__()
        self.args = args
        self.l1loss = nn.L1Loss(reduction='mean')  # L1 loss
        self.l2loss = nn.MSELoss()  # L2 loss
        # self.baur = BaurLoss()

    def forward(self, model, x, x_hat):

        # x: [3, 193, 229, 193]
        # x_hat: [3, 193, 229, 193]

        # Reconstruction Loss = MSE loss + SSIM
        # MSE loss
        recon_l1_loss = self.l1loss(x, x_hat)
        recon_l2_loss = self.l2loss(x, x_hat)
        # baurloss = self.baur(model, x, x_hat)

        # SSIM
        # if not self.args.ad_mode == 'pretrain':
        x = x.unsqueeze(1)  # [3, 1, 193, 229, 193]
        x = x.permute(0, 4, 1, 2, 3)  # [3, 193, 1, 193, 229]
        x = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4))  # [579, 1, 193, 229]

        x_hat = x_hat.unsqueeze(1)  # [3, 1, 193, 229, 193]
        x_hat = x_hat.permute(0, 4, 1, 2, 3)  # [3, 193, 1, 193, 229]
        x_hat = x_hat.contiguous().view(-1, x_hat.size(2), x_hat.size(3), x_hat.size(4))  # [579, 1, 193, 229]
        # else:
        # x = x.unsqueeze(1)  # [579, 1, 193, 229]
        # x_hat = x_hat.unsqueeze(1)  # [579, 1, 193, 229]

        img1 = Variable(x).cuda()
        img2 = Variable(x_hat).cuda()

        # print(ssim(img1, img2))
        ssim_calculator = SSIM(window_size=11)
        ssim_loss = -ssim_calculator(img1, img2)
        # print(ssim_loss(img1, img2))

        # L2 Regularization
        l2_regularization = torch.tensor(0).float().cuda()
        for name, param in model.named_parameters():
           if 'bias' not in name:
               l2_regularization += self.args.lambda2 * torch.norm(param.cuda(), 2)
               # l2_regularization += self.args.lambda2 * torch.norm(param.cuda(), 1)

        # Take the average
        # loss = torch.mean(recon_loss) + l2_regularization
        # loss = recon_loss + ssim_loss + l2_regularization
        loss = recon_l1_loss + recon_l2_loss + ssim_loss + l2_regularization
        # loss = torch.mean(recon_loss) + baurloss + l2_regularization

        # return loss, recon_loss, baurloss
        return loss, recon_l1_loss, recon_l2_loss, ssim_loss

# ClassifierLoss
class ClassifierLoss(torch.nn.Module):

    def __init__(self, args):
        super(ClassifierLoss, self).__init__()
        self.args = args
        self.loss = nn.CrossEntropyLoss()

    def forward(self, model, y_pred, y):
        # Prediction Loss
        pred_loss = self.loss(y_pred, y)

        # L2 Regularization
        # l2_regularization = torch.tensor(0).float().cuda()
        # for name, param in model.named_parameters():
        #    if 'bias' not in name:
        #        l2_regularization += self.args.lambda2 * torch.norm(param.cuda(), 2)

        # Take the average
        # loss = torch.mean(pred_loss) + l2_regularization
        loss = torch.mean(pred_loss)

        return loss

# VAELOSS
class VAELoss(torch.nn.Module):

    def __init__(self, args):
        super(VAELoss, self).__init__()
        # self.beta = beta
        self.args = args
        self.mse = nn.MSELoss()

    def forward(self, model, z_mu, z_logvar, x, x_hat):
        # Reconstruction Loss
        # recon_loss = -Normal(x_hat_mu, torch.exp(0.5 * x_hat_logvar)).log_prob(x).sum(1)
        # recon_loss = F.binary_cross_entropy(x_hat.view(-1, 784), x.view(-1, 784), reduction='sum')
        recon_loss = self.mse(x, x_hat)

        # Variational Encoder Loss
        KLD_enc = - self.args.beta * 0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), 1)

        # Take the average
        loss = torch.mean(recon_loss) + torch.mean(KLD_enc)

        return loss

# SOMLoss
class SOMLoss(torch.nn.Module):

    def __init__(self, args):
        super(SOMLoss, self).__init__()
        # self.beta = beta
        self.args = args

    def forward(self, weights, distances):
        # Calculate loss
        loss = torch.sum(weights * distances, 1).mean()

        # Regularization
        # l1_regularization = torch.tensor(0).float().cuda()
        # for name, param in model.named_parameters():
        #    if 'bias' not in name:
        #        l1_regularization += self.args.lambda1 * torch.norm(param.cuda(), 1)

        return loss

# HSOMLoss
class HSOMLoss(torch.nn.Module):

    def __init__(self, args):
        super(HSOMLoss, self).__init__()
        # self.beta = beta
        self.args = args

    def forward(self, weights_lower, distances_lower, weights_upper, distances_upper):
        # Calculate loss
        loss = torch.sum(weights_lower * distances_lower, 1).mean() + torch.sum(weights_upper * distances_upper, 1).mean()

        return loss

class BaurLoss(torch.nn.Module):
    def __init__(self, lambda_reconstruction=1.0, lambda_gdl=1.0):
        super(BaurLoss, self).__init__()

        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gdl = lambda_gdl
        self.lambda_gdl = 0

        # self.l1_loss = lambda x, y: nn.PairwiseDistance(p=1)(
        #     x.view(x.shape[0], -1), y.view(y.shape[0], -1)
        # ).mean()
        #
        # self.l2_loss = lambda x, y: nn.PairwiseDistance(p=2)(
        #     x.view(x.shape[0], -1), y.view(y.shape[0], -1)
        # ).mean()

        self.l1_loss = lambda x, y: nn.PairwiseDistance(p=1)(x, y).mean()
        self.l2_loss = lambda x, y: nn.PairwiseDistance(p=2)(x, y).mean()

    @staticmethod
    def __image_gradients(image):
        input_shape = image.shape
        batch_size, depth, height, width = input_shape

        dz = image[:, 1:, :, :] - image[:, :-1, :, :]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        dx = image[:, :, :, 1:] - image[:, :, :, :-1]

        dzz = torch.tensor(()).new_zeros(
            (batch_size, 1, height, width),
            device=image.device,
            dtype=dz.dtype,
        )
        dz = torch.cat([dz, dzz], 1)
        dz = torch.reshape(dz, input_shape)

        dyz = torch.tensor(()).new_zeros(
            (batch_size, depth, 1, width),
            device=image.device,
            dtype=dy.dtype
        )
        dy = torch.cat([dy, dyz], 2)
        dy = torch.reshape(dy, input_shape)

        dxz = torch.tensor(()).new_zeros(
            (batch_size, depth, height, 1),
            device=image.device,
            dtype=dx.dtype,
        )
        dx = torch.cat([dx, dxz], 3)
        dx = torch.reshape(dx, input_shape)

        return dx, dy, dz

    def forward(self, model, X, X_hat):

        X = X
        X_hat = X_hat

        l1_reconstruction = (
            self.l1_loss(X, X_hat) * self.lambda_reconstruction
        )
        l2_reconstruction = (
            self.l2_loss(X, X_hat) * self.lambda_reconstruction
        )

        X_gradients = self.__image_gradients(X)
        X_hat_gradients = self.__image_gradients(X_hat)

        l1_gdl = (
            self.l1_loss(X_gradients[0], X_hat_gradients[0])
            + self.l1_loss(X_gradients[1], X_hat_gradients[1])
            + self.l1_loss(X_gradients[2], X_hat_gradients[2])
        ) * self.lambda_gdl

        l2_gdl = (
            self.l2_loss(X_gradients[0], X_hat_gradients[0])
            + self.l2_loss(X_gradients[1], X_hat_gradients[1])
            + self.l2_loss(X_gradients[2], X_hat_gradients[2])
        ) * self.lambda_gdl

        loss_total = l1_reconstruction + l2_reconstruction + l1_gdl + l2_gdl

        return loss_total

