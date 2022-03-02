import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import torchvision
import einops


class MedGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_content', type=float, default=0.0001, help='weight for content loss')
            parser.add_argument('--lambda_style', type=float, default=0.0001, help='weight for style loss')
            parser.add_argument('--lambda_pec', type=float, default=20.0, help='weight for pec loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_Pec', 'G_content', 'G_Style', 'D_real', 'D_fake']
        # ['G_GAN', 'G_Pec','G_Content', 'G_Style', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.vgg19 =torchvision.models.vgg19(pretrained=True).to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # calculate L_content
        self.loss_G_content = cal_ContentLoss(self.real_B, self.fake_B, self.vgg19) * self.opt.lambda_content
        # cal L_Style
        self.loss_G_Style = cal_StyleLoss(self.real_B, self.fake_B, self.vgg19) * self.opt.lambda_style
        # cal L_Pec
        self.loss_G_Pec = self.cal_percep(real_AB, fake_AB) * self.opt.lambda_pec
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_content + self.loss_G_Style + self.loss_G_Pec
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def cal_percep(self, real_AB, fake_AB, layer=[0,2,5,8]):
        L_percep = 0
        real_feats = self.netD(real_AB, layer)
        fake_feats = self.netD(fake_AB, layer)
        l1_crition = torch.nn.L1Loss(reduction='sum')
        for i in range(len(layer)):
            d = real_feats[i].shape[1]
            h = real_feats[i].shape[2]
            w = real_feats[i].shape[3]
            L_percep = L_percep + l1_crition(real_feats[i],fake_feats[i])/ (h*w*d)
        return L_percep

def cal_ContentLoss(real_img, fake_img, vgg19, layer_chosed=[1,6,11,20,29]):
    assert(real_img.shape[1]==3)
    # real_img = real_img.repeat(1,3,1,1)
    # fake_img = fake_img.repeat(1,3,1,1)
    l1_crition = torch.nn.L1Loss()
    real_feat = real_img
    fake_feat = fake_img
    loss_temp = 0
    for i,layer in enumerate(vgg19.features):
        real_feat = layer(real_feat)
        fake_feat = layer(fake_feat)
        if i in layer_chosed:
            dim = real_feat.shape[1]
            h = real_feat.shape[2]
            w = real_feat.shape[3]
            loss_temp = l1_crition(real_feat, fake_feat) / (dim*h*w) + loss_temp
    return loss_temp

def cal_StyleLoss(real_img, fake_img, vgg16, layer_chosed=[1,6,11,20,29]):
    assert(real_img.shape[1]==3)
    # real_img = real_img.repeat(1,3,1,1)
    # fake_img = fake_img.repeat(1,3,1,1)
    l1_crition = torch.nn.L1Loss(reduction='sum')
    loss_temp = 0
    real_feat = real_img
    fake_feat = fake_img
    for i,layer in enumerate(vgg16.features):
        real_feat = layer(real_feat)
        fake_feat = layer(fake_feat)
        if i in layer_chosed:
            dim = real_feat.shape[1] # [b, c, h, w]
            loss_temp = loss_temp + l1_crition(cal_GMatrix(real_feat), cal_GMatrix(fake_feat)) / (4*dim*dim)
    return loss_temp

def cal_GMatrix(features):
    dim = features.shape[1]
    h = features.shape[2]
    w = features.shape[3]
    GM = torch.einsum('bmhw, bnhw -> bmn', features, features)
    # GM = torch.zeros([dim,dim])
    # for i in range(dim):
    #     for j in range(dim):
    #         GM[i,j] = torch.bmm(features[:,i,:].view(1,1,-1), features[:,j,:].view(1,-1,1))
    return GM / (h*w*dim)

