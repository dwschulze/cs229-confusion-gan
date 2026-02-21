import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
import torch.nn as nn
from models.IHC_Classifier import IHC_Classifier


class ConfusionGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--lambda_A",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_B",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )
            parser.add_argument("--beta", type=float, default=1.0)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
        ]
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B", "E_A", "E_B"]
        else:
            self.model_names = ["G_A", "G_B"]

        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netG_B = networks.define_G(
            opt.output_nc,
            opt.input_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        self.IHC_classfier = IHC_Classifier().to(self.device)
        checkpoint = torch.load(self.opt.pretrained_IHC_Classifier, map_location="cpu")
        for k, v in self.IHC_classfier.named_parameters():
            v.requires_grad = False
        print(self.IHC_classfier.load_state_dict(checkpoint))
        self.IHC_classfier = self.IHC_classfier.to(self.gpu_ids[0])

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD_B = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netE_A = networks.define_E(
                opt.input_nc * opt.batch_size,
                opt.ndf,
                opt.netE,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
                32,
            )
            self.netE_B = networks.define_E(
                opt.input_nc * opt.batch_size,
                opt.ndf,
                opt.netE,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
                32,
            )

        if self.isTrain:
            if (
                opt.lambda_identity > 0.0
            ):  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(
                opt.pool_size
            )  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(
                opt.pool_size
            )  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(
                self.device
            )  # define GAN loss.

            self.Patho_loss = torch.nn.BCELoss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_E = torch.optim.Adam(
                itertools.chain(self.netE_A.parameters(), self.netE_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        # RS_IHC and RS_HE are now [B, 32, C, H, W] after batching
        # We need to process them to get list of [32, C, H, W] for each batch item
        Rs_IHC_batch = input["RS_IHC"].to(self.device)  # [B, 32, C, H, W]
        Rs_HE_batch = input["RS_HE"].to(self.device)  # [B, 32, C, H, W]

        # Convert to list of [32, C, H, W] tensors for each item in batch
        B = Rs_IHC_batch.shape[0]
        self.Rs_IHC = [Rs_IHC_batch[i] for i in range(B)]
        self.Rs_HE = [Rs_HE_batch[i] for i in range(B)]
        self.A_label = input["A_label"].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        self.fake_B_label = self.IHC_classfier(self.fake_B)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def insert_fea_loss(self, real_feas, fake_feas, netE, fake_label):
        """
        Mix real and fake features and compute loss.
        real_feas: [num_rs, C, H, W] - features from reference set (e.g., 32 samples)
        fake_feas: [B, C, H, W] - features from fake images (B = batch_size)
        """
        num_rs, C, H, W = real_feas.shape
        B = fake_feas.shape[0]

        # For each fake in the batch, create a mixed set with all real features
        # We'll process each fake separately and average the losses
        total_loss = 0.0

        for b in range(B):
            fake_feat = fake_feas[b : b + 1]  # [1, C, H, W]
            # Concatenate all real features with this one fake
            all_feas = torch.cat([real_feas, fake_feat], 0)  # [num_rs+1, C, H, W]

            # Labels: 0 for real, fake_label for fake
            all_labels = [0] * num_rs + [fake_label]
            all_labels = torch.Tensor(all_labels).to(real_feas.device)

            if fake_label == 1:
                # Random permutation including the fake (last index = num_rs)
                idx = random.sample(list(range(1, num_rs + 1)), num_rs)
                idx = torch.LongTensor(idx).to(real_feas.device)
                mix_labels = torch.gather(all_labels, 0, idx)

                idx = idx.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, C, H, W)
                mix_feas = torch.gather(all_feas, 0, idx)
            else:  # fake_label == 0
                # Random permutation from real features only
                idx = random.sample(list(range(0, num_rs)), num_rs)
                idx = torch.LongTensor(idx).to(real_feas.device)
                mix_labels = torch.gather(all_labels, 0, idx)

                idx = idx.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, C, H, W)
                mix_feas = torch.gather(all_feas, 0, idx)

            # Reshape for the encoder network
            mix_feas = mix_feas.reshape(1, num_rs * C, H, W)

            # Forward through encoder
            pred_mix = netE(mix_feas)
            _, _, h, w = pred_mix.shape
            mix_labels = (
                mix_labels.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(1, 1, h, w)
            )

            # Compute loss for this fake
            loss_E_mix = nn.MSELoss()(pred_mix, mix_labels)
            total_loss += loss_E_mix

        # Average loss across all fakes in the batch
        return total_loss / B

    def backward_E_basic(self, netE, real_Rs, fake):
        loss_E_mix_fake = self.insert_fea_loss(real_Rs, fake, netE, fake_label=1)
        loss_E_mix_rel = self.insert_fea_loss(real_Rs, fake, netE, fake_label=0)

        loss_E = (loss_E_mix_fake + loss_E_mix_rel) * 0.5
        loss_E.backward()

        return loss_E

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_E_IHC(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_feas_IHC = self.netD_A(fake_B, "triplesem").detach()

        # Process each batch item separately with its own reference set
        B = fake_B.shape[0]
        total_loss = 0.0
        for i in range(B):
            # Get features for this batch item's reference set [32, C, H, W]
            real_feas_IHC = self.netD_A(self.Rs_IHC[i], "triplesem").detach()
            fake_feat = fake_feas_IHC[i : i + 1]  # [1, C, H, W]

            loss_fake = self.insert_fea_loss(
                real_feas_IHC, fake_feat, self.netE_B, fake_label=1
            )
            loss_real = self.insert_fea_loss(
                real_feas_IHC, fake_feat, self.netE_B, fake_label=0
            )
            total_loss += (loss_fake + loss_real) * 0.5

        self.loss_E_IHC = total_loss / B
        self.loss_E_IHC.backward()

    def backward_E_HE(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_feas_HE = self.netD_B(fake_A, "triplesem").detach()

        # Process each batch item separately with its own reference set
        B = fake_A.shape[0]
        total_loss = 0.0
        for i in range(B):
            # Get features for this batch item's reference set [32, C, H, W]
            real_feas_HE = self.netD_B(self.Rs_HE[i], "triplesem").detach()
            fake_feat = fake_feas_HE[i : i + 1]  # [1, C, H, W]

            loss_fake = self.insert_fea_loss(
                real_feas_HE, fake_feat, self.netE_A, fake_label=1
            )
            loss_real = self.insert_fea_loss(
                real_feas_HE, fake_feat, self.netE_A, fake_label=0
            )
            total_loss += (loss_fake + loss_real) * 0.5

        self.loss_E_HE = total_loss / B
        self.loss_E_HE.backward()

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        beta = self.opt.beta
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        self.loss_A_pos = self.Patho_loss(self.fake_B_label, self.A_label)

        fake_feas_IHC = self.netD_A(self.fake_B, "triplesem").detach()
        fake_feas_HE = self.netD_B(self.fake_A, "triplesem").detach()

        # Process each batch item separately with its own reference set
        B = self.fake_B.shape[0]
        loss_E_IHC_total = 0.0
        loss_E_HE_total = 0.0
        for i in range(B):
            # Get features for this batch item's reference set
            real_feas_IHC = self.netD_A(self.Rs_IHC[i], "triplesem").detach()
            real_feas_HE = self.netD_B(self.Rs_HE[i], "triplesem").detach()

            fake_feat_IHC = fake_feas_IHC[i : i + 1]
            fake_feat_HE = fake_feas_HE[i : i + 1]

            loss_E_IHC_total += self.insert_fea_loss(
                real_feas_IHC, fake_feat_IHC, self.netE_B, fake_label=0
            )
            loss_E_HE_total += self.insert_fea_loss(
                real_feas_HE, fake_feat_HE, self.netE_A, fake_label=0
            )

        self.loss_E_IHC = loss_E_IHC_total / B
        self.loss_E_HE = loss_E_HE_total / B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G_E = (self.loss_E_IHC + self.loss_E_HE) * 1

        self.loss_A_pos = self.loss_A_pos

        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
            + self.loss_G_E
            + self.loss_A_pos
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netE_A, self.netE_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.set_requires_grad([self.netE_A, self.netE_B], False)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # E_A and E_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netE_A, self.netE_B], True)
        self.optimizer_E.zero_grad()
        self.backward_E_HE()
        self.backward_E_IHC()
        self.optimizer_E.step()
