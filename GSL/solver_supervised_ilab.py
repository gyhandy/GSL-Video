"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model_supervised_ilab import Generator_fc, Classifier_Latent
from dataset_supervised_ilab import return_data
from PIL import Image
import torch.nn as nn
import functools
import networks
from torchvision import transforms


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    combine_sup_loss=[],
                    combine_unsup_loss=[],
                    images=[],
                    combine_supimages=[],
                    combine_unsupimages=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        # self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        # model params
        self.c_dim = args.c_dim
        self.image_size = args.image_size
        self.g_conv_dim = args.g_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_conv_dim = args.d_conv_dim
        self.d_repeat_num = args.d_repeat_num
        self.norm_layer = get_norm_layer(norm_type=args.norm)
        self.z_pose_dim = args.z_pose_dim
        self.z_unknow_dim = args.z_unknow_dim
        self.z_no_pose_dim = self.z_dim - self.z_pose_dim - self.z_unknow_dim

        self.lambda_combine = args.lambda_combine
        self.lambda_unsup = args.lambda_unsup



        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_sup':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup_unbalance':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup_unbalance_free':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup_threeswap':
            self.nc = 3
            self.decoder_dist = 'gaussian'

        else:
            raise NotImplementedError
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        # self.Autoencoder = Generator(self.nc, self.g_conv_dim, self.g_repeat_num)
        self.Autoencoder = Generator_fc(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        # self.Autoencoder = BetaVAE_ilab(self.z_dim, self.nc)

        self.Autoencoder.to(self.device)
        self.id_classifier = Classifier_Latent(60, 963)
        self.id_classifier.to(self.device)
        self.pose_classifier = Classifier_Latent(20, 6)
        self.pose_classifier.to(self.device)
        self.bg_classifier = Classifier_Latent(20, 417)
        self.bg_classifier.to(self.device)
        self.auto_optim = optim.Adam(list(self.Autoencoder.parameters()) + list(self.id_classifier.parameters()) +
                                     list(self.pose_classifier.parameters()) +
                                     list(self.bg_classifier.parameters()), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        ''' use D '''
        # self.netD = networks.define_D(self.nc, self.d_conv_dim, 'basic',
        #                                 3, 'instance', True, 'normal', 0.02,
        #                                 '0,1')



        # log
        self.log_dir = './checkpoints/' + args.viz_name
        self.model_save_dir = args.model_save_dir



        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_combine_sup = None
        self.win_combine_unsup = None
        # self.win_d_no_pose_losdata_loaders = None
        # self.win_d_pose_loss = None
        # self.win_equal_pose_loss = None
        # self.win_have_pose_loss = None
        # self.win_auto_loss_fake = None
        # self.win_loss_cor_coe = None
        # self.win_d_loss = None

        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)
        self.resume_iters = args.resume_iters

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        # if self.ckpt_name is not None:
        #     self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(resume_iters))
        self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{} (iter {})'".format(self.viz_name, resume_iters))

    def Cor_CoeLoss(self, y_pred, y_target):
        x = y_pred
        y = y_target
        x_var = x - torch.mean(x)
        y_var = y - torch.mean(y)
        r_num = torch.sum(x_var * y_var)
        r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
        r = r_num / r_den

        # return 1 - r  # best are 0
        return 1 - r ** 2  # abslute constrain

    def train(self):
        # self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        id_dict = {}
        bg_dict = {}
        pose_dict = {}
        id_cnt = 0
        bg_cnt = 0
        pose_cnt = 0
        for roots, dirs, files in os.walk('/home2/ilab2M_pose/train_img_c00_10class'):
            for file in files:
                category = file.split('-')[0]
                id = file.split('-')[1]
                background = file.split('-')[2]
                pose = file.split('-')[3] + file.split('-')[4]
                if id not in id_dict:
                    id_dict[id] = id_cnt
                    id_cnt += 1
                if background not in bg_dict:
                    bg_dict[background] = bg_cnt
                    bg_cnt += 1
                if pose not in pose_dict:
                    pose_dict[pose] = pose_cnt
                    pose_cnt += 1
        while not out:


            for sup_package in self.data_loader:
                if self.global_iter == 121:
                    print(121)
                # appe, pose, combine
                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)

                A_z_appe = A_z[:, 0:self.z_no_pose_dim] # 0-700
                A_z_unknow = A_z[:, self.z_no_pose_dim:self.z_no_pose_dim + self.z_unknow_dim] # 700-800
                A_z_pose = A_z[:, self.z_no_pose_dim + self.z_unknow_dim:] #800-1000
                B_z_appe = B_z[:, 0:self.z_no_pose_dim]
                B_z_unknow = B_z[:, self.z_no_pose_dim:self.z_no_pose_dim + self.z_unknow_dim]
                B_z_pose = B_z[:, self.z_no_pose_dim + self.z_unknow_dim:]
                C_z_appe = C_z[:, 0:self.z_no_pose_dim]
                C_z_unknow = C_z[:, self.z_no_pose_dim:self.z_no_pose_dim + self.z_unknow_dim]
                C_z_pose = C_z[:, self.z_no_pose_dim + self.z_unknow_dim:]
                D_z_appe = D_z[:, 0:self.z_no_pose_dim]
                D_z_unknow = D_z[:, self.z_no_pose_dim:self.z_no_pose_dim + self.z_unknow_dim]
                D_z_pose = D_z[:, self.z_no_pose_dim + self.z_unknow_dim:]

                ## 2. combine with strong supervise
                # C A same pose diff id, back
                ApCo_combine_2C = torch.cat((C_z_appe, C_z_unknow), dim=1)
                ApCo_combine_2C = torch.cat((ApCo_combine_2C, A_z_pose), dim=1)
                mid_ApCo = self.Autoencoder.fc_decoder(ApCo_combine_2C)
                mid_ApCo = mid_ApCo.view(ApCo_combine_2C.shape[0], 256, 8, 8)
                ApCo_2C = self.Autoencoder.decoder(mid_ApCo)

                AoCp_combine_2A = torch.cat((A_z_appe, A_z_unknow), dim=1)
                AoCp_combine_2A = torch.cat((AoCp_combine_2A, C_z_pose), dim=1)
                mid_AoCp = self.Autoencoder.fc_decoder(AoCp_combine_2A)
                mid_AoCp = mid_AoCp.view(AoCp_combine_2A.shape[0], 256, 8, 8)
                AoCp_2A = self.Autoencoder.decoder(mid_AoCp)

                # C B same id diff pose, back
                BaCo_combine_2C = torch.cat((B_z_appe, C_z_unknow), dim=1)
                BaCo_combine_2C = torch.cat((BaCo_combine_2C, C_z_pose), dim=1)
                mid_BaCo = self.Autoencoder.fc_decoder(BaCo_combine_2C)
                mid_BaCo = mid_BaCo.view(BaCo_combine_2C.shape[0], 256, 8, 8)
                BaCo_2C = self.Autoencoder.decoder(mid_BaCo)

                BoCa_combine_2B = torch.cat((C_z_appe, B_z_unknow), dim=1)
                BoCa_combine_2B = torch.cat((BoCa_combine_2B, B_z_pose), dim=1)
                mid_BoCa = self.Autoencoder.fc_decoder(BoCa_combine_2B)
                mid_BoCa = mid_BoCa.view(BoCa_combine_2B.shape[0], 256, 8, 8)
                BoCa_2B = self.Autoencoder.decoder(mid_BoCa)

                # C D same background diff id, pose
                DbCo_combine_2C = torch.cat((C_z_appe, D_z_unknow), dim=1)
                DbCo_combine_2C = torch.cat((DbCo_combine_2C, C_z_pose), dim=1)
                mid_DbCo = self.Autoencoder.fc_decoder(DbCo_combine_2C)
                mid_DbCo = mid_DbCo.view(DbCo_combine_2C.shape[0], 256, 8, 8)
                DbCo_2C = self.Autoencoder.decoder(mid_DbCo)

                DoCb_combine_2D = torch.cat((D_z_appe, C_z_unknow), dim=1)
                DoCb_combine_2D = torch.cat((DoCb_combine_2D, D_z_pose), dim=1)
                mid_DoCb = self.Autoencoder.fc_decoder(DoCb_combine_2D)
                mid_DoCb = mid_DoCb.view(DoCb_combine_2D.shape[0], 256, 8, 8)
                DoCb_2D = self.Autoencoder.decoder(mid_DoCb)

                # combine_2C
                ApBaDb_combine_2C = torch.cat((B_z_appe, D_z_unknow), dim=1)
                ApBaDb_combine_2C = torch.cat((ApBaDb_combine_2C, A_z_pose), dim=1)
                mid_ApBaDb = self.Autoencoder.fc_decoder(ApBaDb_combine_2C)
                mid_ApBaDb = mid_ApBaDb.view(ApBaDb_combine_2C.shape[0], 256, 8, 8)
                ApBaDb_2C = self.Autoencoder.decoder(mid_ApBaDb)



                # '''  need unsupervise '''
                AaBpDb_combine_2N = torch.cat((A_z_appe, D_z_unknow), dim=1)
                AaBpDb_combine_2N = torch.cat((AaBpDb_combine_2N, B_z_pose), dim=1)
                mid_AaBpDb = self.Autoencoder.fc_decoder(AaBpDb_combine_2N)
                mid_AaBpDb = mid_AaBpDb.view(AaBpDb_combine_2N.shape[0], 256, 8, 8)
                AaBpDb_2N = self.Autoencoder.decoder(mid_AaBpDb)

                # '''  need unsupervise '''
                # AaBp_combine_2N = torch.cat((A_z_appe, C_z_unknow), dim=1)
                # AaBp_combine_2N = torch.cat((AaBp_combine_2N, B_z_pose), dim=1)
                # mid_AaBp = self.Autoencoder.fc_decoder(AaBp_combine_2N)
                # mid_AaBp = mid_AaBp.view(AaBp_combine_2N.shape[0], 256, 8, 8)
                # AaBp_2N = self.Autoencoder.decoder(mid_AaBp)


                '''
                optimize for autoencoder
                '''

                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss

                # 2. sup_combine_loss
                ApCo_2C_loss = torch.mean(torch.abs(C_img - ApCo_2C))
                AoCp_2A_loss = torch.mean(torch.abs(A_img - AoCp_2A))
                BaCo_2C_loss = torch.mean(torch.abs(C_img - BaCo_2C))
                BoCa_2B_loss = torch.mean(torch.abs(B_img - BoCa_2B))
                DbCo_2C_loss = torch.mean(torch.abs(C_img - DbCo_2C))
                DoCb_2D_loss = torch.mean(torch.abs(D_img - DoCb_2D))
                ApBaDb_2C_loss = torch.mean(torch.abs(C_img - ApBaDb_2C))
                combine_sup_loss = ApCo_2C_loss + AoCp_2A_loss + BaCo_2C_loss + BoCa_2B_loss + DbCo_2C_loss + DoCb_2D_loss + ApBaDb_2C_loss

                # 3. unsup_combine_loss
                _, AaBpDb_z = self.Autoencoder(AaBpDb_2N)
                combine_unsup_loss = torch.mean(torch.abs(A_z_appe - AaBpDb_z[:, 0:self.z_no_pose_dim])) + torch.mean(torch.abs(D_z_unknow - AaBpDb_z[:, self.z_no_pose_dim:self.z_no_pose_dim + self.z_unknow_dim])) + torch.mean(torch.abs(B_z_pose - AaBpDb_z[:, self.z_no_pose_dim + self.z_unknow_dim:]))

                labels = sup_package['labels']
                id_list = [A_z_appe, B_z_appe, C_z_appe, D_z_appe]
                pose_list = [A_z_pose, B_z_pose, C_z_pose, D_z_pose]
                bg_list = [A_z_unknow, B_z_unknow, C_z_unknow, D_z_unknow]
                classification_criterion = nn.CrossEntropyLoss()
                classification_loss = 0
                for i in range(4):
                    classification_loss += classification_criterion(self.id_classifier(id_list[i]),
                                                                    torch.tensor(labels['id'][i],
                                                                                 device=self.device).long())
                    classification_loss += classification_criterion(self.pose_classifier(pose_list[i]),
                                                                    torch.tensor(labels['pose'][i],
                                                                                 device=self.device).long())
                    classification_loss += classification_criterion(self.bg_classifier(bg_list[i]),
                                                                    torch.tensor(labels['bg'][i],
                                                                                 device=self.device).long())

                # whole loss
                vae_unsup_loss = recon_loss + classification_loss#+ self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                #　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n',
                              '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} classification_loss:{:.3f}'.format(
                                  self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data,
                                  classification_loss.data)])
                f.close()
                print(['\n',
                       '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} classification_loss:{:.3f}'.format(
                           self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data,
                           classification_loss.data)])

                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,recon_loss=recon_loss.data,
                                    combine_sup_loss=combine_sup_loss.data, combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoCp_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoCa_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DbCo_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoCb_2D).data)
                        self.viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(ApBaDb_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(AaBpDb_2N).data)
                        self.viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter%self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))


                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def save_sample_img(self, tensor, mode):
        unloader = transforms.ToPILImage()
        dir = os.path.join(self.model_save_dir, self.viz_name, 'sample_img')
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        if mode == 'recon':
            image_ori_A = image[0].squeeze(0)  # remove the fake batch dimension
            image_ori_B = image[1].squeeze(0)
            image_ori_C = image[2].squeeze(0)
            image_ori_D = image[3].squeeze(0)
            image_recon = image[4].squeeze(0)

            image_ori_A = unloader(image_ori_A)
            image_ori_B = unloader(image_ori_B)
            image_ori_C = unloader(image_ori_C)
            image_ori_D = unloader(image_ori_D)
            image_recon = unloader(image_recon)

            image_ori_A.save(os.path.join(dir, '{}-A_img.png'.format(self.global_iter)))
            image_ori_B.save(os.path.join(dir, '{}-B_img.png'.format(self.global_iter)))
            image_ori_C.save(os.path.join(dir, '{}-C_img.png'.format(self.global_iter)))
            image_ori_D.save(os.path.join(dir, '{}-D_img.png'.format(self.global_iter)))
            image_recon.save(os.path.join(dir, '{}-A_img_recon.png'.format(self.global_iter)))
        elif mode == 'combine_sup':
            image_AoCp_2A = image[0].squeeze(0)  # remove the fake batch dimension
            image_BoCa_2B = image[1].squeeze(0)
            image_DbCo_2C = image[2].squeeze(0)
            image_DoCb_2D = image[3].squeeze(0)

            image_AoCp_2A = unloader(image_AoCp_2A)
            image_BoCa_2B = unloader(image_BoCa_2B)
            image_DbCo_2C = unloader(image_DbCo_2C)
            image_DoCb_2D = unloader(image_DoCb_2D)

            image_AoCp_2A.save(os.path.join(dir, '{}-AoCp_2A.png'.format(self.global_iter)))
            image_BoCa_2B.save(os.path.join(dir, '{}-BoCa_2B.png'.format(self.global_iter)))
            image_DbCo_2C.save(os.path.join(dir, '{}-DbCo_2C.png'.format(self.global_iter)))
            image_DoCb_2D.save(os.path.join(dir, '{}-DoCb_2D.png'.format(self.global_iter)))
        elif mode == 'combine_unsup':
            image_ApBaDb_2C = image[0].squeeze(0)  # remove the fake batch dimension
            image_AaBpDb_2N = image[1].squeeze(0)

            image_ApBaDb_2C = unloader(image_ApBaDb_2C)
            image_AaBpDb_2N = unloader(image_AaBpDb_2N)

            image_ApBaDb_2C.save(os.path.join(dir, '{}-ApBaDb_2C.png'.format(self.global_iter)))
            image_AaBpDb_2N.save(os.path.join(dir, '{}-AaBpDb_2N.png'.format(self.global_iter)))
    def viz_reconstruction(self):
        # self.net_mode(train=False)
        x_A = self.gather.data['images'][0][:100]
        x_A = make_grid(x_A, normalize=True)
        x_B = self.gather.data['images'][1][:100]
        x_B = make_grid(x_B, normalize=True)
        x_C = self.gather.data['images'][2][:100]
        x_C = make_grid(x_C, normalize=True)
        x_D = self.gather.data['images'][3][:100]
        x_D = make_grid(x_D, normalize=True)
        x_A_recon = self.gather.data['images'][4][:100]
        x_A_recon = make_grid(x_A_recon, normalize=True)
        images = torch.stack([x_A, x_B, x_C, x_D, x_A_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'recon')
        # self.net_mode(train=True)
    def viz_combine_recon(self):
        # self.net_mode(train=False)
        AoCp_2A = self.gather.data['combine_supimages'][0][:100]
        AoCp_2A = make_grid(AoCp_2A, normalize=True)
        BoCa_2B = self.gather.data['combine_supimages'][1][:100]
        BoCa_2B = make_grid(BoCa_2B, normalize=True)
        DbCo_2C = self.gather.data['combine_supimages'][2][:100]
        DbCo_2C = make_grid(DbCo_2C, normalize=True)
        DoCb_2D = self.gather.data['combine_supimages'][3][:100]
        DoCb_2D = make_grid(DoCb_2D, normalize=True)
        images = torch.stack([AoCp_2A, BoCa_2B, DbCo_2C, DoCb_2D], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'combine_sup')
    def viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        ApBaDb_2C = self.gather.data['combine_unsupimages'][0][:100]
        ApBaDb_2C = make_grid(ApBaDb_2C, normalize=True)
        AaBpDb_2N = self.gather.data['combine_unsupimages'][1][:100]
        AaBpDb_2N = make_grid(AaBpDb_2N, normalize=True)
        images = torch.stack([ApBaDb_2C, AaBpDb_2N], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_unsupimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'combine_unsup')


    def viz_combine(self, x):
        # self.net_mode(train=False)

        decoder = self.Autoencoder.decoder
        encoder = self.Autoencoder.encoder
        z = encoder(x)
        z_appe = z[:, 0:250, :, :]
        z_pose = z[:, 250:, :, :]
        z_rearrange_combine = torch.cat((z_appe[:-1], z_pose[1:]), dim=1)
        x_rearrange_combine = decoder(z_rearrange_combine)
        x_rearrange_combine = F.sigmoid(x_rearrange_combine).data

        x_show = make_grid(x[:-1].data, normalize=True)
        x_rearrange_combine_show = make_grid(x_rearrange_combine, normalize=True)
        images = torch.stack([x_show, x_rearrange_combine_show], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_combine',
                        opts=dict(title=str(self.global_iter)), nrow=10)




        # samples = []
        # for i in range(10): # every pair need visualize
        #     x_appe = x[i].unsqueeze(0)  # provide appearance
        #
        #     z_appe = z[i, 0:250, :, :].unsqueeze(0)  # provide appearance
        #     x_pose = x[i+1].unsqueeze(0)  # provide pose
        #
        #     z_pose = z[i+1, 250:, :, :].unsqueeze(0)  # provide pose
        #     z_combine = torch.cat((z_appe, z_pose), 1)
        #     x_combine = decoder(z_combine)
        #     x_combine = F.sigmoid(x_combine).data
        #     samples.append(x_appe)
        #     samples.append(x_combine)
        #     samples.append(x_pose)
        #     samples = torch.cat(samples, dim=0).cpu()
        #     title = 'combine(iter:{})'.format(self.global_iter)
        #     if self.viz_on:
        #         self.viz.images(samples, env=self.viz_name+'combine',
        #                         opts=dict(title=title))

    def viz_lines(self):
        # self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        combine_sup_loss = torch.stack(self.gather.data['combine_sup_loss']).cpu()
        combine_unsup_loss = torch.stack(self.gather.data['combine_unsup_loss']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_combine_sup is None:
            self.win_combine_sup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_sup_loss',))
        else:
            self.win_combine_sup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        win=self.win_combine_sup,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_sup_loss',))

        if self.win_combine_unsup is None:
            self.win_combine_unsup = self.viz.line(
                                        X=iters,
                                        Y=combine_unsup_loss,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_unsup_loss',))
        else:
            self.win_combine_unsup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        win=self.win_combine_unsup,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_unsup_loss',))


    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
