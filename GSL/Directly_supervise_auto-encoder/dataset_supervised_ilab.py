"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import transforms as T
from image_folder import make_dataset
from PIL import Image
from PIL import ImageFile
import random
import fnmatch

ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

class ilab_sup_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_sup_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.dir_Appe = os.path.join(self.root, 'appearance')
        self.Appe_paths = make_dataset(self.dir_Appe)
        self.Appe_size = len(self.Appe_paths)

    def __getitem__(self, index):
        Appe_img_path = self.Appe_paths[index % self.Appe_size]
        Pose_img_path = Appe_img_path.replace('appearance', 'pose').replace('appe', 'pose')
        Comb_img_path = Appe_img_path.replace('appearance', 'combine').replace('appe', 'comb')
        Appe_img = Image.open(Appe_img_path).convert('RGB')
        Pose_img = Image.open(Pose_img_path).convert('RGB')
        Comb_img = Image.open(Comb_img_path).convert('RGB')

        if self.transform is not None:
            Appe = self.transform(Appe_img)
            Pose = self.transform(Pose_img)
            Comb = self.transform(Comb_img)

        return {'Appe': Appe, 'Pose': Pose, 'Comb': Comb}

    def __len__(self):
        return self.Appe_size
class ilab_unsup_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_unsup_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)
        self.C_size = len(self.paths)

    def __getitem__(self, index):
        C_img_path = self.paths[index % self.C_size]
        # C_pose = C_img_path.split('/')[4]
        # C_category = C_img_path.split('/')[5].split('-')[0]
        # C_identity = C_img_path.split('/')[5].split('-')[1]
        C_pose = C_img_path.split('/')[5]
        C_category = C_img_path.split('/')[6].split('-')[0]
        C_identity = C_img_path.split('/')[6].split('-')[1]
        B_root = self.root.replace('vae_unbalance_train', 'vae_unbalance_identity')
        # B has same identity
        B_category = C_category
        B_identity = C_identity
        B_img_root = os.path.join(B_root, B_category, B_identity)
        B_files = os.listdir(B_img_root)
        B_image_index = random.randint(0, len(B_files) - 1)
        B_img_path = os.path.join(B_img_root, B_files[B_image_index])
        # A has same pose
        A_pose = C_pose
        A_img_root = os.path.join(self.root, A_pose)
        A_files = os.listdir(A_img_root)
        A_image_index = random.randint(0, len(A_files) - 1)
        A_img_path = os.path.join(A_img_root, A_files[A_image_index])

        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')

        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)

        return {'A': A, 'B': B, 'C': C}

    def __len__(self):
        return self.C_size
class ilab_threeswap_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_threeswap_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)
        self.C_size = len(self.paths)
        self.id_dict = {}
        self.bg_dict = {}
        self.pose_dict = {}
        self.id_cnt = 0
        self.bg_cnt = 0
        self.pose_cnt = 0
        for roots, dirs, files in os.walk('/home2/ilab2M_pose/train_img_c00_10class'):
            for file in files:
                category = file.split('-')[0]
                id = file.split('-')[1]
                background = file.split('-')[2]
                pose = file.split('-')[3] + file.split('-')[4]
                if id not in self.id_dict:
                    self.id_dict[id] = self.id_cnt
                    self.id_cnt += 1
                if background not in self.bg_dict:
                    self.bg_dict[background] = self.bg_cnt
                    self.bg_cnt += 1
                if pose not in self.pose_dict:
                    self.pose_dict[pose] = self.pose_cnt
                    self.pose_cnt += 1

    def findABD(self, index):
        FOUNDED = False
        while not FOUNDED:
            FOUNDED = True
            C_img_path = self.paths[index % self.C_size]
            '''
            local
            '''
            C_pose_root = C_img_path.split('/')[4]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path.split('/')[5].split('-')[0]
            C_identity = C_img_path.split('/')[5].split('-')[1]
            C_back = C_img_path.split('/')[5].split('-')[2]

            '''
            server
            '''
            # C_pose_root = C_img_path.split('/')[5]
            # C_pose = C_pose_root.replace('_', '-')
            # C_category = C_img_path.split('/')[6].split('-')[0]
            # C_identity = C_img_path.split('/')[6].split('-')[1]
            # C_back = C_img_path.split('/')[6].split('-')[2]

            '''
            server on igpu9
            '''
            # C_pose_root = C_img_path.split('/')[7]
            # C_pose = C_pose_root.replace('_', '-')
            # C_category = C_img_path.split('/')[8].split('-')[0]
            # C_identity = C_img_path.split('/')[8].split('-')[1]
            # C_back = C_img_path.split('/')[8].split('-')[2]

            B_root = self.root.replace('train_img_c00_10class', 'vae_identity_new')
            # B has same identity
            B_category = C_category
            B_identity = C_identity
            B_img_root = os.path.join(B_root, B_category, B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)
            if not C_pose in B_img_name and not C_back in B_img_name:
                B_img_path = os.path.join(B_img_root, B_img_name)
            else:
                # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 1000
                continue  # break the BREAK_ALL

            # A has same pose
            A_img_root = os.path.join(self.root, C_pose_root)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            if not C_identity in A_img_name and not C_back in A_img_name:
                A_img_path = os.path.join(A_img_root, A_img_name)
            else:
                # print('The A image can not have different pose and back with C because the C path is {0}'.format(
                #     C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL

            # D has same back
            # back-cate-pose
            D_root = self.root.replace('train_img_c00_10class', 'vae_back_new')
            D_back = C_back
            # D has same back
            D_img_root_back = os.path.join(D_root, D_back)
            # D must have different identity and diff pose with C
            '''cate '''
            for roots, dirs, files in os.walk(D_img_root_back):
                cates = dirs
                break
            cates.remove(C_category)
            if len(cates) <= 0:  # no other category to choose
                # print('The D image can not have different cate with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL
            selected_D_cate = random.choice(cates)
            D_img_root_cate = os.path.join(D_img_root_back, selected_D_cate)
            '''pose '''
            for roots, dirs, files in os.walk(D_img_root_cate):
                poses = dirs
                break
            # try:
            #     poses.remove(C_pose_root)
            # except:
            #     print(poses, C_pose_root)
            poses.remove(C_pose_root)



            if len(poses) <= 0:  # no other category to choose
                # print('The D image can not have different pose with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 2 if index < self.C_size - 20 else  index - 200
                continue  # break the BREAK_ALL
            selected_D_pose = random.choice(poses)
            D_img_root = os.path.join(D_img_root_cate, selected_D_pose)
            D_files = os.listdir(D_img_root)
            D_image_index = random.randint(0, len(D_files) - 1)
            D_img_path = os.path.join(D_img_root, D_files[D_image_index])
            C_img_name = C_img_path.split('/')[-1]
            D_img_name = D_img_path.split('/')[-1]
            name_list = [A_img_name, B_img_name, C_img_name, D_img_name]

        return A_img_path, B_img_path, C_img_path, D_img_path, name_list
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        A_img_path, B_img_path, C_img_path, D_img_path, name_list= self.findABD(index)



        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')
        D_img = Image.open(D_img_path).convert('RGB')

        labels = {'id':[], 'pose':[], 'bg':[]}
        for name in name_list:
            labels['id'].append(self.id_dict[name.split('-')[1]])
            labels['pose'].append(self.pose_dict[name.split('-')[3] + name.split('-')[4]])
            labels['bg'].append(self.bg_dict[name.split('-')[2]])

        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)
            D = self.transform(D_img)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'labels': labels}

    def __len__(self):
        return self.C_size

class ilab_GZS_synthesis_imgfolder(Dataset):
    '''
    controllable synthesis
    '''
    def __init__(self, root, transform=None):
        super(ilab_GZS_synthesis_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)
        self.C_size = len(self.paths)


    def findpose(self, index):

        C_img_path = self.paths[index % self.C_size]
        '''
        local
        '''
        # C_pose_root = C_img_path.split('/')[4]
        # C_pose = C_pose_root.replace('_', '-')
        # C_category = C_img_path.split('/')[5].split('-')[0]
        # C_identity = C_img_path.split('/')[5].split('-')[1]
        # C_back = C_img_path.split('/')[5].split('-')[2]

        '''
        server
        '''
        C_pose = C_img_path.split('/')[5]


        '''C_pose
        server on igpu9
        '''
        # C_pose_root = C_img_path.split('/')[7]
        # C_pose = C_pose_root.replace('_', '-')
        # C_category = C_img_path.split('/')[8].split('-')[0]
        # C_identity = C_img_path.split('/')[8].split('-')[1]
        # C_back = C_img_path.split('/')[8].split('-')[2]

        pose_root = '/home2/ilab2M_pose/story/GZS_synthesis_provider'
        # B has same identity
        pose_name = C_pose + '.jpg'
        pose_img_path = os.path.join(pose_root, pose_name)



        return pose_img_path, C_img_path
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        pose_img_path, C_img_path = self.findpose(index)



        C_img = Image.open(C_img_path).convert('RGB')
        pose_img = Image.open(pose_img_path).convert('RGB')


        if self.transform is not None:
            C = self.transform(C_img)
            pose = self.transform(pose_img)

        return {'C': C, 'pose': pose, 'C_img_path':C_img_path}

    def __len__(self):
        return self.C_size

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    train = args.train
    crop_size = args.crop_size
    image_size = args.image_size
    # assert image_sizeimage_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset

    elif name.lower() == 'ilab':
        root = '/lab/andy/PycharmProjects/PurePoseGan/stargan/data/train_img_c00_10class'
        if not os.path.exists(root):
            print('No ilab dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = ImageFolder(root, transform)
    elif name.lower() == 'ilab_sup' or name.lower() == 'ilab_unsup':
        if args.use_server == True:
            root = '/home2/andy/ilab2M_pose/vae_unsup'
        else:
            root = '/home2/iLab-2M-Light/sampled/'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_unbalance':
        if args.use_server == True:
            root = '/home2/andy/ilab2M_pose/vae_unbalance_tri'
        else:
            root = '/home2/ilab2M_pose/vae_unbalance_tri'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_unbalance_free':
        if args.use_server == True:
            root = '/home2/andy/ilab2M_pose/vae_unbalance_train'
        else:
            root = '/home2/ilab2M_pose/vae_unbalance_train'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_unbalance_free_largegap':
        if args.use_server == True:
            root = '/media/home2/andy/dataset/vae_unbalance_train_largegap'
        else:
            root = '/home2/ilab2M_pose/vae_unbalance_train'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_threeswap':
        if args.use_server == True:
            if args.which_server == '15':
                root = '/home2/andy/ilab2M_pose/train_img_c00_10class'
            elif args.which_server == '21':
                root = '/home2/andy/ilab2M_pose/train_img_c00_10class'
            elif args.which_server == '9':
                root = '/media/pohsuanh/Data/andy/ilab2M_pose/train_img_c00_10class'
        else:
            root = '/home2/ilab2M_pose/train_img_c00_10class'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_threeswap_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_threeswap_GZS_synthesis':
        if args.use_server == True:
            if args.which_server == '15':
                root = '/home2/andy/ilab2M_pose/train_img_c00_10class'
            elif args.which_server == '21':
                root = '/home2/andy/ilab2M_pose/train_img_c00_10class'
            elif args.which_server == '9':
                root = '/media/pohsuanh/Data/andy/ilab2M_pose/train_img_c00_10class'
        else:
            root = '/home2/ilab2M_pose/train_img_c00_10class'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_GZS_synthesis_imgfolder(root, transform)




    else:
        raise NotImplementedError


    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers)

    # train_data = dset(**train_kwargs)
    # train_loader = DataLoader(train_data,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           num_workers=num_workers,
    #                           pin_memory=True,
    #                           drop_last=True)

    # data_loader = train_loader

    return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
