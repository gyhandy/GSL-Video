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
excluded_id = ['boat', 'tank']


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

included_ep = ['happy', 'neutral', 'surprised']


class ilab_threeswap_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_threeswap_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.idf_dict = {}
        self.exp_dict = {}
        self.posef_dict = {}
        self.idf_cnt = 0
        self.exp_cnt = 0
        self.posef_cnt = 0
        for roots, dirs, files in os.walk('/home2/RaFD/train/data/'):
            for file in files:
                idf = file.split('_')[0]
                expression = file.split('_')[2].split('.')[0]
                posef = file.split('_')[1]
                if idf not in self.idf_dict:
                    self.idf_dict[idf] = self.idf_cnt
                    self.idf_cnt += 1
                if expression not in self.exp_dict:
                    self.exp_dict[expression] = self.exp_cnt
                    self.exp_cnt += 1
                if posef not in self.posef_dict:
                    self.posef_dict[posef] = self.posef_cnt
                    self.posef_cnt += 1
        print(root)
        self.paths = make_dataset(self.root)
        file = open('debug.txt','w')
        file.write(str(self.paths))
        file.close()
        self.C_size = len(self.paths)-1
        print(self.C_size)

    def findABD(self, index):
        FOUNDED = False
        while not FOUNDED:

            C_img_path = self.paths[index % self.C_size]
            # print(C_img_path.split('/'))
            #C_img_path = '/home2/RaFD/sep/data/27_090_neutral.jpg'
            #C_img_path ='/home2/RaFD/sep/data/45_045_surprised.jpg'

            # print(C_img_path,  index)
            files = os.listdir(self.root)
            E_img_name = random.choice(files)

            E_img_path = os.path.join(self.root, E_img_name)
            # print(E_img_path)
            '''
            local
            '''


            C_pose = C_img_path.split('/')[5].split('_')[1]
            C_identity = C_img_path.split('/')[5].split('_')[0]
            C_back = C_img_path.split('/')[5].split('_')[2].split('.')[0]



            '''
            server
            '''
            # print(C_img_path.split('/'))
            # C_pose = C_img_path.split('/')[7].split('_')[1]
            # C_identity = C_img_path.split('/')[7].split('_')[0]
            # C_back = C_img_path.split('/')[7].split('_')[2].split('.')[0]

            # check C and E

            # if C_identity in excluded_id or E_identity in excluded_id:
            #     continue


            B_root = self.root.replace('data', 'img_id')
            # B has same identity
            # print(B_root)
            B_identity = C_identity

            B_img_root = os.path.join(B_root,  B_identity)
            # print(B_img_root)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)

            b_founded = 0
            cnt = 0
            while not b_founded:
                cnt += 1
                # print(C_img_path,'finding b',cnt)
                if not C_pose == B_img_name.split('_')[1] and not C_back == B_img_name.split('_')[2]:
                    B_img_path = os.path.join(B_img_root, B_img_name)
                    b_founded =1
                else:
                    # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                    B_img_name = random.choice(B_files)
                    if cnt >= 100:
                        break
            if b_founded == 0:
                # print('b not found')
                index += 1
                continue
            # print(B_img_path)
            # A has same pose
            A_img_root = os.path.join(self.root.replace('data', 'img_pz'), C_pose)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            a_founded = 0
            cnt = 0
            while not a_founded:
                cnt += 1
                # print(C_img_path,'finding a')
                # print(C_identity, C_back, A_img_name)
                if C_identity != A_img_name.split('_')[0] and C_back != A_img_name.split('_')[0]:
                    A_img_path = os.path.join(A_img_root, A_img_name)
                    a_founded = 1
                else:
                    # print('The A image can not have different pose and back with C because the C path is {0}'.format(
                    #     C_img_path))
                    A_img_name = random.choice(A_files)
                    if cnt >= 100:
                        break
            if a_founded == 0:
                index += 1
                continue
            # print(A_img_path)

            # D has same back
            # back-cate-pose
            D_root = self.root.replace('data', 'img_ep')
            D_back = C_back
            # D has same back
            D_img_root = os.path.join(D_root, D_back)
            D_files = os.listdir(D_img_root)
            D_img_name = random.choice(D_files)
            d_founded = 0
            cnt = 0
            while not d_founded:
                cnt += 1
                # print(C_img_path,'finding d')
                if not C_identity == D_img_name.split('_')[0] and not C_pose == D_img_name.split('_')[1]:
                    D_img_path = os.path.join(D_img_root, D_img_name)
                    d_founded = 1
                else:
                    # print('D must have different identity and diff pose with C)
                    D_img_name = random.choice(D_files)
                    if cnt >= 100:
                        break
            if d_founded == 0:
                index += 1
                continue
            # D must have different identity and diff pose with C
            '''cate '''
            # print(D_img_path)
            # print('---------------------------')
            FOUNDED = 1
            # check D
            # if D_identity in excluded_id:
            #     continue
            C_img_name = C_img_path.split('/')[-1]
            name_list = [A_img_name, B_img_name, C_img_name, D_img_name, E_img_name]

        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, name_list


    def find_test(self, index):
        FOUNDED = False
        while not FOUNDED:

            C_img_path = self.paths[index % self.C_size]
            # print('____________________________________',C_img_path.split('/'))
            #C_img_path = '/home2/RaFD/sep/data/27_090_neutral.jpg'
            #C_img_path ='/home2/RaFD/sep/data/45_045_surprised.jpg'

            # print(C_img_path,  index)
            files = os.listdir(self.root)
            E_img_name = random.choice(files)

            E_img_path = os.path.join(self.root, E_img_name)
            # print(E_img_path)
            '''
            local
            '''

            # print(C_img_path.split('/'))
            C_pose = C_img_path.split('/')[5].split('_')[1]
            C_identity = C_img_path.split('/')[5].split('_')[0]
            C_back = C_img_path.split('/')[5].split('_')[2].split('.')[0]



            '''
            server
            '''
            # C_pose = C_img_path.split('/')[6].split('_')[1]
            # C_identity = C_img_path.split('/')[6].split('_')[0]
            # C_back = C_img_path.split('/')[6].split('_')[2].split('.')[0]

            # check C and E

            # if C_identity in excluded_id or E_identity in excluded_id:
            #     continue


            B_root = self.root.replace('data', 'img_id')
            B_root = B_root.replace('train', 'test')
            # B has same identity
            B_identity = C_identity
            B_img_root = os.path.join(B_root,  B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)

            b_founded = 0
            cnt = 0
            while not b_founded:
                cnt += 1
                # print(C_img_path,'finding b',cnt)
                if not C_pose == B_img_name.split('_')[1] and not C_back == B_img_name.split('_')[2]:
                    B_img_path = os.path.join(B_img_root, B_img_name)
                    b_founded =1
                else:
                    # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                    B_img_name = random.choice(B_files)
                    if cnt >= 100:
                        break
            if b_founded == 0:
                # print('b not found')
                index += 1
                continue
            # print(B_img_path)
            # A has same pose
            A_img_root = os.path.join(self.root.replace('data', 'img_pz'), C_pose)
            A_img_root = A_img_root.replace('train', 'test')
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            a_founded = 0
            cnt = 0
            while not a_founded:
                cnt += 1
                # print(C_img_path,'finding a')
                # print(C_identity, C_back, A_img_name)
                if C_identity != A_img_name.split('_')[0] and C_back != A_img_name.split('_')[0]:
                    A_img_path = os.path.join(A_img_root, A_img_name)
                    a_founded = 1
                else:
                    # print('The A image can not have different pose and back with C because the C path is {0}'.format(
                    #     C_img_path))
                    A_img_name = random.choice(A_files)
                    if cnt >= 100:
                        break
            if a_founded == 0:
                index += 1
                continue
            # print(A_img_path)

            # D has same back
            # back-cate-pose
            D_root = self.root.replace('data', 'img_ep')
            D_root = D_root.replace('train', 'test')
            D_back = C_back
            # D has same back
            D_img_root = os.path.join(D_root, D_back)
            D_files = os.listdir(D_img_root)
            D_img_name = random.choice(D_files)
            d_founded = 0
            cnt = 0
            while not d_founded:
                cnt += 1
                # print(C_img_path,'finding d')
                if not C_identity == D_img_name.split('_')[0] and not C_pose == D_img_name.split('_')[1]:
                    D_img_path = os.path.join(D_img_root, D_img_name)
                    d_founded = 1
                else:
                    # print('D must have different identity and diff pose with C)
                    D_img_name = random.choice(D_files)
                    if cnt >= 100:
                        break
            if d_founded == 0:
                index += 1
                continue
            # D must have different identity and diff pose with C
            '''cate '''
            # print(D_img_path)
            # print('---------------------------')
            FOUNDED = 1
            # check D
            # if D_identity in excluded_id:
            #     continue
        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path


    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, name_list = self.findABD(index)
        # A_img_path, B_img_path, C_img_path, D_img_path, E_img_path = self.find_test(index)

        labels = {'id':[], 'exp':[], 'pose':[]}
        for name in name_list:
            labels['id'].append(self.idf_dict[name.split('_')[0]])
            labels['pose'].append(self.posef_dict[name.split('_')[1]])
            labels['exp'].append(self.exp_dict[name.split('_')[2].split('.')[0]])

        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')
        D_img = Image.open(D_img_path).convert('RGB')
        E_img = Image.open(E_img_path).convert('RGB')


        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)
            D = self.transform(D_img)
            E = self.transform(E_img)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'E':E, 'labels':labels}

    def __len__(self):
        return self.C_size-1

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
    # assert image_size == 64, 'currently only image size of 64 is supported'

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
            if args.which_server == '9':
                root = '/media/pohsuanh/Data/andy/train/data'
            elif args.which_server == '21':
                root = '/home2/andy/RaFD/train/data/'
        else:
            root = '/home2/RaFD/train/data/'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize((image_size, image_size)))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_threeswap_imgfolder(root, transform)

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
