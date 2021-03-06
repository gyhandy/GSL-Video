import numpy as np
import os
from PIL import Image
import matplotlib
import scipy.misc
import random

import shutil


# def findBC(roots, A_path):
#     A_pose = roots.split('/')[-1]
#     A_category = A_path.split('/')[-1].split('-')[0]
#     A_instance = A_path.split('/')[-1].split('-')[1]
#
#     pose_info = ['c00_r01', 'c00_r02', 'c00_r03', 'c00_r04', 'c00_r06', 'c00_r07']
#     pose_info.remove(A_pose)
#     pose_index = random.randint(0, 4)
#     selected_pose = pose_info[pose_index]
#     B_root = os.path.join(Raw_Data_root, selected_pose)
#     B_files = os.listdir(B_root)
#     image_index = random.randint(0, len(B_files) - 1)
#     B_path = os.path.join(B_root, B_files[image_index])
#     B_category = B_path.split('/')[-1].split('-')[0]
#     B_instance = B_path.split('/')[-1].split('-')[1]
#     # take the pose of A, instance of B
#     C_path = A_path.replace(A_category, B_category).replace(A_instance, B_instance)
#     exist = os.path.exists(C_path)
#     return exist, B_path, C_path

'''
identity
'''
# Raw_Data_root = '/home2/ilab2M_pose/train_img_c00_10class'
# Raw_Data_root =  '/home2/ilab2M_pose/train_unbtest_img_c00_10class'
Raw_Data_root = '/home2/ilab2M_pose/train_val_test_img_c00_10class'
Target_root ='/home2/ilab2M_pose/vae_trainvaltest_identity_new/'
# Raw_Data_root = '/home2/ilab2M_pose/vae_unbalance_train_largegap'
# Target_root ='/home2/ilab2M_pose/vae_unbalance_identity_largegap/'
pose_info = ['c00_r01', 'c00_r02', 'c00_r03', 'c00_r04', 'c00_r06', 'c00_r07']
for roots, dirs, files in os.walk(Raw_Data_root):
    for i, file in enumerate(files):
        category = file.split('-')[0]
        identity = file.split('-')[1]
        A_path = os.path.join(roots, file)
        if not os.path.exists(os.path.join(Target_root, category, identity)):
            os.makedirs(os.path.join(Target_root, category, identity))
        shutil.copy(os.path.join(roots, file), os.path.join(Target_root, category, identity, file))

'''
background
'''

# Raw_Data_root = '/home2/ilab2M_pose/train_img_c00_10class'
# Target_root ='/home2/ilab2M_pose/vae_back_new/'
# # Raw_Data_root = '/home2/ilab2M_pose/vae_unbalance_train_largegap'
# # Target_root = '/home2/ilab2M_pose/vae_unbalance_identity_largegap/'
# pose_info = ['c00_r01', 'c00_r02', 'c00_r03', 'c00_r04', 'c00_r06', 'c00_r07']
# for roots, dirs, files in os.walk(Raw_Data_root):
#     for i, file in enumerate(files):
#         category = file.split('-')[0]
#         pose = file.split('-')[3] + '_' + file.split('-')[4]
#         back = file.split('-')[2]
#         A_path = os.path.join(roots, file)
#         if not os.path.exists(os.path.join(Target_root, back, category, pose)):
#             os.makedirs(os.path.join(Target_root, back, category, pose))
#         shutil.copy(os.path.join(roots, file), os.path.join(Target_root, back, category, pose, file))





        # if 'IP' in file:
        #     mr_3d = md.read_image(file_path)  # (x,y,z)
        #     mr_3d_np = mr_3d.to_numpy() # (z,y,x)
        #     for i in range(mr_3d_np.shape[0]): # z
        #
        #         '''
        #         get part of data
        #         '''
        #         # the approximately leg part in mri is range (0-60)
        #         # the approximately pelvicum part in mri is range (70-230)
        #         # the approximately lib part in mri is range (200-360)
        #         # if i >= 70 and i <= 230:
        #         '''
        #         whole body
        #         '''
        #         slice0 = np.expand_dims(mr_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
        #         '''
        #         resize the mr
        #         '''
        #         # resize[1, 384, 549] to [1, 384, 548] for net_G
        #         slice0 = slice0[:, :, :-1]
        #         # resize[1, 384, 548] to [1, 512, 512]
        #         slice0 = slice0[:, :, 18:530]
        #         slice0 = np.pad(slice0, ((0, 0), (64, 64), (0, 0)), 'constant', constant_values=0)
        #         '''
        #         normalize 0-255
        #         '''
        #         mrIP_intensity_min = np.float32(0.0)
        #         mrIP_intensity_max = np.float32(400.0)
        #
        #         #  cut off the image
        #         slice0[slice0 > mrIP_intensity_max] = mrIP_intensity_max
        #         slice0[slice0 < mrIP_intensity_min] = mrIP_intensity_min
        #         slice0 = (slice0 - mrIP_intensity_min) / (mrIP_intensity_max - mrIP_intensity_min) * 255
        #
        #         # nct_ref.from_numpy(slice0)  # put slice data into ref
        #
        #         # im = Image.fromarray(slice0)
        #         target_path = target_path_mr
        #         target_filename = 'mr_' + roots.split('/')[-1] + '_'+ str(i) + IMAGE_TYPE
        #         if not os.path.exists(target_path + '/' + target_filename):
        #             scipy.misc.imsave(target_path + '/' + target_filename, slice0[0])
        #             # im.save(target_path + '/' + target_filename)
        #
        #
        #
        # elif 'nfct' in file:
        # # if 'nfCT' in file:
        #     nct_3d = md.read_image(file_path)  # (x,y,z)
        #     nct_3d_np = nct_3d.to_numpy()  # (z,y,x)
        #     for i in range(nct_3d_np.shape[0]):  # z
        #         '''
        #         get part of data
        #         '''
        #         # the approximately leg part in nfct is range (0-60)
        #         # the approximately pelvicum part in nfct is range (40-200)
        #         # the approximately lib part in nfct is range (200-360)
        #         # if i >= 40 and i <= 200:
        #         '''
        #         whole body
        #         '''
        #         slice0 = np.expand_dims(nct_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
        #         if slice0.max() > 0:  # some of the image are totally black
        #             '''
        #               normalize 0-255
        #             '''
        #             nctIP_intensity_min = np.float32(-1000.0)
        #             nctIP_intensity_max = np.float32(1400.0)
        #             #  cut off the image
        #             slice0[slice0 > nctIP_intensity_max] = nctIP_intensity_max
        #             slice0[slice0 < nctIP_intensity_min] = nctIP_intensity_min
        #             slice0 = (slice0 - nctIP_intensity_min) / (nctIP_intensity_max - nctIP_intensity_min) * 255
        #             # nct_ref.from_numpy(slice0)  # put slice data into ref
        #             target_path = target_path_nct
        #             target_filename = 'nct_' + roots.split('/')[-1] + '_' + str(i) + IMAGE_TYPE
        #             if not os.path.exists(target_path + '/' + target_filename):
        #                 scipy.misc.imsave(target_path + '/' + target_filename, slice0[0])
        #             # md.write_image(nct_ref, target_path + '/' + target_filename)  #
        #
        # # get mask of mr
        # elif 'mr_mask' in file:
        #     mr_mask_3d = md.read_image(file_path)  # (x,y,z)
        #     mr_mask_3d_np = mr_mask_3d.to_numpy() # (z,y,x)
        #     for i in range(mr_mask_3d_np.shape[0]): # z
        #
        #         '''
        #         get part of data
        #         '''
        #         # the approximately leg part in mri is range (0-60)
        #         # the approximately pelvicum part in mri is range (70-230)
        #         # the approximately lib part in mri is range (200-360)
        #         # if i >= 70 and i <= 230:
        #         '''
        #         whole body
        #         '''
        #         slice0 = np.expand_dims(mr_mask_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
        #
        #         '''
        #         resize the mr
        #         '''
        #         # resize[1, 384, 549] to [1, 384, 548] for net_G
        #         slice0 = slice0[:, :, :-1]
        #         # resize[1, 384, 548] to [1, 512, 512]
        #         slice0 = slice0[:, :, 18:530]
        #         slice0 = np.pad(slice0, ((0, 0), (64, 64), (0, 0)), 'constant', constant_values=0)
        #
        #         # scipy.misc.imsave(target_path_mr_mask + '/' + target_filename, slice0[0])
        #         # '''
        #         # normalize 0-255
        #         # '''
        #         # mrIP_intensity_min = np.float32(0.0)
        #         # mrIP_intensity_max = np.float32(400.0)
        #         #
        #         # #  cut off the image
        #         # slice0[slice0 > mrIP_intensity_max] = mrIP_intensity_max
        #         # slice0[slice0 < mrIP_intensity_min] = mrIP_intensity_min
        #         # slice0 = (slice0 - mrIP_intensity_min) / (mrIP_intensity_max - mrIP_intensity_min) * 255
        #         #
        #         # # nct_ref.from_numpy(slice0)  # put slice data into ref
        #         #
        #         # # im = Image.fromarray(slice0)
        #         target_path = target_path_mr_mask
        #         target_filename = 'mrmask_' + roots.split('/')[-1] + '_'+ str(i) + IMAGE_TYPE
        #         if not os.path.exists(target_path + '/' + target_filename):
        #             scipy.misc.imsave(target_path + '/' + target_filename, slice0[0])
        #             # im.save(target_path + '/' + target_filename)

















































# import numpy as np
# import md
# import os
#
# Data_root_3d = '/data0/geyunhao/CT_MR/'
# Data_root_2d = '/home/geyunhao/Mapping/Mapping/pytorch-CycleGAN-and-pix2pix-master/datasets/MR2CT/'
# DATA_NAME = 'train'
# IMAGE_TYPE = '.mhd'
# '''
#     Turn the 3D slice to 2D slice and satisfy the structure of cycleGAnN
# '''
# # choose a templete from original MR & CT
# if os.path.exists(Data_root_3d + 'mr_ref.mhd') == False:
#     mr_3d_samp = md.read_image(Data_root_3d + '0001/MR.nii.gz') #  (x,y,z)
#     # mr_3d_samp_np = mr_3d_samp.to_numpy()# (z,y,x)
#     # mr_3d_samp.from_numpy(mr_3d_samp_np) # turn the np 2 image, the (z,y,x)auto 2 (x,y,z)
#     mr_slice = md.image3d_tools.center_crop(mr_3d_samp, [104, 70, 213], [2.4, 2.4, 2.4], [208, 140, 1])
#     md.write_image(mr_slice, Data_root_3d + 'mr_ref.mhd')
#
# if os.path.exists(Data_root_3d + 'nct_ref.mhd') == False:
#     nct_3d_samp = md.read_image(Data_root_3d + '0001/nfCT.nii.gz') #  (x,y,z)
#     nct_slice = md.image3d_tools.center_crop(nct_3d_samp, [104, 70, 213], [2.4, 2.4, 2.4], [208, 140, 1])
#     md.write_image(nct_slice, Data_root_3d + 'nct_ref.mhd')
#
# # get the 2d slice of MR
# mr_ref = md.read_image(Data_root_3d + 'mr_ref.mhd')
# nct_ref = md.read_image(Data_root_3d + 'nct_ref.mhd')
#
# target_path_mr = Data_root_2d + DATA_NAME + 'A'
# target_path_nct = Data_root_2d + DATA_NAME + 'B'
# if not os.path.exists(target_path_mr):
#     os.makedirs(target_path_mr)
# if not os.path.exists(target_path_nct):
#     os.makedirs(target_path_nct)
#
#
# for roots, dirs, files in os.walk(Data_root_3d + DATA_NAME):
#     for file in files:
#         file_path = os.path.join(roots, file)
#         if 'MR' in file:
#             mr_3d = md.read_image(file_path)  # (x,y,z)
#             mr_3d_np = mr_3d.to_numpy() # (z,y,x)
#             for i in range(mr_3d_np.shape[0]): # z
#                 slice0 = np.expand_dims(mr_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
#                 mr_ref.from_numpy(slice0)  # put slice data into ref
#                 target_path = target_path_mr
#                 target_filename = 'mr_' + roots.split('/')[-1] + '_'+ str(i) + IMAGE_TYPE
#                 md.write_image(mr_ref, target_path + '/' + target_filename)  #
#
#         elif 'nfCT' in file:
#             nct_3d = md.read_image(file_path)  # (x,y,z)
#             nct_3d_np = nct_3d.to_numpy()  # (z,y,x)
#             for i in range(nct_3d_np.shape[0]):  # z
#                 slice0 = np.expand_dims(nct_3d_np[i, :, :], 0)  # get slice data without decay (1,y,x)
#                 nct_ref.from_numpy(slice0)  # put slice data into ref
#                 target_path = target_path_nct
#                 target_filename = 'nct_' + roots.split('/')[-1] + '_' + str(i) + IMAGE_TYPE
#                 md.write_image(mr_ref, target_path + '/' + target_filename)  #







# ipython code
# mr_3d_samp = md.read_image(Data_root_3d + '0001/MR.nii.gz') #  (x,y,z)
# mr_ref = md.read_image(Data_root_2d + 'mr_ref.mhd')
# mr_3d_samp_np = mr_3d_samp.to_numpy()
# slice0 = np.expand_dims(mr_3d_samp_np[0, :, :], 0) # get slice data without decay
# mr_ref.from_numpy(slice0)# put slice data into ref
# md.write_image(mr_ref, Data_root_2d + 'single-slice.mhd') #
# ex = md.read_image(Data_root_2d + 'single-slice.mhd')











# nct_3d_samp = md.read_image('/data0/geyunhao/CT_MR/0001/nfCT.nii.gz')
# ct_3d_samp = md.read_image('/data0/geyunhao/CT_MR/0001/CT.nii.gz')
