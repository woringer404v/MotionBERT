import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader
from pyskl.datasets import BaseDataset
from pyskl.datasets.pipelines import UniformSampleFrames, PoseDecode, PoseCompact, Resize, RandomResizedCrop, Flip, GeneratePoseTarget, FormatShape, Collect, ToTensor, Compose
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

keypoint_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    #dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    #dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
    
def get_action_names(file_path = "data/action/ntu_actions.txt"):
    f = open(file_path, "r")
    s = f.read()
    actions = s.split('\n')
    action_names = []
    for a in actions:
        action_names.append(a.split('.')[1][1:])
    return action_names

def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y
    
def random_move(data_numpy,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # C,T,V,M -> M,T,V,C
    return data_numpy    

def human_tracking(x):
    M, T = x.shape[:2]
    if M==1:
        return x
    else:
        diff0 = np.sum(np.linalg.norm(x[0,1:] - x[0,:-1], axis=-1), axis=-1)        # (T-1, V, C) -> (T-1)
        diff1 = np.sum(np.linalg.norm(x[0,1:] - x[1,:-1], axis=-1), axis=-1)
        x_new = np.zeros(x.shape)
        sel = np.cumsum(diff0 > diff1) % 2
        sel = sel[:,None,None]
        x_new[0][0] = x[0][0]
        x_new[1][0] = x[1][0]
        x_new[0,1:] = x[1,1:] * sel + x[0,1:] * (1-sel)
        x_new[1,1:] = x[0,1:] * sel + x[1,1:] * (1-sel)
        return x_new

class ActionDataset(Dataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=True):   # data_split: train/test etc.
        np.random.seed(0)
        dataset = read_pkl(data_path)
        if check_split:
            assert data_split in dataset['split'].keys()
            self.split = dataset['split'][data_split]
        annotations = dataset['annotations']
        self.random_move = random_move
        self.is_train = "train" in data_split or (check_split==False)
        if "oneshot" in data_split:
            self.is_train = False
        self.scale_range = scale_range
        motions = []
        labels = []
        pipeline = Compose(keypoint_pipeline)
        for sample in annotations:
            if check_split and (not sample['frame_dir'] in self.split):
                continue
            heatmap = pipeline(sample)['imgs']
            #print(heatmap)
            # resample_id = resample(ori_len=sample['total_frames'], target_len=n_frames, replay=False, randomness=self.is_train)
            # #print(resample_id)
            # motion_cam = make_cam(x=sample['keypoint'], img_shape=sample['img_shape'])
            # motion_cam = human_tracking(motion_cam)
            # motion_cam = coco2h36m(motion_cam)
            # motion_conf = sample['keypoint_score'][..., None]
            # # motion_conf = motion_conf.squeeze(axis=0)
            # # motion_conf = np.expand_dims(motion_conf, axis=-1)
            # # Debugging prints
            # # print("resample_id:", resample_id)
            # # print("motion_cam shape:", motion_cam.shape)
            # # print("motion_conf shape:", motion_conf.shape)
            # motion = np.concatenate((motion_cam[:,resample_id], motion_conf[:,resample_id]), axis=-1)
            # print("motion shape:", motion.shape)
            # # if motion.shape[0]==1:                                  # Single person, make a fake zero person
            # #     fake = np.zeros(motion.shape)
            # #     motion = np.concatenate((motion, fake), axis=0)
            motions.append(heatmap)#.astype(np.float32)) 
            labels.append(sample['label'])
            # print("motion_camddd shape:", len(motions[0][0][0]))
            # print("motion_confddd shape:", (labels[0].shape))
        self.motions = np.array(motions)
        # with open('output1.txt', 'w') as file:
        #     file.write(str(motions))

        self.labels = np.array(labels)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)

    def __getitem__(self, index):
        raise NotImplementedError 

class NTURGBD(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1]):
        super(NTURGBD, self).__init__(data_path, data_split, n_frames, random_move, scale_range)
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        # if self.random_move:
        #     motion = random_move(motion)
        # if self.scale_range:
        #     result = crop_scale(motion, scale_range=self.scale_range)
        # else:
        result = motion
        return result, label#.astype(np.float32), label
    
class NTURGBD1Shot(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=False):
        super(NTURGBD1Shot, self).__init__(data_path, data_split, n_frames, random_move, scale_range, check_split)
        oneshot_classes = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114]
        new_classes = set(range(120)) - set(oneshot_classes)
        old2new = {}
        for i, cid in enumerate(new_classes):
            old2new[cid] = i
        filtered = [not (x in oneshot_classes) for x in self.labels]
        self.motions = self.motions[filtered]
        filtered_labels = self.labels[filtered]
        self.labels = [old2new[x] for x in filtered_labels]
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label
    

# class MotionBertForPySKL(BaseDataset):
#     def __init__(self, ann_file, data_split, pipeline, **kwargs):
#         # Initialize motionbert NTURGBD dataset
#         self.dataset = NTURGBD(data_path=ann_file, data_split=data_split)
#         super().__init__(ann_file, pipeline, **kwargs)
    
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         motion, label = self.dataset[idx]
#         results = {'imgs': motion, 'label': label, 'ann_info': {'img_shape': (motion.shape[2], motion.shape[3])}}
#         return self.pipeline(results)

class NTURGBDPySKL(NTURGBD):
    def __init__(self, data_path, resize_scale_1=(-1, 64), resize_scale_2=(56, 56), crop_area_range=(0.56, 1.0), input_format='NCTHW_Heatmap', keys=['imgs', 'label'], meta_keys=[]):
        dataset = read_pkl(data_path)
        self.annotations = dataset['annotations']
        self.uniform_sampler = UniformSampleFrames(clip_len=48)
        self.pose_decoder = PoseDecode()
        self.pose_compactor = PoseCompact()
        self.resizer_1 = Resize(resize_scale_1, keep_ratio=True, interpolation='bilinear')
        self.random_resized_cropper = RandomResizedCrop(crop_area_range)
        self.resizer_2 = Resize(resize_scale_2, keep_ratio=False, interpolation='bilinear')
        self.pose_generator = GeneratePoseTarget(with_kp=True, with_limb=False)
        self.format_shaper = FormatShape(input_format)
        self.collector = Collect(keys, meta_keys)
        self.totensor = ToTensor(keys)

    def __getitem__(self, idx):
        #motion, label = self.motions[idx], self.labels[idx]  # (M, T, J, C)

        # Process motion data
        processed_motion = self.annotations
        print(processed_motion)
        # print(processed_motion)
        # with open('output.txt', 'w') as file:
        #     file.write(str(processed_motion))
        # 1. UniformSampleFrames
        #results = {'total_frames': processed_motion.shape[1], 'keypoint': processed_motion} 
        results = self.uniform_sampler(processed_motion)

        # 2. PoseDecode 
        if 'keypoint' in results or 'keypoint_score' in results:
            results = self.pose_decoder(results)
            processed_motion = results['keypoint']

        # 3. PoseCompact
        results = self.pose_compactor(results)
        processed_motion = results['keypoint']

        # 4. Resize
        results = self.resizer(results)
        processed_motion = results['keypoint']

        # 5. RandomResizedCrop
        results = self.random_resized_cropper(results)
        processed_motion = results['keypoint']

        # 6. GeneratePoseTarget
        results = self.pose_generator.gen_an_aug(results)

        # 7. FormatShape
        results = self.format_shaper(results)
        if 'imgs' in results:
            processed_motion = results['imgs']

        # 8. Collect
        results = self.collector(results)
        processed_motion = results['imgs']#.get('imgs', None) 

        # 9. ToTensor
        results = self.totensor(results)
        processed_motion = results['imgs']#.get('imgs', None) 

        # Return the processed data
        return processed_motion

