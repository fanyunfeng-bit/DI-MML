import torch
import torchvision.transforms as transforms
from PIL import Image
import utils.gtransforms as gtransforms
from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import datetime
import os
import pickle

def read_txt(txt_path):
    '''
    读取txt 文件
    :param txt_path:
    :return: txt中每行的数据,结尾用'\n'
    '''

    f = open(txt_path)
    data = f.readlines()
    for index in range(len(data)):
        data[index] = data[index][:-1]
    return data

def ucf101_mean_std(split):
    if split == 'train':
        mean = [106.46409, 101.46796, 93.22408]
        std = [68.28274, 66.40727, 67.58706]
    else:
        mean = [108.639244, 103.3139, 95.21823]
        std = [67.80855, 65.88047, 67.68048]
    return mean, std

def clip_transform_ucf101(split, max_len):
    mean, std = ucf101_mean_std(split)
    if split == 'train':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupRandomCrop(224),
            gtransforms.GroupRandomHorizontalFlip(),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    elif split == 'val':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupCenterCrop(224),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    # Note: RandomCrop (instead of CenterCrop) because
    # We're doing 3 random crops per frame for validation
    elif split == '3crop':
        transform = transforms.Compose([
            # gtransforms.GroupResize(256),
            gtransforms.GroupRandomCrop(224),
            gtransforms.ToTensor(),
            gtransforms.GroupNormalize(mean, std),
            gtransforms.LoopPad(max_len),
        ])

    return transform

class UCF101(Dataset):
    def __init__(self, args, clip_len, mode, mode2, sample_interal=2):
        '''
        动作的label直接从文件夹的名字中提取:a01_s01_e01   a01 的01
        :param data_root: 存放NW-HMDB51 数据集的路径
        :param split_path: path for split txt
        :param clip_len: 每个动作取多少帧用于训练
        :mode2 : rgb or depth
        :param train: 训练还是测试
        :param sample_interal: 数据抽样加快训练
        '''
        super(UCF101, self).__init__()
        
        self.args = args
        self.test_data = []
        self.train_data = []

        self.clip_len = clip_len
        frames_path_len_min = 10000
        self.mode = mode
        self.mode2 = mode2
        
        if self.mode == 'train':
            split_path = '/root/autodl-tmp/ETF-PMR-upload/data/UCF/trainlist01.txt'
        else:
            split_path = '/root/autodl-tmp/ETF-PMR-upload/data/UCF/testlist001.txt'

        txt_name = split_path.split('/')[-1]
        if 'train' in txt_name:
            self.train = True
        elif 'test' in txt_name:
            self.train = False
        else:
            print('error file')

        split_line = read_txt(split_path)
        for line in split_line:
            related_path = line.split(' ')[0]
            related_path = related_path.split('.')[0].split('/')[1]
            label = int(line.split(' ')[1])-1
            #video_path = os.path.join(data_root, related_path)
            rgb_path = os.path.join('/root/autodl-tmp/jpegs_256/', related_path)
            flow_x_path = os.path.join('/root/autodl-tmp/tvl1_flow/u/', related_path)
            flow_y_path = os.path.join('/root/autodl-tmp/tvl1_flow/v/', related_path)
            
            #img_list = os.listdir(flow_x_path)
            img_len = len(glob.glob(flow_x_path+'/*.jpg'))
            # print(img_len,img_len/3)
            #assert np.mod(img_len, 3) == 1
            single_modality_len = img_len
            rgb_list = []
            flow_x_list = []
            flow_y_list = []
            
            for i in range(single_modality_len):
                i = str(i+1)
                i = i.zfill(6)
                rgb_list.append(os.path.join(rgb_path, "frame" + i + ".jpg"))
                flow_x_list.append(os.path.join(flow_x_path, "frame" + i + ".jpg"))
                flow_y_list.append(os.path.join(flow_y_path, "frame" + i + ".jpg"))
            if self.train:
                self.train_data.append(
                    {"frame": rgb_list, "flow_x": flow_x_list, "flow_y": flow_y_list, "label": label})
            else:
                self.test_data.append(
                    {"frame": rgb_list, "flow_x": flow_x_list, "flow_y": flow_y_list, "label": label})

        self.loader = lambda fl: Image.open(fl)

        if self.train:
            self.data = self.train_data
        else:
            self.data = self.test_data

        if self.train:
            self.clip_transform = clip_transform_ucf101('train', clip_len)
        else:
            self.clip_transform = clip_transform_ucf101('val', clip_len)

    def sample_rgb(self, entray):
        imgs = entray['frame']
        if len(imgs) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(imgs) - self.clip_len)
                imgs = imgs[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(imgs) // 2 - self.clip_len // 2
                imgs = imgs[offset:offset + self.clip_len]
            assert len(imgs) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        imgs = [self.loader(img) for img in imgs]
        return imgs

    def sample_flow(self, entray):
        flow_x = entray['flow_x']
        flow_y = entray['flow_y']

        if len(flow_x) > self.clip_len:

            if self.train:  # random sample
                offset = np.random.randint(0, len(flow_x) - self.clip_len)
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(flow_x) // 2 - self.clip_len // 2
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        flow_x_imgs = [self.loader(img) for img in flow_x]
        flow_y_imgs = [self.loader(img) for img in flow_y]

        return flow_x_imgs, flow_y_imgs

    def sample_all(self, entry):
        '''
        用于对多模态数据对进行采样
        :param entry: 包含多模态数据对的dict
        :return:
        '''
        rgb_img = entry['frame']
        flow_x = entry['flow_x']
        flow_y = entry['flow_y']

        if len(flow_x) > self.clip_len:

            if self.train:  # random sample
                np.random.seed(999)
                offset = np.random.randint(0, len(flow_x) - self.clip_len)
                rgb_img = rgb_img[offset:offset + self.clip_len]
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            else:  # center crop
                offset = len(flow_x) // 2 - self.clip_len // 2
                rgb_img = rgb_img[offset:offset + self.clip_len]
                flow_x = flow_x[offset:offset + self.clip_len]
                flow_y = flow_y[offset:offset + self.clip_len]
            assert len(flow_x) == self.clip_len, 'frame selection error!'
        else:
            raise RuntimeError("len(imgs) > self.clip_len")

        rgb_imgs = [self.loader(img) for img in rgb_img]
        flow_x_imgs = [self.loader(img) for img in flow_x]
        flow_y_imgs = [self.loader(img) for img in flow_y]

        return rgb_imgs, flow_x_imgs, flow_y_imgs

    def __getitem__(self, index):
        entry = self.data[index]
        if self.mode2 == 'rgb':
            rgb_frames = self.sample_rgb(entry)
            rgb_frames = self.clip_transform(rgb_frames)  # (T, 3, 224, 224)
            rgb_frames = rgb_frames.permute(1, 0, 2, 3)  # (3, T, 224, 224)
            b = datetime.datetime.now()
            # print((b-c).total_seconds())
            # print(entry['label'])
            instance = {'frames': rgb_frames, 'label': entry['label']}
        elif self.mode2 == 'flow':
            flow_x, flow_y = self.sample_flow(entry)
            # print(len(flow_x),len(flow_y))
            input = {"flow_x": flow_x, "flow_y": flow_y}
            output = self.clip_transform(input)
            flow_x, flow_y = output["flow_x"], output["flow_y"]
            zero_z = torch.zeros_like(flow_x)
            flow_xy = torch.cat((flow_x, flow_y), dim=1)
            flow_xy = flow_xy.permute(1, 0, 2, 3)
            instance = {'flow': flow_xy, 'label': entry['label']}
        elif self.mode2 == 'all':
            rgb_frames, flow_x, flow_y = self.sample_all(entry)
            input = {"rgb": rgb_frames, "flow_x": flow_x, "flow_y": flow_y}
            output = self.clip_transform(input)
            rgb_frames, flow_x, flow_y = output["rgb"], output["flow_x"], output["flow_y"]
            flow_xy = torch.cat((flow_x, flow_y), dim=1)
            flow_xy = flow_xy.permute(1, 0, 2, 3)
            rgb_frames = rgb_frames.permute(1, 0, 2, 3)
            
            np.random.seed(999)
            select_index = np.random.choice(self.clip_len, size=self.args.num_frame, replace=False)
            select_index.sort()
            rgb_frames = rgb_frames[:,select_index]
        
            #instance = {'frames': rgb_frames, 'flow': flow_xy, 'label': entry['label']}
            instance = (flow_xy, rgb_frames, entry['label'])
        else:
            raise RuntimeError("self.mode2=='rgb'")

        return instance

    def __len__(self):
        # print(len(self.data))
        return len(self.data)