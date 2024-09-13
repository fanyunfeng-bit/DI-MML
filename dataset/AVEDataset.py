import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random


class AVEDataset(Dataset):

    def __init__(self, args, mode='train', class_imbalanced=False):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = './data/'
        # class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = '../AVE_Dataset'
        self.audio_feature_path = '../AVE_Dataset/Audio-1004-SE'

        self.train_txt = os.path.join(self.data_root, args.dataset + '/trainSet.txt')
        self.test_txt = os.path.join(self.data_root, args.dataset + '/testSet.txt')
        self.val_txt = os.path.join(self.data_root, args.dataset + '/valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        if class_imbalanced and mode == 'train':
            with open(self.train_txt, encoding='UTF-8-sig') as f:
                files = f.readlines()
                subset_class = [[] for _ in range(len(class_dict))]  # instances for each class
                class_sample_num = [0 for _ in range(len(class_dict))]
                for item in files:
                    item = item.split('&')
                    subset_class[class_dict[item[0]]].append(item[1]+'&'+item[0])
                    class_sample_num[class_dict[item[0]]] += 1

            # max_num = class_sample_num[-1]
            # print(class_sample_num)
            num_perclass = []
            props = [0.2, 0.1, 0.3, 0.9, 0.7, 0.9, 0.7, 0.9, 0.2, 0.1, 0.4, 0.5, 0.1, 0.6, 0.6, 0.5, 0.8, 0.9, 0.2, 0.0, 0.0, 0.8, 0.9, 0.1, 0.0, 0.8, 0.4, 0.9]
            for ii in range(len(props)):
                # prop = np.random.randint(0, 10, 1) / 10
                prop = props[ii]
                num_perclass.append(int((1-prop)*class_sample_num[ii]))

            imbalanced_dataset = []
            print(num_perclass)
            # [120, 127, 95, 14, 45, 14, 43, 15, 110, 117, 87, 56, 73, 32, 60, 71, 25, 14, 64, 67, 57, 25, 4, 46, 144, 17, 74, 12]

            for ii in range(len(class_sample_num)):
                random.shuffle(subset_class[ii])
                imbalanced_dataset += subset_class[ii][:num_perclass[ii]]

            random.shuffle(imbalanced_dataset)
            for item in imbalanced_dataset:
                audio_path = os.path.join(self.audio_feature_path, item.split('&')[0] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps), item.split('&')[0])
                # print(item.split('&')[0], item.split('&')[1])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item.split('&')[1]])
                else:
                    continue
        else:
            with open(txt_file, 'r') as f2:
                files = f2.readlines()
                for item in files:
                    item = item.split('&')
                    audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                    visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(self.args.fps), item[1])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        if audio_path not in self.audio:
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[0]])
                    else:
                        continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        np.random.seed(999)
        select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))

        for i,n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[n])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label