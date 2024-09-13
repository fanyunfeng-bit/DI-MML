import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.AVEDataset import AVEDataset
from dataset.CramedDataset import CramedDataset
from dataset.UCFDataset import UCF101
from dataset.ModelNet40 import ModelNet40
from models.basic_model import AVClassifier, AClassifier, VClassifier, VVClassifier, FClassifier, FVClassifier
from models.fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='CE', type=str, help='CE, CL, CEwCL, combine_modality')
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE, UCF')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')
    parser.add_argument('--load_path', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')
    parser.add_argument('--load_path_other', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')

    parser.add_argument('--random_seed', default=0, type=int)
    
    parser.add_argument('--temperature', default=1, type=float, help='loss temperature')

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--var', default=0.1, type=float)

    return parser.parse_args()


def train_combine_classifier_epoch(args, epoch, audio_encoder, visual_encoder, classifier,classifier1,classifier2, device,
                                   dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    softmax2 = nn.Softmax(dim=0)
    
    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'UCF':
        n_classes = 101
    elif args.dataset == 'ModelNet':
        n_classes = 40
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    classifier.train()
    audio_encoder.eval()
    visual_encoder.eval()
    print("Start training classifier ... ")

    _loss = 0
    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        B = label.shape[0]

        optimizer.zero_grad()
        with torch.no_grad():
            if args.dataset == 'UCF' or args.dataset == 'ModelNet':
                a_features = audio_encoder(spec.float())
                (_, C, H, W) = a_features.size()
                a_features = a_features.view(B, -1, C, H, W)
                a_features = a_features.permute(0, 2, 1, 3, 4)
                a_features = F.adaptive_avg_pool3d(a_features, 1)
                a_features = torch.flatten(a_features, 1)
            else:
                a_features = audio_encoder(spec.unsqueeze(1).float())
                a_features = F.adaptive_avg_pool2d(a_features, 1)
                a_features = torch.flatten(a_features, 1)

            v_features = visual_encoder(image.float())
            (_, C, H, W) = v_features.size()
            v_features = v_features.view(B, -1, C, H, W)
            v_features = v_features.permute(0, 2, 1, 3, 4)
            v_features = F.adaptive_avg_pool3d(v_features, 1)
            v_features = torch.flatten(v_features, 1)
        
        out1 = classifier1(a_features)
        out2 = classifier2(v_features)
        _, _, out3 = classifier(a_features, v_features)
        
        loss = criterion(out3, label)
        loss.backward()
        optimizer.step()

        _loss += loss.item()
    scheduler.step()
    return _loss / len(dataloader)


def valid_combine_classifier(args, audio_encoder, visual_encoder, classifier,classifier1,classifier2, device, dataloader):
    audio_encoder.eval()
    visual_encoder.eval()
    classifier.eval()
    classifier1.eval()
    classifier2.eval()

    softmax = nn.Softmax(dim=1)
    softmax2 = nn.Softmax(dim=0)

    if args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'UCF':
        n_classes = 101
    elif args.dataset == 'ModelNet':
        n_classes = 40
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc2 = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            B = label.shape[0]
            
            if args.dataset == 'UCF' or args.dataset == 'ModelNet':
                audio_features = audio_encoder(spec.float())
                (_, C, H, W) = audio_features.size()
                audio_features = audio_features.view(B, -1, C, H, W)
                audio_features = audio_features.permute(0, 2, 1, 3, 4)
                audio_features = F.adaptive_avg_pool3d(audio_features, 1)
                audio_features = torch.flatten(audio_features, 1)
            else:
                audio_features = audio_encoder(spec.unsqueeze(1).float())
                audio_features = F.adaptive_avg_pool2d(audio_features, 1)
                audio_features = torch.flatten(audio_features, 1)

            visual_features = visual_encoder(image.float())
            (_, C, H, W) = visual_features.size()
            visual_features = visual_features.view(B, -1, C, H, W)
            visual_features = visual_features.permute(0, 2, 1, 3, 4)
            visual_features = F.adaptive_avg_pool3d(visual_features, 1)
            visual_features = torch.flatten(visual_features, 1)

            out1 = classifier1(audio_features)
            out2 = classifier2(visual_features)
            _, _, out3 = classifier(audio_features, visual_features)
            
            prediction1 = softmax(out1)
            prediction2 = softmax(out2)
            prediction3 = softmax(out3)
            w = torch.cat([prediction1.max(1)[0].unsqueeze(0),prediction2.max(1)[0].unsqueeze(0),prediction3.max(1)[0].unsqueeze(0)])
            w = softmax2(w/args.temperature)
            
            out = out1*(w[0].unsqueeze(1).repeat(1,n_classes))+out2*(w[1].unsqueeze(1).repeat(1,n_classes))+out3*(w[2].unsqueeze(1).repeat(1,n_classes))

            prediction = softmax(out)
            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                ma2 = np.argmax(prediction3[i].cpu().data.numpy())
                num[label[i]] += 1.0  # what is label[i]
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == ma2:
                    acc2[label[i]] += 1.0
        print('acc2:',sum(acc2) / sum(num))
    return sum(acc) / sum(num)


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.method == 'CE' or args.method == 'CE_Proto':
        if args.dataset == 'UCF':
            audio_net = FClassifier(args)
            visual_net = VClassifier(args)
            model = FVClassifier(args)
        elif args.dataset == 'ModelNet':
            audio_net = VClassifier(args)
            visual_net = VClassifier(args)
            model = VVClassifier(args)
        else:
            audio_net = AClassifier(args)
            visual_net = VClassifier(args)
            model = AVClassifier(args)
    else:
        raise ValueError('Incorrect method!')

    audio_net.apply(weight_init)
    audio_net.to(device)
    visual_net.apply(weight_init)
    visual_net.to(device)
    model.apply(weight_init)
    model.to(device)

    if args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    elif args.dataset == 'ModelNet':
        train_dataset = ModelNet40(args, mode='train')
        test_dataset = ModelNet40(args, mode='test')
    elif args.dataset == 'UCF':
        train_dataset = UCF101(args, mode='train', clip_len=10, mode2='all')
        test_dataset = UCF101(args, mode='test', clip_len=10, mode2='all')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8,
                                 shuffle=False, pin_memory=False)

    if args.train:
        trainloss_file = args.logs_path + '/CFT-stage2/' + args.fusion_method + '-' + 'tmp-' + str(args.temperature) + '-' + args.load_path.split('/')[2] + '-' + args.load_path.split('/')[3].split('.')[0] + '-var' + str(args.var) + '.txt'
        if not os.path.exists(args.logs_path + '/CFT-stage2'):
            os.makedirs(args.logs_path + '/CFT-stage2')

        save_path = args.ckpt_path + '/CFT-stage2/' + args.fusion_method + '-'  + 'tmp-' + str(args.temperature) + '-' + args.load_path.split('/')[2] + '-' + args.load_path.split('/')[3].split('.')[0] + '-var' + str(args.var)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        load_path1 = args.load_path   # AClassifier
        load_path2 = args.load_path_other  # VClassifier
        load_dict1 = torch.load(load_path1)
        load_dict2 = torch.load(load_path2)
        state_dict1 = load_dict1['model']
        state_dict2 = load_dict2['model']
        audio_net.load_state_dict(state_dict1)
        visual_net.load_state_dict(state_dict2)

        audio_encoder = audio_net.net
        visual_encoder = visual_net.net
        classifier = model.fusion_module
        classifier1 = audio_net.classifier
        classifier2 = visual_net.classifier
         
        optimizer = optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9,
                              weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

        best_acc = 0
        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))
            
            batch_loss = train_combine_classifier_epoch(args, epoch, audio_encoder, visual_encoder, classifier,classifier1,classifier2, device,
                                                  train_dataloader, optimizer, scheduler)
            acc = valid_combine_classifier(args, audio_encoder, visual_encoder, classifier,classifier1,classifier2, device, test_dataloader)
            print('epoch: ', epoch, 'loss: ', batch_loss, 'acc: ', acc)

            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(acc) +
                              "\n")
            f_trainloss.flush()

            if acc > best_acc:
                best_acc = float(acc)
                print('Saving model....')
                torch.save(
                    {
                        'classifier': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },
                    os.path.join(save_path, 'best.pt'.format(epoch))
                )
                print('Saved model!!!')

        f_trainloss.close()
        
        print(best_acc)

if __name__ == '__main__':
    main()
