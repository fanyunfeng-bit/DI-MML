import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.CramedDataset import CramedDataset
from dataset.AVEDataset import AVEDataset
from dataset.UCFDataset import UCF101
from dataset.ModelNet40 import ModelNet40
from models.basic_model import AVClassifier, AClassifier, VClassifier, FClassifier, GrayClassifier, ColoredClassifier
from utils.utils import setup_seed, weight_init
from loss.contrast_loss import SupConLoss


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE, UCF')

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--temperature', default=1, type=float, help='loss temperature')
    parser.add_argument('--weight', default=1, type=float, help='loss weight')

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument('--var', default=0.1, type=float)

    return parser.parse_args()


def train_epoch(args, epoch, model1, model2, classifier, device,
                dataloader, optimizer, scheduler,
                temperature, weight):
    criterion = nn.CrossEntropyLoss()
    criterion2 = SupConLoss(temperature)

    model1.train()
    model2.train()
    classifier.train()
    print("Start training ... ")

    _loss_a = 0
    _loss_v = 0
    _loss_c = 0
    _loss = 0
    loss_c = torch.tensor(0)
    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        B = label.shape[0]
        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.dataset == 'UCF' or args.dataset == 'ModelNet':
            a, out1 = model1(spec.float(), B)
        else:
            a, out1 = model1(spec.unsqueeze(1).float())
        v, out2 = model2(image.float(), B)

        fusion = torch.cat((a, v), 1)
        out = classifier(fusion)

        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        _loss += loss.item()
    scheduler.step()
    return _loss / len(dataloader)


def valid(args, model1, model2, classifier, device, dataloader):
    softmax = nn.Softmax(dim=1)

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
        model1.eval()
        model2.eval()
        classifier.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            B = label.shape[0]

            if args.dataset == 'UCF' or args.dataset == 'ModelNet':
                a, out1 = model1(spec.float(), B)
            else:
                a, out1 = model1(spec.unsqueeze(1).float())
            v, out2 = model2(image.float(), B)
            fusion = torch.cat((a, v), 1)
            out = classifier(fusion)
            prediction = softmax(out)
            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0  # what is label[i]
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0

    return sum(acc) / sum(num)


def get_indices(args, model1, model2, device, n_sample, dataloader):
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
        model1.eval()
        model2.eval()
        h_audio_features = torch.zeros([n_classes, 512]).to('cuda')
        h_image_features = torch.zeros([n_classes, 512]).to('cuda')
        total_audio_features = torch.zeros([0, 512]).to('cuda')
        total_image_features = torch.zeros([0, 512]).to('cuda')
        labels = torch.zeros([0]).to('cuda')
        cnts = torch.zeros([n_classes]).to('cuda')
        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
            image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
            label = label.to(device)  # B

            B = label.shape[0]
            if args.dataset == 'UCF' or args.dataset == 'ModelNet':
                a, out1 = model1(spec.float(), B)
            else:
                a, out1 = model1(spec.unsqueeze(1).float())
            v, out2 = model2(image.float(), B)

            h_audio_features[label.item()] += a[0]
            h_image_features[label.item()] += v[0]
            cnts[label.item()] += 1
            total_audio_features = torch.cat((total_audio_features, a), dim=0)
            total_image_features = torch.cat((total_image_features, v), dim=0)
            labels = torch.cat((labels, label), dim=0)

    a_h = (h_audio_features.T / cnts).T
    v_h = (h_image_features.T / cnts).T

    dist_a = torch.abs((total_audio_features.unsqueeze(1).repeat(1, n_classes, 1) - a_h))
    dist_v = torch.abs((total_image_features.unsqueeze(1).repeat(1, n_classes, 1) - v_h))

    argmin_a = (torch.argmin(dist_a, dim=1) == labels.unsqueeze(1).repeat(1, 512)).float()
    argmin_v = (torch.argmin(dist_v, dim=1) == labels.unsqueeze(1).repeat(1, 512)).float()

    top_audio_indices = torch.nonzero((torch.mean(argmin_a, dim=0) > torch.mean(argmin_a)))[:, 0]
    top_image_indices = torch.nonzero((torch.mean(argmin_v, dim=0) > torch.mean(argmin_v)))[:, 0]

    audio_only_indices = list(set(top_audio_indices.tolist()) - set(top_image_indices.tolist()))
    image_only_indices = list(set(top_image_indices.tolist()) - set(top_audio_indices.tolist()))
    print(len(audio_only_indices), len(image_only_indices))

    return audio_only_indices, image_only_indices


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.dataset == 'UCF':
        model1 = FClassifier(args)
    elif args.dataset == 'ModelNet':
        model1 = VClassifier(args)
    else:
        model1 = AClassifier(args)
    model2 = VClassifier(args)

    model1.to(device)
    model2.to(device)

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

    classifier = nn.Linear(args.embed_dim*2, n_classes)
    classifier.to(device)

    parameters = list(model1.parameters()) + list(model2.parameters()) + list(classifier.parameters())
    optimizer = optim.SGD(parameters, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    elif args.dataset == 'UCF':
        train_dataset = UCF101(args, mode='train', clip_len=10, mode2='all')
        test_dataset = UCF101(args, mode='test', clip_len=10, mode2='all')
    elif args.dataset == 'ModelNet':
        train_dataset = ModelNet40(args, mode='train')
        test_dataset = ModelNet40(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8,
                                 shuffle=False, pin_memory=False)

    n_sample = 256
    indice_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=False)

    if args.train:

        trainloss_file = args.logs_path + '/Joint-training' + '/train_loss-' + args.dataset + '-bsz' + \
                          str(args.batch_size) + '-lr' + str(args.learning_rate) + '-align' + str(
            args.weight) + '-var' + str(args.var) + '.txt'

        if not os.path.exists(args.logs_path + '/Joint-training'):
            os.makedirs(args.logs_path + '/Joint-training')

        save_path = args.ckpt_path + '/Joint-training' + '/model-' + args.dataset + '-bsz' + \
                     str(args.batch_size) + '-lr' + str(args.learning_rate) + '-align' + str(
            args.weight) + '-var' + str(args.var)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件

        f_trainloss = open(trainloss_file, 'a')


        best_acc = 0.0
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss = train_epoch(args, epoch, model1, model2, classifier,
                                    device, train_dataloader, optimizer,
                                    scheduler, args.temperature, args.weight)
            acc = valid(args, model1, model2, classifier, device, test_dataloader)
            print('epoch: ', epoch, 'acc: ', acc,
                  'loss: ', batch_loss)

            f_trainloss.write(str(epoch) +
                               "\t" + str(batch_loss) +
                               "\t" + str(acc) +
                               "\n")
            f_trainloss.flush()


            if acc > best_acc or (epoch + 1) % 10 == 0:
                if acc > best_acc:
                    best_acc = float(acc)

                # save model parameter
                print('Saving model....')
                torch.save(
                    {
                        'model1': model1.state_dict(),
                        'model2': model2.state_dict(),
                        'classifier': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },
                    os.path.join(save_path, 'epoch.pt'.format(epoch))
                )
                print('Saved model!!!')

        f_trainloss.close()


if __name__ == '__main__':
    main()