import torch
import torch.nn as nn
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def distance_loss(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist_sum = dist.sum(1)
    _, index = dist_sum.sort(descending='descend')
    return index.long()


def calculate_prototype(args, model, dataloader, device, ratio=1.0, a_proto=None, v_proto=None):
    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    audio_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    visual_prototypes = torch.zeros(n_classes, args.embed_dim).to(device)
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        for step, (spec, image, label) in enumerate(dataloader):
            if (step+1) / len(dataloader) > ratio:
                break
            spec = spec.to(device)  # B x 257 x 1004
            image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
            label = label.to(device)  # B

            # TODO: make it simpler and easier to extend
            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            for c, l in enumerate(label):
                l = l.long()
                count_class[l] += 1
                audio_prototypes[l, :] += a[c, :]
                visual_prototypes[l, :] += v[c, :]

            # sample_count += 1
            # if args.dataset == 'AVE':
            #     pass
            # else:
            #     if sample_count >= all_num // 10:
            #         break
    for c in range(audio_prototypes.shape[0]):
        audio_prototypes[c, :] /= count_class[c]
        visual_prototypes[c, :] /= count_class[c]

    # if epoch <= 0:
    #     audio_prototypes = audio_prototypes
    #     visual_prototypes = visual_prototypes
    # else:
    #     audio_prototypes = (1 - args.momentum_coef) * audio_prototypes + args.momentum_coef * a_proto
    #     visual_prototypes = (1 - args.momentum_coef) * visual_prototypes + args.momentum_coef * v_proto
    return audio_prototypes, visual_prototypes
