import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        if isinstance(img_group, dict) and len(img_group) == 2:  # flowx+flowy
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            flow_x_new = [self.worker(img) for img in flow_x]
            flow_y_new = [self.worker(img) for img in flow_y]
            img_group = {"flow_x": flow_x_new, "flow_y": flow_y_new}
            return img_group
        elif isinstance(img_group, dict) and len(img_group) == 3:  # flowx+y,rgb
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            rgb = img_group['rgb']
            flow_x_new = [self.worker(img) for img in flow_x]
            flow_y_new = [self.worker(img) for img in flow_y]
            rgb_new = [self.worker(img) for img in rgb]
            img_group = {"rgb": rgb_new, "flow_x": flow_x_new, "flow_y": flow_y_new}
            return img_group
        else:
            return [self.worker(img) for img in img_group]


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        if isinstance(img_group, dict) and len(img_group) == 2:  # flowx+flowy

            flow_x_out = list()
            flow_y_out = list()
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']

            w, h = flow_x[0].size
            th, tw = self.size
            # print(w, h, th, tw)
            if w < tw or h <th:
                for img in flow_x:
                    img=img.resize(self.size)
                    flow_x_out.append(img)
                for img in flow_y:
                    img=img.resize(self.size)
                    flow_y_out.append(img)
            else:

                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)

                for img in flow_x:
                    assert (img.size[0] == w and img.size[1] == h)
                    if w == tw and h == th:
                        flow_x_out.append(img)
                    else:
                        flow_x_out.append(img.crop((x1, y1, x1 + tw, y1 + th)))

                for img in flow_y:
                    assert (img.size[0] == w and img.size[1] == h)
                    if w == tw and h == th:
                        flow_y_out.append(img)
                    else:
                        flow_y_out.append(img.crop((x1, y1, x1 + tw, y1 + th)))

            img_group = {"flow_x": flow_x_out, "flow_y": flow_y_out}
            return img_group
        elif isinstance(img_group, dict) and len(img_group) == 3:  # flowx+y,rgb
            flow_x_out = list()
            flow_y_out = list()
            rgb_out = list()
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            rgb = img_group['rgb']

            w, h = flow_x[0].size
            th, tw = self.size
            if w < tw:
                flow_x = flow_x.resize((self.size, self.size))
                flow_y = flow_y.resize((self.size, self.size))

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            for img in flow_x:
                assert (img.size[0] == w and img.size[1] == h)
                if w == tw and h == th:
                    flow_x_out.append(img)
                else:
                    flow_x_out.append(img.crop((x1, y1, x1 + tw, y1 + th)))

            for img in flow_y:
                assert (img.size[0] == w and img.size[1] == h)
                if w == tw and h == th:
                    flow_y_out.append(img)
                else:
                    flow_y_out.append(img.crop((x1, y1, x1 + tw, y1 + th)))

            for img in rgb:
                #assert (img.size[0] == w and img.size[1] == h)
                if w == tw and h == th:
                    rgb_out.append(img)
                else:
                    rgb_out.append(img.crop((x1, y1, x1 + tw, y1 + th)))

            img_group = {"rgb": rgb_out, "flow_x": flow_x_out, "flow_y": flow_y_out}
            return img_group
        else:
            w, h = img_group[0].size
            th, tw = self.size
            out_images = list()

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            if w < tw or h <th:
                for img in img_group:
                    img=img.resize(self.size)
                    out_images.append(img)
            else:
                for img in img_group:
                    assert (img.size[0] == w and img.size[1] == h)
                    if w == tw and h == th:
                        out_images.append(img)
                    else:
                        out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

            return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        if isinstance(img_group, dict) and len(img_group) == 2:  # flowx+flowy
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            flow_x_new = [self.worker(img) for img in flow_x]
            flow_y_new = [self.worker(img) for img in flow_y]
            img_group = {"flow_x": flow_x_new, "flow_y": flow_y_new}
            return img_group
        elif isinstance(img_group, dict) and len(img_group) == 3:  # flowx+y,rgb
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            rgb = img_group['rgb']
            flow_x_new = [self.worker(img) for img in flow_x]
            flow_y_new = [self.worker(img) for img in flow_y]
            rgb_new = [self.worker(img) for img in rgb]
            img_group = {"rgb": rgb_new, "flow_x": flow_x_new, "flow_y": flow_y_new}
            return img_group
        else:
            return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:

            if isinstance(img_group, dict) and len(img_group) == 2:  # flowx+flowy
                flow_x = img_group['flow_x']
                flow_y = img_group['flow_y']
                flow_x_new = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in flow_x]
                img_group = {"flow_x": flow_x_new, "flow_y": flow_y}
                return img_group
            if isinstance(img_group, dict) and len(img_group) == 3:  # flowx+flowy
                flow_x = img_group['flow_x']
                flow_y = img_group['flow_y']
                rgb = img_group['rgb']

                flow_x_new = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in flow_x]
                rgb_new = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in rgb]
                img_group = {"rgb": rgb_new, "flow_x": flow_x_new, "flow_y": flow_y}
                return img_group
            else:
                img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
                return img_group
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):  # (T, 3, 224, 224)

        if isinstance(tensor, dict) and len(tensor) == 2:  # flowx+flowy
            flow_x = tensor['flow_x']
            flow_y = tensor['flow_y']

            # print(len(flow_x))

            for b in range(flow_x.size(0)):
                for t, m, s in zip(flow_x[b], [127.5], [1.0]):
                    t.sub_(m).div_(s)
            for b in range(flow_y.size(0)):
                for t, m, s in zip(flow_y[b], [127.5], [1.0]):
                    t.sub_(m).div_(s)

            img_group = {"flow_x": flow_x, "flow_y": flow_y}
            return img_group

        if isinstance(tensor, dict) and len(tensor) == 3:  # flowx+flowy
            flow_x = tensor['flow_x']
            flow_y = tensor['flow_y']
            rgb = tensor['rgb']

            for b in range(flow_x.size(0)):
                for t, m, s in zip(flow_x[b], [127.5], [1.0]):
                    t.sub_(m).div_(s)
            for b in range(flow_y.size(0)):
                for t, m, s in zip(flow_y[b], [127.5], [1.0]):
                    t.sub_(m).div_(s)
            for b in range(rgb.size(0)):
                for t, m, s in zip(rgb[b], self.mean, self.std):
                    t.sub_(m).div_(s)
            img_group = {"rgb": rgb, "flow_x": flow_x, "flow_y": flow_y}
            return img_group
        else:
            for b in range(tensor.size(0)):
                for t, m, s in zip(tensor[b], self.mean, self.std):
                    t.sub_(m).div_(s)
            return tensor


class LoopPad(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):

        if isinstance(tensor, dict) and len(tensor) == 2:
            flow_x = tensor['flow_x']
            length = flow_x.size(0)

            if length == self.max_len:
                return tensor
            else:
                flow_y = tensor['flow_y']
                n_pad = self.max_len - length
                pad = [flow_y] * (n_pad // length)
                if n_pad % length > 0:
                    pad += [flow_y[0:n_pad % length]]

                flow_y = torch.cat([flow_y] + pad, 0)
                flow_x = torch.cat([flow_x] + pad, 0)
                tensor = {"flow_x": flow_x, "flow_y": flow_y}
                return tensor

        if isinstance(tensor, dict) and len(tensor) == 3:
            flow_x = tensor['flow_x']
            length = flow_x.size(0)

            if length == self.max_len:
                return tensor
            else:
                flow_y = tensor['flow_y']
                n_pad = self.max_len - length
                pad = [flow_y] * (n_pad // length)
                if n_pad % length > 0:
                    pad += [flow_y[0:n_pad % length]]
                flow_y = torch.cat([flow_y] + pad, 0)
                flow_x = torch.cat([flow_x] + pad, 0)

                rgb = tensor['rgb']
                n_pad = self.max_len - length
                pad = [rgb] * (n_pad // length)
                if n_pad % length > 0:
                    pad += [rgb[0:n_pad % length]]
                rgb = torch.cat([rgb] + pad, 0)

                tensor = {"rgb": rgb, "flow_x": flow_x, "flow_y": flow_y}
                return tensor
        else:
            # repeat the clip as many times as is necessary
            length = tensor.size(0)

            if length == self.max_len:
                return tensor
            n_pad = self.max_len - length
            pad = [tensor] * (n_pad // length)
            if n_pad % length > 0:
                pad += [tensor[0:n_pad % length]]

            tensor = torch.cat([tensor] + pad, 0)
            return tensor


# NOTE: Returns [0-255] rather than torchvision's [0-1]
class ToTensor(object):
    def __init__(self):
        self.worker = lambda x: F.to_tensor(x) * 255

    def __call__(self, img_group):
        # print(len(img_group), type(img_group))
        if isinstance(img_group, dict) and len(img_group) == 2:  # flowx+flowy
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            flow_x_new = [self.worker(img) for img in flow_x]
            flow_y_new = [self.worker(img) for img in flow_y]

            flow_x_new = torch.stack(flow_x_new, 0)
            flow_y_new = torch.stack(flow_y_new, 0)
            img_group = {"flow_x": flow_x_new, "flow_y": flow_y_new}
            return img_group
        elif isinstance(img_group, dict) and len(img_group) == 3:  # flowx+y,rgb
            flow_x = img_group['flow_x']
            flow_y = img_group['flow_y']
            rgb = img_group['rgb']
            flow_x_new = [self.worker(img) for img in flow_x]
            flow_y_new = [self.worker(img) for img in flow_y]
            rgb_new = [self.worker(img) for img in rgb]

            flow_x_new = torch.stack(flow_x_new, 0)
            flow_y_new = torch.stack(flow_y_new, 0)
            rgb_new = torch.stack(rgb_new, 0)

            img_group = {"rgb": rgb_new, "flow_x": flow_x_new, "flow_y": flow_y_new}
            return img_group
        else:

            img_group = [self.worker(img) for img in img_group]
            return torch.stack(img_group, 0)