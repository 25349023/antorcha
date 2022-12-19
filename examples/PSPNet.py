import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T
from torchvision.transforms import functional as TF

from antorcha.data.loaders import PreprocessedDataLoader
from antorcha.toy_models.segmentations import TinyPSPNet
from antorcha.train import trainer
from antorcha.utils.plotting import show_images


def to_gpu(x, y):
    return x.to('cuda'), y.to('cuda')


class SegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, lbl = self.dataset[item]
        if self.transform is not None:
            img, lbl = self.transform((img, lbl))

        lbl = np.array(lbl, dtype=np.int64)

        return img, lbl


class RandomCropPair(T.RandomCrop):
    def forward(self, img_lbl):
        img, lbl = img_lbl

        _, height, width = TF.get_dimensions(img)

        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
            lbl = TF.pad(lbl, padding, 255, self.padding_mode)  # pad class `void`
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
            lbl = TF.pad(lbl, padding, 255, self.padding_mode)

        crop_params = self.get_params(img, self.size)
        return TF.crop(img, *crop_params), TF.crop(lbl, *crop_params)


def color_map(n=256, normalized=False):
    # ref: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = np.float32 if normalized else np.uint8
    cmap = np.zeros((n, 3), dtype=dtype)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << (7 - j))
            g = g | (bitget(c, 1) << (7 - j))
            b = b | (bitget(c, 2) << (7 - j))
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = (cmap / 255) if normalized else cmap
    return cmap


if __name__ == '__main__':
    img_size = (224, 224)
    num_classes = 21

    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    train_data = datasets.VOCSegmentation('../data', '2012', 'trainval',
                                          transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    train_seg_data = SegmentationDataset(train_data, transform=RandomCropPair(img_size, pad_if_needed=True))
    train_loader = DataLoader(train_seg_data, batch_size=32, shuffle=True, num_workers=1, prefetch_factor=4)
    train_loader = PreprocessedDataLoader(train_loader, to_gpu)

    test_data = datasets.VOCSegmentation('../data', '2007', 'test',
                                         transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    test_seg_data = SegmentationDataset(test_data, transform=RandomCropPair(img_size, pad_if_needed=True))
    test_loader = DataLoader(test_seg_data, batch_size=32, num_workers=1, prefetch_factor=4)
    test_loader = PreprocessedDataLoader(test_loader, to_gpu)

    pspnet = TinyPSPNet(num_classes).to('cuda').train()
    print(pspnet, '\n')

    weight = torch.ones(21)
    weight[0] = 0.5
    loss_fn = nn.CrossEntropyLoss(weight=weight.to('cuda'), ignore_index=255)
    opt = torch.optim.Adam(pspnet.parameters(), lr=3e-4)

    trainer.fit(pspnet, train_loader, test_loader, [loss_fn], opt,
                metrics=True, epochs=30)
    pspnet.eval()

    cmap = color_map()

    batch = next(iter(test_loader))
    out = pspnet(batch[0])

    show_images(batch[0].cpu() * std.reshape((3, 1, 1)) + mean.reshape((3, 1, 1)))

    pred = out.argmax(dim=1).to('cpu', dtype=torch.uint8)
    show_images(cmap[pred], transpose=False)

    gt = batch[1].to('cpu').numpy()
    show_images(cmap[gt], transpose=False)
