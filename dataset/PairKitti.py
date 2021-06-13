from random import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class PairKitti(Dataset):

    def __init__(self, path, set_type, resize=(128, 256)):
        super(Dataset, self).__init__()
        self.resize = resize

        self.ar = []
        idx_path = './dataset/data_paths/KITTI_stereo_' + set_type + '.txt'
        with open(idx_path) as f:
            content = f.readlines()

        for i in range(0, len(content), 2):
            left_id = content[i].strip()
            right_id = content[i + 1].strip()
            self.ar.append((path + '/' + left_id, path + '/' + right_id))

        if set_type == 'train':
            self.transform = self.train_deterministic_cropping
        elif set_type == 'test' or set_type == 'val':
            self.transform = self.test_val_deterministic_cropping

    def train_deterministic_cropping(self, img, side_img):
        # Center Crop
        img = TF.center_crop(img, (370, 740))
        side_img = TF.center_crop(side_img, (370, 740))

        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Random Horizontal Flip
        if random() > 0.5:
            img = TF.hflip(img)
            side_img = TF.hflip(side_img)

        # Convert to Tensor
        img = transforms.ToTensor()(img)
        side_img = transforms.ToTensor()(side_img)

        return img, side_img

    def test_val_deterministic_cropping(self, img, side_img):
        # Center Crop
        img = TF.center_crop(img, (370, 740))
        side_img = TF.center_crop(side_img, (370, 740))

        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Convert to Tensor
        img = transforms.ToTensor()(img)
        side_img = transforms.ToTensor()(side_img)

        return img, side_img

    def __getitem__(self, index):
        left_path, right_path = self.ar[index]

        img = Image.open(left_path)
        side_img = Image.open(right_path)
        image_pair = self.transform(img, side_img)

        return image_pair[0], image_pair[1], (left_path, right_path), index

    def __len__(self):
        return len(self.ar)

    def __str__(self):
        return 'KITTI_stereo'


if __name__ == '__main__':
    ds = PairKitti(path='./', set_type='train')
    ds = DataLoader(dataset=ds)
    for data in ds:
        img, cor_img, idx, _ = data
        print(img.shape, idx)
    print(len(ds))
