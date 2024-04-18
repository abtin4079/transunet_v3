import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# Additional Scripts
from config import cfg
from torch.utils.data import DataLoader

class DentalDataset(Dataset):
    output_size = cfg.transunet.img_dim

    def __init__(self, path, transform, sail_path):
        super().__init__()

        self.transform = transform

        img_folder = os.path.join(path, 'img')
        img_sail_folder = os.path.join(sail_path, 'grad_img')
        mask_folder = os.path.join(sail_path, 'mask')

        self.img_paths = []
        self.img_sail_paths = []
        self.mask_paths = []
        for p in os.listdir(img_sail_folder):
            name = p.split('.')[0]

            self.img_paths.append(os.path.join(img_folder, name + '.png'))
            self.img_sail_paths.append(os.path.join(img_sail_folder, name + '.png'))
            self.mask_paths.append(os.path.join(mask_folder, name + '.png'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_paths[idx]
        img_sail = self.img_sail_paths[idx]
        mask = self.mask_paths[idx]

        img_sail = cv2.imread(img_sail)
        img_sail = cv2.cvtColor(img_sail, cv2.COLOR_BGR2RGB)
        img_sail = cv2.resize(img_sail, (self.output_size, self.output_size))


        # preprocessing of original images
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.output_size, self.output_size))
        

        # preprocessing of masks
        mask = cv2.imread(mask, 0)
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)

        sample = {'img': img, 'img_sail': img_sail, 'mask': mask}
        

        #print("Before transform:", sample)
        if self.transform:
            sample = self.transform(sample)
        #print("After transform:", sample)

        
        img, img_sail, mask = sample['img'], sample['img_sail'], sample['mask']

        img = img / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype('float32'))

        img_sail = img_sail / 255.
        img_sail = img_sail.transpose((2, 0, 1))
        img_sail = torch.from_numpy(img_sail.astype('float32'))
        
        mask = mask / 255.
        mask = mask.transpose((2, 0, 1))
        mask = torch.from_numpy(mask.astype('float32'))

        return {'img': img, 'img_sail' : img_sail, 'mask': mask}

    def __len__(self):
      return len(self.img_paths)
    






if __name__ == '__main__':
    import torchvision.transforms as transforms
    import transforms as T
    transform = transforms.Compose([T.RandomAugmentation(2)])

    md = DentalDataset('F:/UNIVERCITY/sharifian/t1/datasets/tumor_dataset/process/train',
                       transform, 'F:/UNIVERCITY/sharifian/t1/datasets/tumor_dataset/sailency/train')
    loader = DataLoader(md, batch_size=cfg.batch_size, shuffle=True)

    print(loader)
    # for sample in md:
    #     print(sample['img'].shape)
    #     print(sample['mask'].shape)
    #     print(sample['img_sail'].shape)

    #     break
