from tqdm import tqdm
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
# Additional Scripts
from utils import transforms as T
from utils.dataset import DentalDataset
from utils.utils import EpochCallback

from config import cfg

from train_transunet import TransUNetSeg
import matplotlib.pyplot as plt

class TrainTestPipe:
    def __init__(self, train_path, train_sail_path, test_path, test_sail_path, model_path, device):
        self.device = device
        self.model_path = model_path

        self.train_loader = self.__load_dataset(train_path, train_sail_path, train=True)
        self.test_loader = self.__load_dataset(test_path, test_sail_path)

        self.transunet = TransUNetSeg(self.device)

    def __load_dataset(self, path, sail_path, train=False):
        shuffle = False
        transform = False

        if train:
            shuffle = True
            transform = transforms.Compose([T.RandomAugmentation(2)])

        set = DentalDataset(path= path, transform= transform, sail_path= sail_path)
        loader = DataLoader(set, batch_size=cfg.batch_size, shuffle=shuffle)

        return loader

    def __loop(self, loader, step_func, t):
        total_loss = 0

        for step, data in enumerate(loader):
            img, img_sail, mask = data['img'], data['img_sail'], data['mask']
            img = img.to(self.device)
            img_sail = img_sail.to(self.device)
            mask = mask.to(self.device)

            loss, cls_pred , metrics = step_func(img=img, img_sail=img_sail, mask=mask)

            total_loss += loss

            t.update()

        return total_loss , metrics

    def train(self):
        # Load pre-trained model weights before starting training
        if os.path.exists(self.model_path):
            self.transunet.load_model(self.model_path)  

        # Freeze the weights of the earlier layers, if desired
        for param in self.transunet.model.parameters():
            param.requires_grad = True
        # for param in self.transunet.model.fc.parameters():
        #     param.requires_grad = True    

        # num_features = self.transunet.model.fc.in_features
        # self.transunet.model.fc = nn.Linear(num_features, cfg.transunet.class_num)


        train_loss_plot = []
        test_loss_plot = []
        
        callback = EpochCallback(self.model_path, cfg.epoch,
                                 self.transunet.model, self.transunet.optimizer, 'test_loss', cfg.patience)

        for epoch in range(cfg.epoch):
            with tqdm(total=len(self.train_loader) + len(self.test_loader)) as t:
                train_loss ,  metrics = self.__loop(self.train_loader, self.transunet.train_step, t)

                test_loss = self.__loop(self.test_loader, self.transunet.test_step, t)

            callback.epoch_end(epoch + 1,
                               {'train_loss': train_loss / len(self.train_loader),
                                'test_loss': test_loss[0] / len(self.test_loader), 
                                "IOU": metrics[0] , 
                                "DSC": 1 -  train_loss / len(self.train_loader),
                                "F1-score": metrics[1] , 
                                "accuracy": metrics[2]})

            train_loss_plot.append(train_loss / len(self.train_loader))
            test_loss_plot.append(test_loss[0] / len(self.test_loader))

            # Plot the training and testing losses
            plt.figure()  # Create a new figure to avoid overlap
            plt.plot(train_loss_plot, label='Train Loss')
            plt.plot(test_loss_plot, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
    
            # Save the plot to the same file, overwriting the previous plot
            plt.savefig('F:/UNIVERCITY/sharifian/t3/plot/plot.png')
            plt.close()  # Close the figure to free memory      


            if callback.end_training:
                break

        #plot the train loss and test loss
        plt.plot(train_loss_plot, label=' Loss')
        plt.plot(test_loss_plot, label='test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()