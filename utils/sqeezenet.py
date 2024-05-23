import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, s1x1, kernel_size=1)
        self.expand1x1 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.squeeze(x))
        return torch.cat([self.relu(self.expand1x1(x)), self.relu(self.expand3x3(x))], 1)

class SqueezeSegNet(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeSegNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            FireModule(512, 64, 256, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            FireModule(256, 64, 256, 256),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            FireModule(128, 32, 128, 128),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            FireModule(64, 16, 64, 64),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Ensure the output size matches the input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        return x

# Define the number of classes for your segmentation task
num_classes = 3  # Example: Change this to the actual number of classes you have

# Create the model
model = SqueezeSegNet(num_classes=num_classes)

# Print the model to verify its structure
print(model)

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    # print (sn)
    print (out.shape)
