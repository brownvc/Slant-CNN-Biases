import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()


class UNet(nn.Module):
    def __init__(self, channels=1, latent_dim=128):
        super(UNet, self).__init__()
        self.latent_dim = latent_dim
        self.gf = 32
        self.channels = channels

        self.conv2d1 = self.conv2d(chann_in=self.channels, chann_out=self.gf)
        self.conv2d2 = self.conv2d(chann_in=self.gf, chann_out=self.gf * 2)
        self.conv2d3 = self.conv2d(chann_in=self.gf * 2, chann_out=self.gf * 4)
        self.conv2d4 = self.conv2d(chann_in=self.gf * 4, chann_out=self.gf * 4)
        self.maxpool = nn.MaxPool2d((4, 4), padding=(1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2048, self.latent_dim)
        self.linear2 = nn.Linear(self.latent_dim, 256)
        self.deconv2d1 = self.deconv2d(chann_in=4, chann_out=self.gf * 4)
        self.deconv2d2 = self.deconv2d(chann_in=self.gf * 8, chann_out=self.gf * 4)
        self.deconv2d3 = self.deconv2d(chann_in=self.gf * 8, chann_out=self.gf * 2)
        self.deconv2d4 = self.deconv2d(chann_in=self.gf * 4, chann_out=self.gf)
        self.final = self.final_layer()

    def conv2d(self, chann_in, chann_out, k_size=(4, 4)):
        layer = nn.Sequential(
            nn.Conv2d(chann_in, chann_out, kernel_size=k_size, stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(chann_out),
        )
        return layer

    def deconv2d(self, chann_in, chann_out, k_size=(4, 4)):
        """Layers used during upsampling"""
        layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(chann_in, chann_out, kernel_size=k_size, stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(chann_out),
        )
        return layer

    def final_layer(self):
        layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(self.gf, self.channels, kernel_size=(4, 4), stride=(1, 1), padding='same'),
            nn.Tanh(),
        )
        return layer

    def forward(self, input):
        d1 = self.conv2d1(input)
        d2 = self.conv2d2(d1)
        d3 = self.conv2d3(d2)
        d4 = self.conv2d4(d3)
        d5 = self.maxpool(d4)
        d6 = self.flatten(d5)
        d7 = self.linear1(d6)

        u1 = self.linear2(d7)
        u2 = torch.reshape(u1, (-1, 4, 8, 8))
        u3 = self.deconv2d1(u2)
        u4 = self.deconv2d2(torch.cat([u3, d4], dim=1))
        u5 = self.deconv2d3(torch.cat([u4, d3], dim=1))
        u6 = self.deconv2d4(torch.cat([u5, d2], dim=1))
        out = self.final(u6)

        return d7, out

