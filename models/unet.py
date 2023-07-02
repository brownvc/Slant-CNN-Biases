import torch
import torch.nn as nn
from dataloader_torch import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import argparse
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datetime

from torch.utils.tensorboard import SummaryWriter

from dataloader_torch import LoaderDotSizeVar

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='data_exp1')
    parser.add_argument('--sample_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--lambda_1', type=float, default=0.)
    parser.add_argument('--lambda_2', type=float, default=1)
    parser.add_argument('--lambda_tv', type=float, default=1e-6)
    args = parser.parse_args()

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    dataset = args.dataset
    sample_interval = args.sample_interval
    save_interval = args.save_interval
    latent_dim = args.latent_dim
    lr = args.lr
    gpus = args.gpus
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2
    lambda_tv = args.lambda_tv

    model_name = 'AE_vgg'

    start_time = datetime.datetime.now()
    logdir = 'train_log/%s/%s/epoch_%d_lr_%.4f_l_dim_%d/%s' \
             % (model_name, dataset, nb_epochs, lr, latent_dim,
                str(start_time.strftime("%Y%m%d-%H%M%S")))
    # os.makedirs(os.path.join(logdir, "saved_weights"))

    # copy the model file to logdir
    # from shutil import copyfile
    # namefile = os.path.basename(os.path.realpath(__file__))
    # copyfile(namefile, os.path.join(logdir, namefile))

    # tensorboard writer
    writer = SummaryWriter(os.path.join(logdir, "tensorboard_out"))

    model = UNet(latent_dim=latent_dim).to(device)

    if gpus > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        encoder_decoder_model = nn.DataParallel(model, device_ids=list(range(gpus)))
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_train = LoaderDotSizeVar(dataset_path=os.path.join('../datasets', dataset), img_res=(256, 256), is_testing=False)
    dataset_test = LoaderDotSizeVar(dataset_path=os.path.join('../datasets', dataset), img_res=(256, 256), is_testing=True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    epoch = 0
    iteration = 0
    while epoch < nb_epochs:
        epoch += 1
        print('epoch: {}'.format(epoch))

        for index, data in enumerate(train_loader):
            for j in range(len(data)):
                data[j] = data[j].to(device)
            imgs = data[0].float()

            iteration += 1

            # at first the content image
            # set all the gradients to zero
            optimizer.zero_grad()
            l = model(imgs)

            # backprop
            loss_c = loss.sum()
            loss_c.backward()
            optimizer.step()

            print(loss_c)

            if iteration % 100 == 0:
                writer.add_scalar('data/training_loss_content', torch.sum(loss_c).item(), iteration)
                writer.add_scalar('data/training_feature_loss_content', torch.sum(content_feature_loss).item(),
                                  iteration)
                writer.add_scalar('data/training_per_pixel_loss_content', torch.sum(per_pixel_loss).item(), iteration)

        # save model
        if (epoch + 1) % save_interval == 0:
            checkpoint_path_encoder = os.path.join(logdir, "saved_weights", f"encoder_{epoch:04d}.pt")
            checkpoint_path_decoder = os.path.join(logdir, "saved_weights", f"decoder_{epoch:04d}.pt")
            if gpus > 1:
                torch.save(encoder_decoder_model.module.encoder.state_dict(), checkpoint_path_encoder)
                torch.save(encoder_decoder_model.module.decoder.state_dict(), checkpoint_path_decoder)
            else:
                torch.save(encoder_decoder_model.encoder.state_dict(), checkpoint_path_encoder)
                torch.save(encoder_decoder_model.decoder.state_dict(), checkpoint_path_decoder)

        ########################################################################################################
        # record results
        ########################################################################################################
        if (epoch + 1) % sample_interval == 0:
            print("Recording intermediate results...")
            save_path = os.path.join(logdir, "results", "epoch{:d}".format(epoch))
            data_path = os.path.join(logdir, "data", "epoch{:d}".format(epoch))
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)

            if gpus > 1:
                encoder_t = encoder_decoder_model.module.encoder
                decoder_t = encoder_decoder_model.module.decoder
            else:
                encoder_t = encoder_decoder_model.encoder
                decoder_t = encoder_decoder_model.decoder

            # save training and testing data for analysis
            test_latent, test_texture, test_fov, test_optical_slant, test_physical_slant, test_convexity, test_size_var = [], [], [], [], [], [], []
            count = 0
            for index, data in enumerate(test_loader):
                # data = test_loader.__next__()
                imgs_t, texture_nb_t, fov_t, optical_slant_t, physical_slant_t, test_convexity_t, test_size_var_t = data

                with torch.no_grad():
                    latents = encoder_t(imgs_t.float().to(device))
                    if count < 1:
                        gen_imgs = decoder_t(latents)
                        gen_imgs = (gen_imgs + 1) * 0.5
                        for i in range(5):
                            img = gen_imgs[i].cpu().detach().numpy()
                            img = np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
                            input_ = (imgs_t[i].float().numpy() + 1) * 0.5
                            input_ = np.transpose(input_, (1, 2, 0))
                            plt.imsave(os.path.join(save_path, "input_{:d}_org.png".format(i)), input_[..., 0], cmap=plt.cm.gray)
                            plt.imsave(os.path.join(save_path, "gen_{:d}_gen.png".format(i)), img[..., 0], cmap=plt.cm.gray)
                            count += 1

                test_latent.append(latents.cpu())
                test_texture.append(texture_nb_t)
                test_fov.append(fov_t)
                test_optical_slant.append(optical_slant_t)
                test_physical_slant.append(physical_slant_t)
                test_convexity.append(test_convexity_t)
                test_size_var.append(test_size_var_t)

                # if index > 3:
                #     break

            test_latent = np.concatenate(test_latent, axis=0)
            test_texture = np.concatenate(test_texture, axis=0)
            test_fov = np.concatenate(test_fov, axis=0)
            test_optical_slant = np.concatenate(test_optical_slant, axis=0)
            test_physical_slant = np.concatenate(test_physical_slant, axis=0)
            test_convexity = np.concatenate(test_convexity, axis=0)
            test_size_var = np.concatenate(test_size_var, axis=0)

            train_latent, train_convexity = [], []
            for index, data in enumerate(train_loader):
                imgs_t, _, _, _, _, train_convexity_t, _ = data
                with torch.no_grad():
                    latents = encoder_t(imgs_t.float().to(device))
                train_latent.append(latents.cpu())
                train_convexity.append(train_convexity_t)
                # if index > 3:
                #     break
            train_latent = np.concatenate(train_latent, axis=0)
            train_convexity = np.concatenate(train_convexity, axis=0)

            print('Saving data...')

            np.save(os.path.join(data_path, 'train_latent'), train_latent)
            np.save(os.path.join(data_path, 'train_convexity'), train_convexity)
            np.save(os.path.join(data_path, 'test_latent'), test_latent)
            np.save(os.path.join(data_path, 'test_texture'), test_texture)
            np.save(os.path.join(data_path, 'test_fov'), test_fov)
            np.save(os.path.join(data_path, 'test_optical_slant'), test_optical_slant)
            np.save(os.path.join(data_path, 'test_physical_slant'), test_physical_slant)
            np.save(os.path.join(data_path, 'test_convexity'), test_convexity)
            np.save(os.path.join(data_path, 'test_size_var'), test_size_var)

            print('Analyzing latent space...')
            # convexity prediction
            convexity_prediction()

            # latent space visualization
            latent_space_visualization_for_paper()

            print('Finish recording results...')
        ########################################################################################################
