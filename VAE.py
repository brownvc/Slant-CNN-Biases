import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import argparse
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datetime
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

from dataloader import LoaderDotSizeVar

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()


########################################################################
# Code adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
########################################################################

class VAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims,
                 **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self, recons, input, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # recons = args[0]
        # input = args[1]
        # mu = args[2]
        # log_var = args[3]

        kld_weight = 0.00025  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='data_exp1')
    parser.add_argument('--sample_interval', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gpus', type=int, default=1)
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

    model_name = 'VAE'

    start_time = datetime.datetime.now()
    logdir = 'train_log/%s/%s/epoch_%d_lr_%.4f_l_dim_%d/%s' \
             % (model_name, dataset, nb_epochs, lr, latent_dim,
                str(start_time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.join(logdir, "saved_weights"))

    # copy the model file to logdir
    from shutil import copyfile
    namefile = os.path.basename(os.path.realpath(__file__))
    copyfile(namefile, os.path.join(logdir, namefile))

    model = VAE(in_channels=1, latent_dim=latent_dim, hidden_dims=[32, 32, 64, 64, 128, 128, 512]).to(device)

    if gpus > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(gpus)))
        optimizer = optim.Adam(model.module.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_train = LoaderDotSizeVar(dataset_path=os.path.join('datasets', dataset), img_res=(256, 256), is_testing=False)
    dataset_test = LoaderDotSizeVar(dataset_path=os.path.join('datasets', dataset), img_res=(256, 256), is_testing=True)
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
            output_content, mu, log_var = model(imgs)
            if gpus > 1:
                loss_dict = model.module.loss_function(output_content, imgs, mu, log_var)
            else:
                loss_dict = model.loss_function(output_content, imgs, mu, log_var)

            # backprop
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

        # save model
        if (epoch + 1) % save_interval == 0:
            checkpoint_path_encoder = os.path.join(logdir, "saved_weights", f"encoder_{epoch:04d}.pt")
            checkpoint_path_decoder = os.path.join(logdir, "saved_weights", f"decoder_{epoch:04d}.pt")
            if gpus > 1:
                torch.save(model.module.state_dict(), checkpoint_path_encoder)
                torch.save(model.module.state_dict(), checkpoint_path_decoder)
            else:
                torch.save(model.state_dict(), checkpoint_path_encoder)
                torch.save(model.state_dict(), checkpoint_path_decoder)

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
                encoder_t = model.module.encoder
                decoder_t = model.module.decoder
            else:
                encoder_t = model.encoder
                decoder_t = model.decoder

            # save training and testing data for analysis
            test_latent, test_texture, test_fov, test_optical_slant, test_physical_slant, test_convexity, test_size_var = [], [], [], [], [], [], []
            count = 0
            for index, data in enumerate(test_loader):
                # data = test_loader.__next__()
                imgs_t, texture_nb_t, fov_t, optical_slant_t, physical_slant_t, test_convexity_t, test_size_var_t = data

                with torch.no_grad():
                    gen_imgs, latents, var = model(imgs_t.float().to(device))
                    if count < 1:
                        # gen_imgs = decoder_t(latents)
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
                    _, latents, _ = model(imgs_t.float().to(device))
                train_latent.append(latents.cpu())
                train_convexity.append(train_convexity_t)
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

            print('Finish recording results...')
        ########################################################################################################

