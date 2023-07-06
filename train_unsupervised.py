import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import argparse
import datetime
from models.unet import UNet
from models.unet_minus import UNet_minus
from models.AE_vgg import AE_vgg

from torch.utils.tensorboard import SummaryWriter

from dataloader import LoaderDotSizeVar

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='data_exp1')
    parser.add_argument('--model_name', type=str, default='unet', choices=['unet', 'unet-', 'AE_vgg'])
    parser.add_argument('--sample_interval', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    dataset = args.dataset
    model_name = args.model_name
    sample_interval = args.sample_interval
    save_interval = args.save_interval
    latent_dim = args.latent_dim
    lr = args.lr
    gpus = args.gpus

    start_time = datetime.datetime.now()
    logdir = 'train_log/%s/%s/epoch_%d_lr_%.4f_l_dim_%d/%s' \
             % (model_name, dataset, nb_epochs, lr, latent_dim,
                str(start_time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.join(logdir, "saved_weights"))

    # copy the model file to logdir
    from shutil import copyfile
    namefile = os.path.basename(os.path.realpath(__file__))
    copyfile(namefile, os.path.join(logdir, namefile))

    # tensorboard writer
    writer = SummaryWriter(os.path.join(logdir, "tensorboard_out"))

    if model_name == 'unet':
        model = UNet(latent_dim=latent_dim).to(device)
    elif model_name == 'unet-':
        model = UNet_minus(latent_dim=latent_dim).to(device)
    elif model_name == 'AE_vgg':
        model = AE_vgg(latent_dim=latent_dim).to(device)
    else:
        model = None

    if gpus > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(gpus)))
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_train = LoaderDotSizeVar(dataset_path=os.path.join('datasets', dataset), img_res=(256, 256), is_testing=False)
    dataset_test = LoaderDotSizeVar(dataset_path=os.path.join('datasets', dataset), img_res=(256, 256), is_testing=True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

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

            optimizer.zero_grad()
            latent, gen_img = model(imgs)
            # backprop
            loss = loss_fn(gen_img, imgs)
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                writer.add_scalar('data/training_loss', loss.item(), iteration)
                print(f'Training loss = {loss.item()}')

        # save model
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(logdir, "saved_weights", f"model_{epoch:04d}.pt")
            if gpus > 1:
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)

        ########################################################################################################
        # record results
        ########################################################################################################
        if (epoch + 1) % sample_interval == 0:
            print("Recording intermediate results...")
            save_path = os.path.join(logdir, "results", "epoch{:d}".format(epoch))
            data_path = os.path.join(logdir, "data", "epoch{:d}".format(epoch))
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(data_path, exist_ok=True)

            # save training and testing data for faster result analysis
            test_latent, test_texture, test_fov, test_optical_slant, test_physical_slant, test_convexity, test_size_var = [], [], [], [], [], [], []
            count = 0
            for index, data in enumerate(test_loader):
                # data = test_loader.__next__()
                imgs_t, texture_nb_t, fov_t, optical_slant_t, physical_slant_t, test_convexity_t, test_size_var_t = data

                # sample generated images
                with torch.no_grad():
                    latents, gen_imgs = model(imgs_t.float().to(device))
                    if count < 1:
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
                    latents, _ = model(imgs_t.float().to(device))
                train_latent.append(latents.cpu())
                train_convexity.append(train_convexity_t)
            train_latent = np.concatenate(train_latent, axis=0)
            train_convexity = np.concatenate(train_convexity, axis=0)

            print('Saving data for future analysis...')
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
