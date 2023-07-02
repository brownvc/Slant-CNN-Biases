import torch
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
import argparse
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import datetime

from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()


########################################################################
# models
########################################################################

class ConvNet(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvNet, self).__init__()

        # layer1
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2)

        # layer2
        self.cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()

        # layer3
        self.cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(2)

        # layer4
        self.cv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm2d(num_features=128)
        self.relu4 = nn.ReLU()

        # fc1
        self.fc1 = nn.Linear(10368, latent_dim)
        self.relu_fc1 = nn.ReLU()

        # fc1
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.relu_fc2 = nn.ReLU()

        # fc1
        self.fc3 = nn.Linear(latent_dim, 2)
        # self.sfmx3 = nn.LogSoftmax()

    def forward(self, x):
        # l1
        x = self.max_pool1(self.relu1(self.batch_norm1(self.cv1(x))))

        # l2
        x = self.relu2(self.batch_norm2(self.cv2(x)))

        # l3
        x = self.max_pool3(self.relu3(self.batch_norm3(self.cv3(x))))

        # l4
        x = self.relu4(self.batch_norm4(self.cv4(x)))

        x = x.view(x.size(0), -1)

        # fc 1,2,3
        latent = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(latent))

        x = self.fc3(x)
        # x = self.sfmx3(x)

        return x, latent


class Resnet(nn.Module):
    """
    the autoencoder class
    """

    def __init__(self, pretrained=True, latent_dim=64):
        super(Resnet, self).__init__()

        self.resnet = models.resnet18(pretrained=pretrained)

        self.resnet.fc = nn.Linear(512, latent_dim)
        self.relu = nn.LeakyReLU()
        self.linear = nn.Linear(latent_dim, 2)

    def forward(self, input):
        # encode input
        latent = self.resnet(input)
        out = self.relu(latent)
        out = self.linear(out)
        # out = nn.Sigmoid()(out)

        return out, latent


class UNet(nn.Module):
    """
    the autoencoder class
    """

    def __init__(self, pretrained=True, latent_dim=64):
        super(UNet, self).__init__()

        self.latent_dim = latent_dim
        self.gf = 32

        # layer1
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=self.gf, kernel_size=4, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.gf)
        self.relu1 = nn.LeakyReLU(0.2)

        # layer2
        self.cv2 = nn.Conv2d(in_channels=self.gf, out_channels=self.gf * 2, kernel_size=4, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.gf * 2)
        self.relu2 = nn.LeakyReLU(0.2)

        # layer3
        self.cv3 = nn.Conv2d(in_channels=self.gf * 2, out_channels=self.gf * 4, kernel_size=4, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.gf * 4)
        self.relu3 = nn.LeakyReLU(0.2)

        # layer4
        self.cv4 = nn.Conv2d(in_channels=self.gf * 4, out_channels=self.gf * 4, kernel_size=4, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(num_features=self.gf * 4)
        self.relu4 = nn.LeakyReLU(0.2)

        self.max_pool1 = nn.MaxPool2d((4, 4), padding=1)
        self.linear1 = nn.Linear(1152, latent_dim)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, 2)

    def forward(self, input):
        # encode input

        x = self.cv1(input)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.cv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.cv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        x = self.cv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)

        x = self.max_pool1(x)
        x = nn.Flatten()(x)
        latent = self.linear1(x)
        out = self.relu(latent)
        out = self.linear2(out)
        # out = nn.Sigmoid()(out)

        return out, latent


def convexity_prediction(save_fig=True):
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(train_latent, train_convexity)
    convexity_preds_prob = clf.predict_proba(test_latent)
    convexity_preds_log_prob = clf.predict_log_proba(test_latent)
    convexity_preds = clf.predict(test_latent)

    y = clf.decision_function(test_latent)
    w_norm = np.linalg.norm(clf.coef_)
    dist = y / w_norm

    fovs = list(set(test_fov.tolist()))
    optical_slants = list(set(test_optical_slant.tolist()))
    size_vars = list(set(test_size_var.tolist()))
    fovs.sort()
    optical_slants.sort()
    size_vars.sort()

    # FOV VS accuracy
    pred_accs = []
    prob_accs = []
    dist_per_fov, dist_per_fov_concave, dist_per_fov_convex = [], [], []
    dis_per_fov_per_sizevar_dict = {}
    for fov in fovs:
        gt = test_convexity[(test_fov == fov)]
        pred = convexity_preds[(test_fov == fov)]
        prob = convexity_preds_prob[(test_fov == fov)]
        pred_acc = np.mean(gt == pred)
        prob_acc = np.mean(np.abs(gt - prob[:, 0]))
        dist_ = dist[test_fov == fov]
        dist_concave = dist[(test_fov == fov) & (test_convexity == 0)]
        dist_convex = dist[(test_fov == fov) & (test_convexity == 1)]
        dist_avg = np.mean(np.abs(dist_))
        dist_avg_concave = np.mean(np.abs(dist_concave))
        dist_avg_convex = np.mean(np.abs(dist_convex))
        pred_accs.append(pred_acc)
        prob_accs.append(prob_acc)
        dist_per_fov.append(dist_avg)
        dist_per_fov_concave.append(dist_avg_concave)
        dist_per_fov_convex.append(dist_avg_convex)

        for size_var in size_vars:
            if size_var not in dis_per_fov_per_sizevar_dict.keys():
                dis_per_fov_per_sizevar_dict[size_var] = []
            dis_per_fov_per_sizevar = np.mean(np.abs(dist[(test_fov == fov) & (test_size_var == size_var)]))
            dis_per_fov_per_sizevar_dict[size_var].append(dis_per_fov_per_sizevar)

    plt.plot(fovs, pred_accs, label='Classification accuracy')
    # plt.plot(fovs, prob_accs, label='Probability accuracy')
    # plt.legend()
    plt.xlabel('FOV (degree)')
    plt.ylabel('Accuracy')
    plt.title('FOV VS Classification accuracy ')
    if save_fig:
        plt.savefig(os.path.join(save_path, 'FOV VS accuracy'))
    # plt.show()
    plt.clf()

    plt.plot(fovs, dist_per_fov, label='Distance to boundary')
    # plt.plot(fovs, prob_accs, label='Probability accuracy')
    # plt.legend()
    plt.xlabel('FOV (degree)')
    plt.ylabel('Average distance to decision boundary')
    plt.title('FOV VS Decision boundary ')
    if save_fig:
        plt.savefig(os.path.join(save_path, 'FOV VS distance'))
    # plt.show()
    plt.clf()

    plt.plot(fovs, dist_per_fov_concave, label='Concave')
    plt.plot(fovs, dist_per_fov_convex, label='Convex')
    plt.legend()
    plt.xlabel('FOV (degree)')
    plt.ylabel('Average distance to decision boundary')
    plt.title('FOV VS Decision boundary ')
    if save_fig:
        plt.savefig(os.path.join(save_path, 'FOV VS distance 2'))
    # plt.show()
    plt.clf()

    # Slant VS distance per FOV
    for fov in fovs:
        dist_per_fov_slant = []
        for optical_slant in optical_slants:
            d = dist[(test_fov == fov) & (test_optical_slant == optical_slant)]
            pred = convexity_preds[(test_fov == fov) & (test_optical_slant == optical_slant)]
            avg_d = np.mean(np.abs(d))
            dist_per_fov_slant.append(avg_d)
        plt.plot(optical_slants, dist_per_fov_slant, label='FOV = ' + str(fov))
    plt.xlabel('Optical slant (degree)')
    plt.ylabel('Average distance to decision boundary')
    plt.title('Optical slant VS Average distance to decision boundary')
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(save_path, 'Optical slant VS Distance'))
    plt.clf()

    # FOV VS distance per dot size variance level
    for var in size_vars:
        dist_per_fov = dis_per_fov_per_sizevar_dict[var]
        plt.plot(fovs, dist_per_fov, label='Size variance level = ' + str(var))
    plt.xlabel('Field of View (degree)')
    plt.ylabel('Average distance to decision boundary')
    plt.title('Field of View VS Average distance to decision boundary per variance level')
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(save_path, 'FOV VS Distance per variance level'))
    plt.clf()


def convexity_pred_supervised(save_fig=True):
    convexity_preds = np.argmax(test_pred_convexity, axis=-1)
    fovs = list(set(test_fov.tolist()))
    optical_slants = list(set(test_optical_slant.tolist()))
    size_vars = list(set(test_size_var.tolist()))
    fovs.sort()
    optical_slants.sort()
    size_vars.sort()

    # FOV VS accuracy
    pred_accs = []
    # prob_accs = []
    # dist_per_fov, dist_per_fov_concave, dist_per_fov_convex = [], [], []
    # dis_per_fov_per_sizevar_dict = {}
    for fov in fovs:
        gt = test_convexity[(test_fov == fov)]
        pred = convexity_preds[(test_fov == fov)]
        # prob = convexity_preds_prob[(test_fov == fov)]
        pred_acc = np.mean(gt == pred)
        # prob_acc = np.mean(np.abs(gt - prob[:, 0]))
        # dist_ = dist[test_fov == fov]
        # dist_concave = dist[(test_fov == fov) & (test_convexity == 0)]
        # dist_convex = dist[(test_fov == fov) & (test_convexity == 1)]
        # dist_avg = np.mean(np.abs(dist_))
        # dist_avg_concave = np.mean(np.abs(dist_concave))
        # dist_avg_convex = np.mean(np.abs(dist_convex))
        pred_accs.append(pred_acc)
        # prob_accs.append(prob_acc)
        # dist_per_fov.append(dist_avg)
        # dist_per_fov_concave.append(dist_avg_concave)
        # dist_per_fov_convex.append(dist_avg_convex)

        # for size_var in size_vars:
        #     if size_var not in dis_per_fov_per_sizevar_dict.keys():
        #         dis_per_fov_per_sizevar_dict[size_var] = []
        #     dis_per_fov_per_sizevar = np.mean(np.abs(dist[(test_fov == fov) & (test_size_var == size_var)]))
        #     dis_per_fov_per_sizevar_dict[size_var].append(dis_per_fov_per_sizevar)

    plt.plot(fovs, pred_accs, label='Classification accuracy')
    # plt.plot(fovs, prob_accs, label='Probability accuracy')
    # plt.legend()
    plt.xlabel('FOV (degree)')
    plt.ylabel('Accuracy')
    plt.title('FOV VS Classification accuracy ')
    if save_fig:
        plt.savefig(os.path.join(save_path, 'FOV VS accuracy'))
    # plt.show()
    plt.clf()

    # unsigned_pred_physical_slants = test_pred_convexity
    # plt.scatter(test_physical_slant[test_convexity==0], unsigned_pred_physical_slants[test_convexity==0], label='Concave')
    # plt.scatter(test_physical_slant[test_convexity==1], unsigned_pred_physical_slants[test_convexity==1], label='Convex')
    # plt.xlabel('Ground truth physical slant')
    # plt.ylabel('Predicted physical slant')
    # plt.title('Physical slant ground truth VS  prediction')
    # plt.legend()
    # if save_fig:
    #     plt.savefig(os.path.join(save_path, 'Physical slant ground truth VS  prediction'))
    # # plt.show()
    # plt.clf()


def latent_space_visualization_for_paper(save_fig=True):
    for use_pca in [True, False]:
        # PCA
        if use_pca:
            pca = PCA(n_components=2)
            pca.fit(train_latent)
            test_latent_reduced = pca.transform(test_latent)
        else:
            tsne = TSNE(n_components=2)
            tsne.fit(train_latent)
            test_latent_reduced = tsne.fit_transform(test_latent)

        fovs = list(set(test_fov.tolist()))
        fovs.sort()

        # for fov in fovs:
        texture_nb = 1
        test_latent_reduced_concave = test_latent_reduced[
            (test_convexity == 0)]
        test_latent_reduced_convex = test_latent_reduced[
            (test_convexity == 1)]
        test_fov_concav = test_fov[(test_convexity==0)]
        test_fov_convex = test_fov[(test_convexity==1)]
        test_optical_slant_concav = test_optical_slant[(test_convexity == 0)]
        test_optical_slant_convex = test_optical_slant[(test_convexity == 1)]

        # plot a basic graph
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1],
                    label='concave', color='red', marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1],
                    label='convex', color='blue', marker='.')
        plt.legend()
        if use_pca:
            # plt.title(f'Latent space visualization (PCA)')
            if save_fig:
                plt.savefig(os.path.join(save_path, 'Latent space visualization (PCA).png'), dpi=1000)

        else:
            if save_fig:
                plt.savefig(os.path.join(save_path, 'Latent space visualization (TSNE).png'), dpi=1000)
        # plt.show()
        plt.clf()

        # FOV color mapped
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_fov_concav,
                    label='concave', cmap="summer", marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_fov_convex,
                    label='convex', cmap="summer", marker='2')
        plt.legend()
        cb = plt.colorbar(label="FOV", orientation="horizontal")
        if use_pca:
            # plt.title(f'Latent space visualization (PCA)')
            if save_fig:
                plt.savefig(os.path.join(save_path, 'Latent space & FOV (PCA).png'), dpi=1000)
        else:
            # plt.title(
            #     f'Latent space visualization (TSNE)')
            if save_fig:
                # plt.savefig(f'results/{model_name}/%s/%s/%d/Latent space & FOV (TSNE).png'
                #             % (ds, folder, epoch), dpi=1000)
                plt.savefig(os.path.join(save_path, 'Latent space & FOV (TSNE).png'), dpi=1000)
        plt.clf()

        # slant color mapped
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_optical_slant_concav,
                    label='concave', cmap="summer", marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_optical_slant_convex,
                    label='convex', cmap="summer", marker='2')
        plt.legend()
        cb = plt.colorbar(label="Optical Slant", orientation="horizontal")
        if use_pca:
            # plt.title(f'Latent space visualization (PCA)')
            if save_fig:
                plt.savefig(os.path.join(save_path, 'Latent space & Slant (PCA).png'), dpi=1000)
        else:
            if save_fig:
                plt.savefig(os.path.join(save_path, 'Latent space & Slant (TSNE).png'), dpi=1000)
        plt.clf()

        # size variance color mapped
        try:
            test_size_var_concave = test_size_var[(test_convexity == 0)]
            test_size_var_convex = test_size_var[(test_convexity == 1)]
            plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_size_var_concave,
                        label='concave', cmap="summer", marker='.')
            plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_size_var_convex,
                        label='convex', cmap="summer", marker='2')
            plt.legend()
            cb = plt.colorbar(label="Dot size variance", orientation="horizontal")
            if use_pca:
                # plt.title(f'Latent space visualization (PCA)')
                if save_fig:
                    plt.savefig(os.path.join(save_path, 'Latent space & Dot size variance (PCA).png'), dpi=1000)
            else:
                if save_fig:
                    plt.savefig(os.path.join(save_path, 'Latent space & Dot size variance (TSNE).png'), dpi=1000)
            plt.clf()
        except Exception:
            print("size variance plot failed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='combined_rand')
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=25)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='supervised_sign')
    parser.add_argument('--backbone', type=str, default='unet')
    args = parser.parse_args()

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    dataset = args.dataset
    sample_interval = args.sample_interval
    save_interval = args.save_interval
    latent_dim = args.latent_dim
    lr = args.lr
    gpus = args.gpus
    pretrained = args.pretrained
    model_name = args.model_name
    backbone = args.backbone

    start_time = datetime.datetime.now()
    logdir = 'train_log/%s/%s/epoch_%d_lr_%.4f_l_dim_%d_pertrain_%d_%s/%s' \
             % (model_name, dataset, nb_epochs, lr, latent_dim, int(pretrained), backbone,
                str(start_time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.join(logdir, "saved_weights"))

    # copy the model file to logdir
    from shutil import copyfile
    namefile = os.path.basename(os.path.realpath(__file__))
    copyfile(namefile, os.path.join(logdir, namefile))

    # tensorboard writer
    writer = SummaryWriter(os.path.join(logdir, "tensorboard_out"))

    if backbone == 'resnet18':
        model = Resnet(pretrained=False, latent_dim=latent_dim)
    # elif backbone == 'vgg16':
    #     model = VGG(pretrained=False, latent_dim=latent_dim)
    elif backbone == 'unet':
        model = UNet(latent_dim=latent_dim)
    else:
        model = None
    # model = ConvNet(latent_dim=latent_dim)
    model = model.to(device)

    if gpus > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(gpus)))
        # optimizer = optim.Adam(model.module.parameters(), lr=lr)
        optimizer = optim.Adam(model.module.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.module.parameters(), lr=lr, weight_decay=0.001, momentum=0.9)

    else:
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001, momentum=0.9)

    lambda1 = lambda epoch: 0.96 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    dataset_train = LoaderDotSizeVar(dataset_path=os.path.join('dataset', dataset), is_testing=False, nb_chan=3)
    dataset_test = LoaderDotSizeVar(dataset_path=os.path.join('dataset', dataset), is_testing=True, nb_chan=3)
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
            convexity_gt = data[5]
            physical_slant_gt = data[4].float()
            physical_slant_gt = physical_slant_gt * (-1)**(convexity_gt==1.)
            physical_slant_gt = physical_slant_gt / 150. + 0.5

            iteration += 1

            # at first the content image
            # set all the gradients to zero
            optimizer.zero_grad()
            outputs, _ = model(imgs)
            if torch.any(torch.isnan(outputs)):
                print('Has nan!')
            # outputs, latent = model(imgs)

            # backprop
            if 'physical_slant' in model_name:
                loss = torch.mean(torch.abs(torch.sigmoid(outputs.float()) - physical_slant_gt[:, None]))
            else:
                # loss = criterion(outputs.float(), convexity_gt[:, None].float())
                loss = criterion(outputs.float(), convexity_gt)

            loss.backward()
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            print(loss.item())

        scheduler.step()

        # save model
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(logdir, "saved_weights", f"{epoch:04d}.pt")
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

            # save training and testing data for analysis
            test_latent, test_texture, test_fov, test_optical_slant, test_physical_slant, test_convexity, test_size_var, test_pred_convexity = [], [], [], [], [], [], [], []
            count = 0
            for index, data in enumerate(test_loader):
                # data = test_loader.__next__()
                imgs_t, texture_nb_t, fov_t, optical_slant_t, physical_slant_t, test_convexity_t, test_size_var_t = data

                with torch.no_grad():
                    pred_convexity, latents = model(imgs_t.to(device).float())
                    # signed_pred_physical_slants = torch.sigmoid(signed_pred_physical_slants) * 150. - 75.

                test_latent.append(latents.cpu())
                test_pred_convexity.append(pred_convexity.cpu())
                test_texture.append(texture_nb_t)
                test_fov.append(fov_t)
                test_optical_slant.append(optical_slant_t)
                test_physical_slant.append(physical_slant_t)
                test_convexity.append(test_convexity_t)
                test_size_var.append(test_size_var_t)

                # if index > 3:
                #     break

            test_latent = np.concatenate(test_latent, axis=0)
            test_pred_convexity = np.concatenate(test_pred_convexity, axis=0)
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
                    latents = model(imgs_t.float().to(device))[1]
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
            np.save(os.path.join(data_path, 'test_pred_convexity'), test_pred_convexity)
            np.save(os.path.join(data_path, 'test_texture'), test_texture)
            np.save(os.path.join(data_path, 'test_fov'), test_fov)
            np.save(os.path.join(data_path, 'test_optical_slant'), test_optical_slant)
            np.save(os.path.join(data_path, 'test_physical_slant'), test_physical_slant)
            np.save(os.path.join(data_path, 'test_convexity'), test_convexity)
            np.save(os.path.join(data_path, 'test_size_var'), test_size_var)

            print('Analyzing latent space...')
            # convexity prediction
            # convexity_prediction()
            convexity_pred_supervised()

            # latent space visualization
            latent_space_visualization_for_paper()

            print('Finish recording results...')
        ########################################################################################################


