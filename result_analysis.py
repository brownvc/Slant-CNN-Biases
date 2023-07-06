import csv
import glob

import cv2
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import rsatoolbox.data as rsd  # abbreviation to deal with dataset
import rsatoolbox.rdm as rsr
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import t

'''
Analyzing latent space results, generate visualizations, etc.
'''


def smoothen_(list_x, list_y, nb=100):
    index_ = np.argwhere(np.isfinite(list_y))[:, 0]
    list_x = np.array(list_x)[index_]
    list_y = np.array(list_y)[index_]
    poly = np.polyfit(list_x, list_y, 5)
    if len(list_x) > nb:
        nb = len(list_x)
    list_x = np.linspace(np.min(list_x), np.max(list_x), nb)
    poly_y = np.poly1d(poly)(list_x)

    return list_x, poly_y


def latent_space_visualizations(result_path, train_latent, test_latent, test_convexity, test_fov,
                                test_optical_slant, test_physical_slant, test_size_var, save_fig=True,
                                plot_legend=False, plot_color_bar=False):

    figure(figsize=(8, 6), dpi=1000)
    plt.tight_layout()

    legend_font_size = '20'

    # PCA & tSNE
    for use_pca in [True, False]:

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
        test_latent_reduced_concave = test_latent_reduced[
            (test_convexity == 0)]
        test_latent_reduced_convex = test_latent_reduced[
            (test_convexity == 1)]
        test_fov_concav = test_fov[(test_convexity == 0)]
        test_fov_convex = test_fov[(test_convexity == 1)]
        test_optical_slant_concav = test_optical_slant[(test_convexity == 0)]
        test_optical_slant_convex = test_optical_slant[(test_convexity == 1)]
        test_physical_slant_concav = test_physical_slant[(test_convexity == 0)]
        test_physical_slant_convex = test_physical_slant[(test_convexity == 1)]

        # latent space visualizations
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1],
                    label='concave', color='red', marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1],
                    label='convex', color='blue', marker='.')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=2)
        if plot_legend:
            plt.legend(fontsize=legend_font_size)
        if use_pca:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space visualization (PCA).png'), dpi=1000, bbox_inches='tight')
        else:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space visualization (TSNE).png'), dpi=1000, bbox_inches='tight')
        plt.clf()

        # latent space (FOV color mapped)
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_fov_concav,
                    label='concave', cmap="summer", marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_fov_convex,
                    label='convex', cmap="summer", marker='2')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        if plot_color_bar:
            cb = plt.colorbar(orientation="horizontal", pad=0.15)
            cb.set_label(label="FOV Value", fontsize=30, weight='bold')
        if plot_legend:
            plt.legend(fontsize=legend_font_size)
        if use_pca:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space & FOV (PCA).png'), dpi=1000, bbox_inches='tight')
        else:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space & FOV (TSNE).png'), dpi=1000, bbox_inches='tight')
        plt.clf()

        # optical slant color mapped
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_optical_slant_concav,
                    label='concave', cmap="summer", marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_optical_slant_convex,
                    label='convex', cmap="summer", marker='2')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        if plot_color_bar:
            cb = plt.colorbar(orientation="horizontal", pad=0.15)
            cb.set_label(label="Optical Slant Value", fontsize=30, weight='bold')
        if plot_legend:
            plt.legend(fontsize=legend_font_size)
        if use_pca:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space & Slant (PCA).png'), dpi=1000, bbox_inches='tight')
        else:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space & Slant (TSNE).png'), dpi=1000, bbox_inches='tight')
        plt.clf()

        # physical slant color mapped
        plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_physical_slant_concav,
                    label='concave', cmap="summer", marker='.')
        plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_physical_slant_convex,
                    label='convex', cmap="summer", marker='2')
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        if plot_color_bar:
            cb = plt.colorbar(orientation="horizontal", pad=0.15)
            cb.set_label(label="Physical Slant Value", fontsize=30, weight='bold')
        if plot_legend:
            plt.legend(fontsize=legend_font_size)
        if use_pca:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space & Physical Slant (PCA).png'), dpi=1000, bbox_inches='tight')
        else:
            if save_fig:
                plt.savefig(os.path.join(result_path, 'Latent space & Physical Slant (TSNE).png'), dpi=1000, bbox_inches='tight')
        plt.clf()

        # size variance color mapped
        try:
            test_size_var_concave = test_size_var[(test_convexity == 0)]
            test_size_var_convex = test_size_var[(test_convexity == 1)]
            plt.scatter(test_latent_reduced_concave[..., 0], test_latent_reduced_concave[..., 1], c=test_size_var_concave,
                        label='concave', cmap="summer", marker='.')
            plt.scatter(test_latent_reduced_convex[..., 0], test_latent_reduced_convex[..., 1], c=test_size_var_convex,
                        label='convex', cmap="summer", marker='2')
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            if plot_color_bar:
                cb = plt.colorbar(orientation="horizontal", pad=0.15)
                cb.set_label(label="Texture Irregularity Level", fontsize=20, weight='bold')
            if plot_legend:
                plt.legend(fontsize=legend_font_size)
            if use_pca:
                if save_fig:
                    plt.savefig(os.path.join(result_path, 'Latent space & Dot size variance (PCA).png'),
                                dpi=1000, bbox_inches='tight')
            else:
                if save_fig:
                    plt.savefig(os.path.join(result_path, 'Latent space & Dot size variance (TSNE).png'),
                                dpi=1000, bbox_inches='tight')
            plt.clf()
        except Exception:
            print("Texture irregularity plot failed!")


def plot_graphs(save_fig=True, save_stats=True, stats_file=None):
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(train_latent, train_convexity)
    convexity_preds_prob = clf.predict_proba(test_latent)
    convexity_preds_log_prob = clf.predict_log_proba(test_latent)
    convexity_preds = clf.predict(test_latent)

    plt.figure(figsize=(8, 6))

    # compute latent distances
    y = clf.decision_function(test_latent)
    w_norm = np.linalg.norm(clf.coef_)
    latent_dist = y / w_norm

    fovs = list(set(test_fov.tolist()))
    optical_slants = list(set(test_optical_slant.tolist()))
    physical_slants = list(set(test_physical_slant.tolist()))
    if len(test_size_var) > 0:
        irreg_levels = list(set(test_size_var.tolist()))
    else:
        irreg_levels = [0]
    fovs.sort()
    optical_slants.sort()
    irreg_levels.sort()
    physical_slants.sort()
    # fov intervals
    fov_intervals = [[2.5, 7.5], [7.5, 15], [15, 30], [30, 50], [50, 70]]
    fov_intervals = np.array(fov_intervals)
    fov_labels = [5., 10., 20., 40., 60.]

    # compute pseudo perceptual gains
    dist_normalized = np.abs(latent_dist) / np.max(np.abs(latent_dist))
    test_physical_slant_normalized = (test_physical_slant - np.min(test_physical_slant)) \
                                     / (np.max(test_physical_slant) - np.min(test_physical_slant))
    perceptual_gain = dist_normalized / test_physical_slant_normalized

    # overall accuracy
    acc_overall = np.mean(test_convexity == convexity_preds)
    print(f'The overall convexity classification accuracy is: {acc_overall}')

    # FOV VS many stats
    pred_accs_fov = []  # sign of curvature prediction accuracy per FOV
    prob_accs_fov = []
    dist_per_fov, dist_per_fov_concave, dist_per_fov_convex = [], [], []
    dis_per_fov_per_sizevar_dict = {}
    acc_per_fov_per_irreg_dict = {}
    pseudo_gian_per_var_dict = {}
    for fov in fovs:
        gt = test_convexity[(test_fov == fov)]
        pred = convexity_preds[(test_fov == fov)]
        prob = convexity_preds_prob[(test_fov == fov)]
        pred_acc = np.mean(gt == pred)
        prob_acc = np.mean(np.abs(gt - prob[:, 0]))
        dist_ = latent_dist[test_fov == fov]
        dist_concave = latent_dist[(test_fov == fov) & (test_convexity == 0)]
        dist_convex = latent_dist[(test_fov == fov) & (test_convexity == 1)]
        dist_avg = np.mean(np.abs(dist_))
        dist_avg_concave = np.mean(np.abs(dist_concave))
        dist_avg_convex = np.mean(np.abs(dist_convex))
        pred_accs_fov.append(pred_acc)
        prob_accs_fov.append(prob_acc)
        dist_per_fov.append(dist_avg)
        dist_per_fov_concave.append(dist_avg_concave)
        dist_per_fov_convex.append(dist_avg_convex)
        for irreg_level in irreg_levels:
            if irreg_level not in dis_per_fov_per_sizevar_dict.keys():
                dis_per_fov_per_sizevar_dict[irreg_level] = []
            if irreg_level not in pseudo_gian_per_var_dict.keys():
                pseudo_gian_per_var_dict[irreg_level] = []
            if irreg_level not in acc_per_fov_per_irreg_dict.keys():
                acc_per_fov_per_irreg_dict[irreg_level] = []
            dis_per_fov_per_sizevar = np.mean(np.abs(latent_dist[(test_fov == fov) & (test_size_var == irreg_level)]))
            dis_per_fov_per_sizevar_dict[irreg_level].append(dis_per_fov_per_sizevar)
            pseudo_gian_per_var = np.mean(perceptual_gain[(test_fov == fov) & (test_size_var == irreg_level)])
            pseudo_gian_per_var_dict[irreg_level].append(pseudo_gian_per_var)
            # compute accuracy vs fov per irregularity level
            gt_per_irreg = test_convexity[(test_fov == fov) & (test_size_var == irreg_level)]
            pred_per_irreg = convexity_preds[(test_fov == fov) & (test_size_var == irreg_level)]
            acc_per_irreg = np.mean(gt_per_irreg == pred_per_irreg)
            acc_per_fov_per_irreg_dict[irreg_level].append(acc_per_irreg)

    # optical slant VS accuracy
    acc_per_opt_slant = []
    for opt_slant in optical_slants:
        pred_ = convexity_preds[test_optical_slant == opt_slant]
        gt = test_convexity[test_optical_slant == opt_slant]
        acc = np.mean(pred_ == gt)
        acc_per_opt_slant.append(acc)
    # compute correlation
    r_opt_acc = np.corrcoef(np.stack([optical_slants, acc_per_opt_slant], axis=0))[0, 1]
    print(f"The correlation between FOV and averaged classification accuracy is {r_opt_acc}")

    # optical slant VS accuracy per FOV
    for fov_interval, fov_label in zip(fov_intervals, fov_labels):
        acc_per_fov_opt_slant = []
        for opt_slant in optical_slants:
            pred_ = convexity_preds[(test_fov >= fov_interval[0]) & (test_fov <= fov_interval[1])
                                    & (test_optical_slant == opt_slant)]
            gt = test_convexity[(test_fov >= fov_interval[0]) & (test_fov <= fov_interval[1])
                                & (test_optical_slant == opt_slant)]
            acc = np.mean(pred_ == gt)
            acc_per_fov_opt_slant.append(acc)
        plt.plot(optical_slants, acc_per_fov_opt_slant, label=f'FOV = {fov_label}')
        # plt.plot(list_x, list_y, label=f'FOV = {fov_label}')
    plt.xlabel('Optical Slant (degrees)', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'Optical Slant VS accuracy per FOV'), bbox_inches='tight')
    plt.clf()

    # optical slant VS distance
    dis_per_optical_slant, dis_per_optical_slant_concave, dis_per_optical_slant_convex = [], [], []
    for slant in optical_slants:
        dist_ = np.mean(np.abs(latent_dist[test_optical_slant == slant]))
        dist_concave = np.mean(np.abs(latent_dist[(test_optical_slant == slant) & (test_convexity == 0)]))
        dist_convex = np.mean(np.abs(latent_dist[(test_optical_slant == slant) & (test_convexity == 1)]))
        dis_per_optical_slant.append(dist_)
        dis_per_optical_slant_concave.append(dist_concave)
        dis_per_optical_slant_convex.append(dist_convex)

    # compute correlation
    r_opt = np.corrcoef(np.stack([optical_slants, dis_per_optical_slant], axis=0))[0, 1]
    print(f"The correlation between optical slant and averaged latent distance is {r_opt}")

    # FOV vs accuracy
    plt.plot(fovs, pred_accs_fov, label='Classification Accuracy')
    plt.xlabel('FOV (degrees)', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('FOV VS Classification accuracy ')
    if save_fig:
        plt.savefig(os.path.join(result_path, 'FOV VS accuracy'), bbox_inches='tight')
    plt.clf()
    # compute correlation
    r_fov_acc = np.corrcoef(np.stack([fovs, pred_accs_fov], axis=0))[0, 1]
    print(f"The correlation between FOV and averaged classification accuracy is {r_fov_acc}")

    # FOV vs latent distance
    plt.plot(fovs, dist_per_fov, label='Distance to boundary')
    plt.xlabel('FOV (degrees)', fontsize=15)
    plt.ylabel('Latent distance', fontsize=15)
    plt.title('FOV VS Latent distance')
    if save_fig:
        plt.savefig(os.path.join(result_path, 'FOV VS distance'), bbox_inches='tight')
    plt.clf()
    # compute correlation
    r_fov = np.corrcoef(np.stack([fovs, dist_per_fov], axis=0))[0, 1]
    print(f"The correlation between FOV and averaged latent distance is {r_fov}")

    plt.plot(fovs, dist_per_fov_concave, label='Concave')
    plt.plot(fovs, dist_per_fov_convex, label='Convex')
    plt.legend()
    plt.xlabel('FOV (degree)', fontsize=15)
    plt.ylabel('Latent distance', fontsize=15)
    plt.title('FOV VS Latent distance')
    if save_fig:
        plt.savefig(os.path.join(result_path, 'FOV VS distance 2'), bbox_inches='tight')
    plt.clf()

    # Slant VS distance per FOV
    for fov in fovs:
        dist_per_fov_slant = []
        for optical_slant in optical_slants:
            d = latent_dist[(test_fov == fov) & (test_optical_slant == optical_slant)]
            pred = convexity_preds[(test_fov == fov) & (test_optical_slant == optical_slant)]
            avg_d = np.mean(np.abs(d))
            dist_per_fov_slant.append(avg_d)
        plt.plot(optical_slants, dist_per_fov_slant, label='FOV = ' + str(fov))
    plt.xlabel('Optical slant (degrees)', fontsize=15)
    plt.ylabel('Latent distance', fontsize=15)
    plt.title('Optical slant VS Latent distance')
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'Optical slant VS Distance'), bbox_inches='tight')
    plt.clf()

    # FOV VS distance per dot size variance level
    for var in irreg_levels:
        dist_per_fov = dis_per_fov_per_sizevar_dict[var]
        plt.plot(fovs, dist_per_fov, label='Irregularity level = ' + str(var))
    plt.xlabel('Field of View (degrees)', fontsize=15)
    plt.ylabel('Latent distance', fontsize=15)
    plt.title('Field of View VS Latent distance per irregularity level')
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'FOV VS Distance per variance level'), bbox_inches='tight')
    plt.clf()

    # FOV VS pseudo perceptual gain per irregularity level
    for var in irreg_levels:
        pseudo_gain_per_fov = pseudo_gian_per_var_dict[var]
        plt.plot(fovs, pseudo_gain_per_fov, label='Irregularity level = ' + str(var))
    plt.xlabel('Field of View (degrees)', fontsize=15)
    plt.ylabel('Pseudo perceptual gain', fontsize=15)
    plt.title('Field of View VS Pseudo perceptual gain per irregularity level')
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'FOV VS perceptual gain per irregularity level'), bbox_inches='tight')
    plt.clf()

    # FOV VS accuracy per irregularity level
    for var in irreg_levels:
        acc_per_fov = acc_per_fov_per_irreg_dict[var]
        plt.plot(fovs, acc_per_fov, label='Irregularity level = ' + str(var))
    plt.xlabel('Field of View (degrees)', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    # plt.title('Field of View VS Pseudo perceptual gain per irregularity level')
    plt.legend()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'FOV VS accuracy per irregularity level'), bbox_inches='tight')
    plt.clf()

    if save_stats:
        assert stats_file, 'No stats file specified!'
        file_exist = os.path.isfile(stats_file)

        with open(stats_file, mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            if not file_exist:
                writer.writerow(['Model', 'Dataset', 'Latent dim', 'epoch', 'Classification acc',
                                 'FOV vs accuracy', 'Optical slant vs accuracy', 'FOV vs distance',
                                 'Optical slant vs distance', 'log id'])

            writer.writerow([model_name, dataset, str(latent_dim), str(epoch),
                             str(acc_overall), str(r_fov_acc), str(r_opt_acc),
                             str(r_fov), str(r_opt), log_id])


def compute_dissim(result_path, train_latent, test_latent, test_convexity, test_fov,
                   test_optical_slant, test_physical_slant, test_size_var, save_fig=True):
    obs_des = {'convexity': test_convexity, 'fov': test_fov, 'slant': test_optical_slant}

    data = rsd.Dataset(test_latent, obs_descriptors=obs_des, )
    rdms = rsr.calc_rdm(data, descriptor='convexity')
    rdm1 = rsr.rdms.RDMs(test_latent, descriptors=obs_des)
    R1 = rsr.compare(rdm1, rdm1, method='corr')
    # fig, ax, ret_val = rsatoolbox.vis.show_rdm(rdms, show_colorbar='panel')

    fovs = sorted(list(set(test_fov.tolist())))
    optical_slants = sorted(list(set(test_optical_slant.tolist())))
    try:
        size_var = sorted(list(set(test_size_var.tolist())))
    except Exception:
        size_var = [0]

    convexities = [0, 1]

    #########################################################################################################
    # compute and plot dissimilarity matrix
    #########################################################################################################
    for mode in ['FOV', 'Slant', 'Convexity', 'Regularity']:
        if mode == 'FOV':
            var = fovs
            data_var = test_fov
            xlabel = 'FOV 5°~60°'
        elif mode == 'Slant':
            var = optical_slants
            data_var = test_optical_slant
            xlabel = 'Optical Slant 25°~60°'
        elif mode == 'Regularity':
            var = size_var
            data_var = test_size_var
            xlabel = 'Irregularity level 0 to 4'
        else:
            var = convexities
            data_var = test_convexity
            # xlabel = 'Concave        Convex'
            xlabel = 'Sign of Curvature'

        dissim_matrix = np.zeros((len(var), len(var)))
        dissim_list = []
        var_val_list = []
        for i in range(len(var)):
            for j in range(len(var)):
                subgroup = R1[data_var == var[i]]
                subgroup = subgroup[:, data_var == var[j]]
                dissimi = 1 - subgroup
                len_ = len(dissimi)
                avg = np.mean(dissimi)
                # avg2 = np.sum(dissimi) / (len_**2 - len_)
                dissim_matrix[i, j] = avg

                if i == j:
                    dissim_list.append(avg)
                    var_val_list.append(var[j])

        r = np.corrcoef(np.array([var_val_list, dissim_list]))
        print(f'Correlation between {mode} and dissimilarity:')
        print(r)
        print()

        # normalize
        dissim_matrix = dissim_matrix / np.max(dissim_matrix)
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(dissim_matrix, interpolation='nearest')
        if mode == 'Convexity':
            plt.locator_params(axis='both', nbins=2)
            ax.set_xticklabels(['', 'Concave', 'Convex'])
            ax.set_yticklabels(['', 'Concave', 'Convex'])
        elif mode == 'Regularity':
            plt.locator_params(axis='both', nbins=5)
            ax.set_xticklabels(['', '0', '1', '2', '3', '4'])
            ax.set_yticklabels(['', '0', '1', '2', '3', '4'])
        else:
            if mode == 'FOV':
                label = [5.] + fovs
            else:
                label = [20.] + optical_slants
            plt.locator_params(axis='both', nbins=10)
            ax.set_xticklabels([str(round(f, 1)) + '°' for f in label], rotation=90)
            ax.set_yticklabels([str(round(f, 1)) + '°' for f in label])
        plt.xlabel(xlabel, labelpad=15, fontsize=14)
        plt.tick_params(left=False, top=False, bottom=False)
        fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
        plt.imshow(dissim_matrix, interpolation='nearest')
        if save_fig:
            plt.savefig(os.path.join(result_path, f'{mode} dissimilarity matrix'), bbox_inches='tight')


def physical_slant_vs_distance_tests(save_fig=True):
    '''
    Plots physcial slant VS latent distance graphs.
    '''

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(train_latent, train_convexity)
    y = clf.decision_function(test_latent)
    w_norm = np.linalg.norm(clf.coef_)
    dist = y / w_norm

    fovs = list(set(test_fov.tolist()))
    optical_slants = list(set(test_optical_slant.tolist()))
    physical_slants = list(set(test_physical_slant.tolist()))
    if len(test_size_var) > 0:
        size_vars = list(set(test_size_var.tolist()))
    else:
        size_vars = [0]
    fovs.sort()
    optical_slants.sort()
    size_vars.sort()
    physical_slants.sort()

    nb_phslt_bins = 20
    physical_slant_bins = np.linspace(0, 80, nb_phslt_bins)
    physical_slant_intervals = np.stack([physical_slant_bins[:-1], physical_slant_bins[1:]], axis=-1)
    fov_intervals = [[2.5, 7.5], [7.5, 15], [15, 30], [30, 50], [50, 70]]
    fov_intervals = np.array(fov_intervals)
    fov_labels = [5., 10., 20., 40., 60.]

    # physical slant VS decision distance
    dis_per_physical_slant, dis_per_physical_slant_concave, dis_per_physical_slant_convex = [], [], []
    for slant in physical_slants:
        dist_ = np.mean(np.abs(dist[test_physical_slant == slant]))
        dist_concave_avg = np.mean(np.abs(dist[(test_physical_slant == slant) & (test_convexity == 0)]))
        dist_convex_avg = np.mean(np.abs(dist[(test_physical_slant == slant) & (test_convexity == 1)]))
        dis_per_physical_slant.append(dist_)
        dis_per_physical_slant_concave.append(dist_concave_avg)
        dis_per_physical_slant_convex.append(dist_convex_avg)
    physical_slants_concave, poly_y_concave = smoothen_(physical_slants, dis_per_physical_slant_concave)
    physical_slants_convex, poly_y_convex = smoothen_(physical_slants, dis_per_physical_slant_convex)
    # plot
    plt.plot(physical_slants_concave, poly_y_concave, label='Concave')
    plt.plot(physical_slants_convex, poly_y_convex, label='Convex')
    plt.xlabel('Physical slant (degrees)')
    plt.ylabel('Latent distance')
    plt.title('Physical Slant VS Latent distance')
    plt.legend()
    # plt.show()
    if save_fig:
        # plt.savefig(f'results/{model_name}/%s/%s/%d/Physical Slant VS Decision distance 2' % (ds, folder_name, epoch))
        plt.savefig(os.path.join(result_path, 'Physical Slant VS Decision distance 2'))
    plt.clf()

    # physical slant VS decision distance per FOV
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(8, 4))
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Physical slant VS Latent distance')
    for fov_interval, fov_label in zip(fov_intervals, fov_labels):
        dist_per_fov_ph_slant_concave, dist_per_fov_ph_slant_convex = [], []
        for phslt_interval in physical_slant_intervals:
            d_concave = dist[(test_fov >= fov_interval[0]) & (test_fov <= fov_interval[1])
                             & (test_physical_slant >= phslt_interval[0])
                             & (test_physical_slant <= phslt_interval[1]) & (test_convexity == 0)]
            d_convex = dist[(test_fov >= fov_interval[0]) & (test_fov <= fov_interval[1])
                            & (test_physical_slant >= phslt_interval[0])
                            & (test_physical_slant <= phslt_interval[1]) & (test_convexity == 1)]
            avg_d_concave = np.mean(np.abs(d_concave))
            avg_d_convex = np.mean(np.abs(d_convex))
            dist_per_fov_ph_slant_concave.append(avg_d_concave)
            dist_per_fov_ph_slant_convex.append(avg_d_convex)
        # x_list, y_list = smoothen_(physical_slants, dist_per_fov_ph_slant)
        ax1.plot(np.mean(physical_slant_intervals, axis=-1), dist_per_fov_ph_slant_concave,
                 label='FOV = ' + str(fov_label))
        ax2.plot(np.mean(physical_slant_intervals, axis=-1), dist_per_fov_ph_slant_convex,
                 label='FOV = ' + str(fov_label))
    # plot dashed diagonal line
    x = np.linspace(0, 80, 500)
    ax1.plot(x, np.max(np.abs(dist)) / 80. * x, '--')
    ax2.plot(x, np.max(np.abs(dist)) / 80. * x, '--')
    # labels
    ax1.set_title('Concave')
    ax1.legend(loc="upper left")
    ax2.set_title('Convex')
    ax2.legend(loc="upper left")
    ax1.set(ylabel='Latent distance', xlabel='Physical slant (degrees)')
    ax2.set(ylabel='Latent distance', xlabel='Physical slant (degrees)')
    # plt.xlabel('Physical slant (degrees)')
    # plt.ylabel('Latent distance')
    # plt.legend()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'Physical slant VS Distance per fov'))
        # plt.savefig(f'results/{model_name}/%s/%s/%d/Physical slant VS Distance per fov' % (ds, folder_name, epoch))
    # plt.show()
    plt.clf()

    # linear regression on physical slant VS latent distance for concave and convex
    test_physical_slant_concave = test_physical_slant[test_convexity==0]
    test_physical_slant_convex = test_physical_slant[test_convexity==1]
    dist_concave = np.abs(dist[test_convexity==0])
    dist_convex = np.abs(dist[test_convexity==1])
    r_concave = np.corrcoef(np.array([test_physical_slant_concave, dist_concave]))
    r_convex = np.corrcoef(np.array([test_physical_slant_convex, dist_convex]))
    print('linear regression on physical slant VS latent distance for concave and convex')
    print(f'r for concave: {r_concave[0,1]}')
    print(f'r for convex: {r_convex[0,1]}')
    print()

    plt.scatter(test_physical_slant_concave, dist_concave, label='Concave')
    plt.scatter(test_physical_slant_convex, dist_convex, label='Convex')
    # plt.show()
    plt.clf()

    # T-test on concave VS convex mean latent distance
    test_physical_slant_concave = test_physical_slant[test_convexity==0]
    test_physical_slant_convex = test_physical_slant[test_convexity==1]
    slant_min = np.min(test_physical_slant_concave)
    slant_max = np.max(test_physical_slant_convex)
    dist_concave_avg = dist[(test_convexity==0)&(test_physical_slant>=slant_min)&(test_physical_slant<=slant_max)]
    dist_convex_avg = dist[(test_convexity==1)&(test_physical_slant>=slant_min)&(test_physical_slant<=slant_max)]
    stats_same, pvalue_same = stats.ttest_ind(dist_concave_avg, dist_convex_avg, equal_var=False)

    print('T-test on concave VS convex mean latent distance')
    print(f'T test statistics: {stats_same}')
    print(f'P value: {pvalue_same}')
    print()

    # multi_correlation_analysis
    test_physical_slant_concave = test_physical_slant[test_convexity == 0]
    test_optical_slant_concave = test_optical_slant[test_convexity == 0]
    phy_opt_slant_concave = test_physical_slant_concave * test_optical_slant_concave
    test_fov_concave = test_fov[test_convexity == 0]
    test_physical_slant_convex = test_physical_slant[test_convexity == 1]
    test_optical_slant_convex = test_optical_slant[test_convexity == 1]
    phy_opt_slant_convex = test_physical_slant_convex * test_optical_slant_convex
    test_fov_convex = test_fov[test_convexity == 1]
    phy_opt_fov_convex = test_physical_slant_convex * test_fov_convex

    independet_vars_convex = np.stack([test_physical_slant_convex, test_fov_convex], axis=-1)
    independet_vars_convex_df = pd.DataFrame(independet_vars_convex, columns=['Physical slants', 'FOV'])
    dependent_vars_convex_df = pd.DataFrame(dist_convex, columns=['Latent distance'])

    independet_vars_concave = np.stack([test_physical_slant_concave, test_fov_concave], axis=-1)
    independet_vars_concave_df = pd.DataFrame(independet_vars_concave, columns=['Physical slants', 'FOV'])
    dependent_vars_concave_df = pd.DataFrame(dist_concave, columns=['Latent distance'])

    family = sm.families.Poisson()
    print("Convex:")
    glm_convex = sm.GLM(dependent_vars_convex_df, independet_vars_convex_df, family=family)
    res_convex = glm_convex.fit()
    print(res_convex.summary())
    print()

    print("Concave:")
    glm_concave = sm.GLM(dependent_vars_concave_df, independet_vars_concave_df, family=family)
    res_concave = glm_concave.fit()
    print(res_concave.summary())


def ANOVA_latent_dist():
    "ANOVA test on what factors have significant effect on latent distance."
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(train_latent, train_convexity)
    y = clf.decision_function(test_latent)
    w_norm = np.linalg.norm(clf.coef_)
    signed_dist = y / w_norm
    unsigned_dist = np.abs(signed_dist)

    bin_width = 10.
    physical_slant_discrete = test_physical_slant // bin_width * bin_width + bin_width / 2.

    df = pd.DataFrame({'FOV': test_fov,
                       'Convexity': test_convexity,
                       'Optical_Slant': test_optical_slant,
                       'Physical_Slant': physical_slant_discrete,
                       'Latent_Distance': unsigned_dist})
    # print(df.head(10))

    model = ols('Latent_Distance ~ C(FOV) + C(Convexity) + C(Physical_Slant) + C(FOV):C(Convexity) + C('
                'Physical_Slant):C(Convexity) + C(Physical_Slant):C(FOV)',
                data=df).fit()
    table = sm.stats.anova_lm(model, typ=3)
    print(table)


def dist_correlation_tests(save_stats=True, stats_file=None):
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(train_latent, train_convexity)
    y = clf.decision_function(test_latent)
    w_norm = np.linalg.norm(clf.coef_)
    dist = np.abs(y / w_norm)

    fovs = list(set(test_fov.tolist()))
    optical_slants = list(set(test_optical_slant.tolist()))
    physical_slants = list(set(test_physical_slant.tolist()))
    if len(test_size_var) > 0:
        size_vars = list(set(test_size_var.tolist()))
    else:
        size_vars = [0]
    fovs.sort()
    optical_slants.sort()
    size_vars.sort()
    physical_slants.sort()

    for convexity in [0., 1.]:

        # compute texture stats
        L = 0.06  # dot size
        bound = 1.  # surface length
        FOV = test_fov[test_convexity==convexity]  # fov
        rho = test_physical_slant[test_convexity==convexity]  # physical slant
        FOV = FOV / 180. * np.pi
        rho = rho / 180. * np.pi
        d = np.sin(rho) * bound + np.cos(rho) * bound / np.tan(FOV/2.)  # distance from camera to wall
        D = d * np.cos(rho)  # distance from camera to closet point on surface
        lambda0 = 2*np.arctan(L/2./D)  # optical projection length
        sigma_min = rho - FOV/2.  # min optical slant
        sigma_max = rho
        sigma_cen = rho - FOV/4.
        phi_min = np.cos(sigma_min)  # foreshortening
        phi_max = np.cos(sigma_max)
        phi_cen = np.cos(sigma_cen)
        lambda_min = lambda0 * np.cos(sigma_min)  # length
        lambda_max = lambda0 * np.cos(sigma_max)
        lambda_cen = lambda0 * np.cos(sigma_cen)
        lambda_range = lambda_max - lambda_min
        width_min = lambda0 * np.square(np.cos(sigma_min))  # width
        width_max = lambda0 * np.square(np.cos(sigma_max))
        width_cen = lambda0 * np.square(np.cos(sigma_cen))
        width_range = width_max - width_min
        area_min = np.pi * np.square(lambda0) * (np.cos(sigma_min) ** 3) / 4.
        area_max = np.pi * np.square(lambda0) * (np.cos(sigma_max) ** 3) / 4.
        area_cen = np.pi * np.square(lambda0) * (np.cos(sigma_cen) ** 3) / 4.
        area_range = area_max - area_min

        w_min = np.cos(rho)
        w_max = dist[test_convexity == convexity] * np.tan(FOV/2.)
        w_range = w_max - w_min
        w_cen = w_min + w_range / 2.

        # vars = np.stack([dist[test_convexity==0], phi_min, phi_max, phi_cen, lambda_min, lambda_max, lambda_cen,
        # width_min, width_max, width_cen, area_min, area_max, area_cen], axis=0)
        vars = np.stack(
            [dist[test_convexity == convexity], lambda_min, lambda_max, lambda_cen, lambda_range, width_min,
             width_max, width_cen, width_range, area_min, area_max, area_cen, area_range, w_min, w_max, w_cen, w_range], axis=0)
        Rs = np.corrcoef(vars)[0, :]
        print(Rs)

        if save_stats:
            assert stats_file, 'No stats file specified!'
            file_exist = os.path.isfile(stats_file)

            with open(stats_file, mode='a') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                if not file_exist:
                    writer.writerow(['Model', 'Dataset', 'Latent dim', 'epoch', 'Convexity', 'Length min', 'Length max',
                                     'Length median', 'Length range', 'Width min', 'Width max', 'Width median',
                                     'Width range', 'Area min', 'Area max', 'Area median', 'Area range', 'Density min',
                                     'Density max', 'Density median', 'Density range', 'log id'])

                if convexity == 0:
                    writer.writerow([model_name, dataset, str(latent_dim), str(epoch), 'Concave']
                                    + [str(a) for a in Rs[1:]] + [folder_name.split('/')[-1]])
                else:
                    writer.writerow([model_name, dataset, str(latent_dim), str(epoch), 'Convex']
                                    + [str(a) for a in Rs[1:]] + [folder_name.split('/')[-1]])


def supervised_bias_analysis(save_fig=True):
    fovs = list(set(test_fov.tolist()))
    optical_slants = list(set(test_optical_slant.tolist()))
    physical_slants = list(set(test_physical_slant.tolist()))
    if len(test_size_var) > 0:
        size_vars = list(set(test_size_var.tolist()))
    else:
        size_vars = [0]
    fovs.sort()
    optical_slants.sort()
    size_vars.sort()
    physical_slants.sort()

    x = np.linspace(-80, 80, 500)
    pred_physical_slant = test_pred_physical_slant * 75
    pred_physical_slant_unsigned = np.abs(pred_physical_slant)
    signed_gt_physical_slant = test_physical_slant * (-1) ** (test_convexity + 1)
    plt.plot(x,x)
    plt.scatter(signed_gt_physical_slant, pred_physical_slant)
    plt.xlabel('Ground Truth Physical slant (degrees)', fontsize=11)
    plt.ylabel('Predicted Physical slant (degrees)', fontsize=11)
    plt.title('Physical Slant Ground Truth VS Prediction')
    # plt.legend()
    # plt.show()
    if save_fig:
        plt.savefig(os.path.join(result_path, 'Physical Slant Ground Truth VS Prediction'))
    # plt.show()
    plt.clf()

    pred_acc = []
    for fov in fovs:
        gt_convexity = test_convexity[test_fov==fov]
        pred_convexity = test_pred_convexity[test_fov==fov]
        acc = np.mean(gt_convexity == pred_convexity[:, 0])
        pred_acc.append(acc)
    plt.plot(fovs, pred_acc)
    # plt.show()

    pred_slant_diff_concave = (pred_physical_slant_unsigned[:, 0] - test_physical_slant)[test_convexity==0]
    pred_slant_diff_convex = (pred_physical_slant_unsigned[:, 0] - test_physical_slant)[test_convexity==1]
    stats_same, pvalue_same = stats.ttest_ind(pred_slant_diff_concave, pred_slant_diff_convex, equal_var=True)
    print('T test concave vs convex (pred slant - ground truth)')
    print('stats = ', stats_same)
    print('p-value = ', pvalue_same)
    print()

    stats_same, pvalue_same = stats.ttest_ind(signed_gt_physical_slant, pred_physical_slant[:, 0], equal_var=True)
    print('T test gt vs predicted')
    print('stats = ', stats_same)
    print('p-value = ', pvalue_same)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./train_log/AE_vgg/data_exp1/epoch_50_lr_0.0002_l_dim_64')
    args = parser.parse_args()

    log_dir = args.log_dir
    info_list = os.path.basename(log_dir).split('_')
    dataset = log_dir.split('/')[-2]
    model_name = log_dir.split('/')[-3]
    latent_dim = int(info_list[6])

    folders = glob.glob(os.path.join(log_dir, r'20*'))

    for folder in folders:

        print('************************************************************************')
        print('Analyzing ' + folder)
        print('************************************************************************')

        log_id = os.path.basename(folder)
        data_paths = glob.glob(os.path.join(folder, r'data/epoch*'))
        if len(data_paths) == 0:
            print('No data found...')
            continue

        os.makedirs(os.path.join(folder, 'results'), exist_ok=True)

        for data_path in data_paths:

            epoch = os.path.basename(data_path)
            result_path = os.path.join(folder, 'results', epoch)
            os.makedirs(result_path, exist_ok=True)

            # load data
            test_latent = np.load(os.path.join(data_path, 'test_latent.npy'))
            train_latent = np.load(os.path.join(data_path, 'train_latent.npy'))
            train_convexity = np.load(os.path.join(data_path, 'train_convexity.npy'))
            test_texture = np.load(os.path.join(data_path, 'test_texture.npy'))
            test_fov = np.load(os.path.join(data_path, 'test_fov.npy'))
            test_optical_slant = np.load(os.path.join(data_path, 'test_optical_slant.npy'))
            test_physical_slant = np.load(os.path.join(data_path, 'test_physical_slant.npy'))
            test_convexity = np.load(os.path.join(data_path, 'test_convexity.npy'))
            try:
                test_size_var = np.load(os.path.join(data_path, 'test_size_var.npy'))
            except Exception:
                test_size_var = []
                print("No size variation!")
            try:
                test_pred_convexity = np.load(os.path.join(data_path, 'test_pred_convexity.npy'))
            except Exception:
                test_pred_convexity = []
                print("No test_pred_convexity!")
            try:
                test_pred_physical_slant = np.load(os.path.join(data_path, 'test_pred_physical_slant.npy'))
            except Exception:
                test_pred_physical_slant = []
                print("No test_pred_physical_slant!")

            plot_graphs(save_fig=True, save_stats=False, stats_file='train_log/Convexity_pred_stats.csv')
            latent_space_visualizations(result_path=result_path,
                                        train_latent=train_latent,
                                        test_latent=test_latent,
                                        test_fov=test_fov,
                                        test_convexity=test_convexity,
                                        test_size_var=test_size_var,
                                        test_optical_slant=test_optical_slant,
                                        test_physical_slant=test_physical_slant,
                                        save_fig=True,
                                        plot_legend=True,
                                        plot_color_bar=True)

            physical_slant_vs_distance_tests(save_fig=True)
            compute_dissim(result_path=result_path,
                           train_latent=train_latent,
                           test_latent=test_latent,
                           test_fov=test_fov,
                           test_convexity=test_convexity,
                           test_size_var=test_size_var,
                           test_optical_slant=test_optical_slant,
                           test_physical_slant=test_physical_slant,
                           save_fig=True)

