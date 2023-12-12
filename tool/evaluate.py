import numpy as np
from numpy.core.fromnumeric import nonzero
import scipy.io as scio
import  sys
from sklearn.utils.validation import check_non_negative

from torch import dist
sys.path.append('../')
import os
import argparse
import pickle
from sklearn import metrics
import math
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import  MultipleLocator
DATA_DIR='./frame_label'

# normalize scores in each sub video
NORMALIZE = True

# number of history frames, since in prediction based method, the first 4 frames can not be predicted, so that
# the first 4frames are undecidable, we just ignore the first 4 frames
DECIDABLE_IDX = 3


def parser_args():
    parser = argparse.ArgumentParser(description='evaluating the model, computing the roc/auc.')

    parser.add_argument('-f', '--file', type=str, help='the path of loss file.')
    parser.add_argument('-t', '--type', type=str, default='compute_auc',
                        help='the type of evaluation, choosing type is: plot_roc, compute_auc, '
                             'test_func\n, the default type is compute_auc')
    return parser.parse_args()


def score_smoothing(score, ws=25, function='mean', sigma=10):
    assert ws % 2 == 1, 'window size must be odd'
    assert function in ['mean', 'gaussian'], 'wrong type of window function'

    r = ws // 2
    weight = np.ones(ws)
    for i in range(ws):
        if function == 'mean':
            weight[i] = 1. / ws
        elif function == 'gaussian':
            weight[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    weight /= weight.sum()
    new_score = score.copy()
    new_score[r: score.shape[0] - r] = np.correlate(score, weight, mode='valid')
    return new_score


def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter  

class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, auc = {}'.format(self.dataset, self.auc)


class GroundTruthLoader(object):
    AVENUE = 'avenue'
    PED2 = 'ped2'
    SHANGHAITECH = 'shanghaitech'
    SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech_label')
    PED2_LABEL_PATH = os.path.join(DATA_DIR, 'ped2_label')
    AVENUE_LABEL_PATH = os.path.join(DATA_DIR, 'avenue_label')

    NAME_MAT_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/avenue.mat'),
        PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat')
    }

    NAME_FRAMES_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/testing'),
        PED2: os.path.join(DATA_DIR, 'ped2/testing'),
    }

    def __init__(self, dataset_txt, mapping_json=None, ):
        """
        Initial a ground truth loader, which loads the ground truth with given dataset name.

        :param mapping_json: the mapping from dataset name to the path of ground truth.
        """
        self.dataset_txt = dataset_txt
        if mapping_json is not None:
            with open(mapping_json, 'rb') as json_file:
                self.mapping = json.load(json_file)
        else:
            self.mapping = GroundTruthLoader.NAME_MAT_MAPPING

    def __call__(self, dataset):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt(self.dataset_txt)
        elif dataset == GroundTruthLoader.PED2:
            gt = self.__load_ped2_gt(self.dataset_txt)
        else:
            gt = self.__load_avenue_gt(self.dataset_txt)
        return gt


    @staticmethod
    def __load_shanghaitech_gt(dataset_txt):
        with open(file=dataset_txt, mode='r') as f:
            file_list = f.readlines()

        file_list.sort()
        gt = []
        for video in file_list:
            video_file = video.replace('\n', '')
            video_file = video_file + '.npy'
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video_file)))

        return gt

    @staticmethod
    def __load_ped2_gt(dataset_txt):
        with open(file=dataset_txt, mode='r') as f:
            file_list = f.readlines()

        file_list.sort()
        gt = []
        for video in file_list:
            video_file = video.replace('\n', '')
            video_file = video_file + '.npy'
            gt.append(np.load(os.path.join(GroundTruthLoader.PED2_LABEL_PATH, video_file)))

        return gt

    @staticmethod
    def __load_avenue_gt(dataset_txt):
        with open(file=dataset_txt, mode='r') as f:
            file_list = f.readlines()

        file_list.sort()
        gt = []
        for video in file_list:
            video_file = video.replace('\n', '')
            video_file = video_file + '.npy'
            gt.append(np.load(os.path.join(GroundTruthLoader.AVENUE_LABEL_PATH, video_file)))

        return gt

    @staticmethod
    def get_pixel_masks_file_list(dataset):
        # pixel mask folder
        pixel_mask_folder = os.path.join(DATA_DIR, dataset, 'pixel_masks')
        pixel_mask_file_list = os.listdir(pixel_mask_folder)
        pixel_mask_file_list.sort()

        # get all testing videos
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        # get all testing video names with pixel masks
        pixel_video_ids = []
        ids = 0
        for pixel_mask_name in pixel_mask_file_list:
            while ids < len(video_list):
                if video_list[ids] + '.npy' == pixel_mask_name:
                    pixel_video_ids.append(ids)
                    ids += 1
                    break
                else:
                    ids += 1

        assert len(pixel_video_ids) == len(pixel_mask_file_list)

        for i in range(len(pixel_mask_file_list)):
            pixel_mask_file_list[i] = os.path.join(pixel_mask_folder, pixel_mask_file_list[i])

        return pixel_mask_file_list, pixel_video_ids


# results = {'dataset': dataset, 'psnr': video_output}
def load_psnr_gt(results, dataset_txt):

    dataset = results['dataset']
    psnr_records = results['psnr']

    num_videos = len(psnr_records)

    # load ground truth
    gt_loader = GroundTruthLoader(dataset_txt)
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnr_records, gt


def load_psnr_gt_flow(loss_file):
    with open(loss_file, 'rb') as reader:

        results = pickle.load(reader)

    dataset = results['dataset']
    psnrs = results['psnr']
    flows = results['flow']

    num_videos = len(psnrs)

    # load ground truth
    gt_loader = GroundTruthLoader()
    gt = gt_loader(dataset=dataset)

    assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
        .format(num_videos, len(gt))

    return dataset, psnrs, flows, gt


def load_psnr(loss_file):
    """
    load image psnr or optical flow psnr.
    :param loss_file: loss file path
    :return:
    """
    with open(loss_file, 'rb') as reader:

        results = pickle.load(reader)
    psnrs = results['psnr']
    return psnrs


def get_scores_labels(loss_file,reverse,smoothing):
    # the name of dataset, loss, and ground truth
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        if NORMALIZE:
            distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-8)
            if reverse:
                distance = 1 - distance
        if smoothing:
            distance = score_smoothing(distance)
        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)
    return dataset, scores, labels


def precision_recall_auc(loss_file,reverse,smoothing):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file,reverse,smoothing)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        auc = metrics.auc(recall, precision)

        results = RecordResult(recall, precision, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model PR-AUC = {}'.format(optimal_results))
    return optimal_results


def cal_eer(fpr, tpr):
    # makes fpr + tpr = 1
    eer = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    return eer


def compute_eer(loss_file,reverse,smoothing):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult(auc=np.inf)
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file,reverse,smoothing)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        eer = cal_eer(fpr, tpr)

        results = RecordResult(fpr, tpr, eer, dataset, sub_loss_file)

        if optimal_results > results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model EER = {}'.format(optimal_results))
    return optimal_results


# res = {'dataset': dataset, 'psnr': video_output}
def compute_auc(res, dataset_txt, reverse, smoothing):
    dataset, psnr_records, gt = load_psnr_gt(res, dataset_txt)
    # if dataset=='shanghaitech': 
    #     gt[51][5]=0

    num_videos = len(psnr_records)
    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    for i in range(num_videos):
        distance = psnr_records[i]
        if NORMALIZE:
            distance = distance
            # distance -= distance.min() + 1e-8
            # distance /= distance.max()
            if reverse:
                distance = 1 - distance
        # smooth the score
        if smoothing:
            filter_2d = gaussian_filter_(np.arange(1, 50), 20)
            padding_size = len(filter_2d) // 2
            in_ = np.concatenate((distance[:padding_size], distance, distance[-padding_size:]))
            distance = np.correlate(in_, filter_2d, 'valid')
            # distance = score_smoothing(distance)

        scores = np.concatenate((scores[:], distance), axis=0)
        labels = np.concatenate((labels[:], gt[i]), axis=0)

    # print("label.shape:{}, scores.shape:{}".format(labels.shape, scores.shape))
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    results = RecordResult(fpr, tpr, auc, dataset)
    return results


def compute_auc_average(res, dataset_txt, reverse, smoothing):
    auc_list = []
    dataset, psnr_records, gt = load_psnr_gt(res, dataset_txt)

    # the number of videos
    num_videos = len(psnr_records)
    auc = 0
    for i in range(num_videos):
        distance = psnr_records[i]
        if NORMALIZE:
            # distance = (distance-distance.min())/(distance.max()-distance.min()+1e-8)
            if reverse:
                distance = 1 - distance
        # to smooth the score
        if smoothing:
            # distance = score_smoothing(distance)
            filter_2d = gaussian_filter_(np.arange(1, 50), 20)
            padding_size = len(filter_2d) // 2
            in_ = np.concatenate((distance[:padding_size], distance, distance[-padding_size:]))
            distance = np.correlate(in_, filter_2d, 'valid')



        fpr, tpr, _ = metrics.roc_curve(
            np.concatenate(([0], np.array(gt[i], dtype=np.int8), [1])),
            np.concatenate(([0], np.array(distance, dtype=np.float32), [1])),
            pos_label=1)

        _auc = metrics.auc(fpr, tpr)
        # print('video {}: auc is {}'.format(file_name[i], _auc))
        auc += _auc
    auc /= num_videos
    auc_list.append(auc)
    return auc_list


def average_psnr(loss_file, reverse):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    max_avg_psnr = -np.inf
    max_file = ''
    for file in loss_file_list:
        psnr_records = load_psnr(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        avg_psnr = np.mean(psnr_records)
        if max_avg_psnr < avg_psnr:
            max_avg_psnr = avg_psnr
            max_file = file
        print('{}, average psnr = {}'.format(file, avg_psnr))

    print('max average psnr file Averge Score = {}, psnr = {}'.format(max_file, max_avg_psnr))


def calculate_psnr(loss_file,reverse,smoothing):
    optical_result = compute_auc(loss_file,reverse,smoothing)
    print('##### optimal result and model = {}'.format(optical_result))

    mean_psnr = []
    for file in os.listdir(loss_file):
        file = os.path.join(loss_file, file)
        dataset, psnr_records, gt = load_psnr_gt(file)

        psnr_records = np.concatenate(psnr_records, axis=0)
        gt = np.concatenate(gt, axis=0)

        mean_normal_psnr = np.mean(psnr_records[gt == 0])
        mean_abnormal_psnr = np.mean(psnr_records[gt == 1])
        mean = np.mean(psnr_records)
        print('mean normal psrn = {}, mean abnormal psrn = {}, mean = {}'.format(
            mean_normal_psnr,
            mean_abnormal_psnr,
            mean)
        )
        mean_psnr.append(mean)
    print('max mean psnr = {}'.format(np.max(mean_psnr)))


def calculate_score(loss_file,reverse,smoothing):
    if not os.path.isdir(loss_file):
        loss_file_path = loss_file
    else:
        optical_result = compute_auc(loss_file,reverse,smoothing)
        loss_file_path = optical_result.loss_file
        print('##### optimal result and model = {}'.format(optical_result))
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file_path)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        distance = (distance - distance.min()) / (distance.max() - distance.min())
        if reverse:
            distance=1-distance
        if smoothing:
            distance = score_smoothing(distance)
        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:-DECIDABLE_IDX]), axis=0)

    mean_normal_scores = np.mean(scores[labels == 0])
    mean_abnormal_scores = np.mean(scores[labels == 1])
    print('mean normal scores = {}, mean abnormal scores = {}, '
          'delta = {}'.format(mean_normal_scores, mean_abnormal_scores, mean_normal_scores - mean_abnormal_scores))


eval_type_function = {
    'compute_auc': compute_auc,
    'compute_eer': compute_eer,
    'precision_recall_auc': precision_recall_auc,
    'calculate_psnr': calculate_psnr,
    'calculate_score': calculate_score,
    'average_psnr': average_psnr,
    'average_psnr_sample': average_psnr
}


def evaluate(eval_type, save_file):
    assert eval_type in eval_type_function, 'there is no type of evaluation {}, please check {}' \
        .format(eval_type, eval_type_function.keys())
    eval_func = eval_type_function[eval_type]
    optimal_results = eval_func(save_file)
    return optimal_results


# res = {'dataset': dataset, 'psnr': video_output}
def evaluate_all(res, dataset_txt, reverse=True, smoothing=True):
    result = compute_auc(res, dataset_txt, reverse, smoothing)
    aver_result = compute_auc_average(res, dataset_txt, reverse, smoothing)
    return result, aver_result

if __name__ == '__main__':
    pickle_path = './test.pkl'
    result = evaluate_all(pickle_path, reverse=True, smoothing=True)
    print(result)
