import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import util.misc as utils


def segment_tiou(target_segments, test_segments):
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)

        tiou[i, :] = intersection / union
    return tiou


def average_recall_vs_nr_proposals(proposals,
                                   ground_truth,
                                   tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    video_lst = proposals['video-name'].unique()

    score_lst = []
    for videoid in video_lst:
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init', 'f-end']].values.astype(np.float)

        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init', 'f-end']].values

        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    pcn_lst = np.arange(1, 201) / 200.0
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))

    for ridx, tiou in enumerate(tiou_thresholds):
        for i, score in enumerate(score_lst):
            positives[i] = score.shape[0]

            for j, pcn in enumerate(pcn_lst):
                nr_proposals = int(score.shape[1] * pcn)
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1)
                                 > 0).sum()

        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    recall = recall.mean(axis=0)
    proposals_per_video = pcn_lst * (float(proposals.shape[0]) /
                                     video_lst.shape[0])
    return recall, proposals_per_video


def recall_vs_tiou_thresholds(proposals,
                              ground_truth,
                              nr_proposals=1000,
                              tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    video_lst = proposals['video-name'].unique()

    score_lst = []
    for videoid in video_lst:
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init', 'f-end']].values

        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init', 'f-end']].values
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    pcn = (video_lst.shape[0] * float(nr_proposals)) / proposals.shape[0]

    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])

    for ridx, tiou in enumerate(tiou_thresholds):

        for i, score in enumerate(score_lst):
            positives[i] = score.shape[0]

            nr_proposals = int(score.shape[1] * pcn)

            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) >
                                0).sum()

        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()

    return recall, tiou_thresholds


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def formatting(results):
    results_list = []
    frame_dict = load_json('datasets/thumos_frames_info.json')
    for vid, info in results.items():
        num_frames = frame_dict[vid]
        for preds_dict in info:
            scores = preds_dict['scores']
            boxes = preds_dict['boxes']
            boxes = boxes.detach().cpu().numpy()

            scores = scores.detach().cpu().numpy()
            for sample_idx in range(boxes.shape[0]):
                results_list.append([
                    float(boxes[sample_idx][0]),
                    float(boxes[sample_idx][1]),
                    float(scores[sample_idx]), num_frames, vid
                ])

    results_list = np.stack(results_list)

    results_pd = pd.DataFrame(
        results_list,
        columns=['f-init', 'f-end', 'score', 'video-frames', 'video-name'])
    return results_pd


def eval_props(results):
    results = formatting(results)
    ground_truth = pd.read_csv('datasets/thumos14_test_groundtruth.csv')

    average_recall, average_nr_proposals = average_recall_vs_nr_proposals(
        results, ground_truth)

    f = interp1d(average_nr_proposals,
                 average_recall,
                 axis=0,
                 fill_value='extrapolate')

    return {
        '50': str(f(50)),
        '100': str(f(100)),
        '200': str(f(200)),
        '500': str(f(500))
    }, results


class Thumos14Evaluator(object):

    def __init__(self):
        self.predictions = []

    def update(self, vid, predictions):
        self.predictions += [(vid, predictions)]

    def get_result(self):
        return self.predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        results = {}
        for vid, p in self.predictions:
            try:
                results[vid].append(p)
            except KeyError:
                results[vid] = []
                results[vid].append(p)
        return results
