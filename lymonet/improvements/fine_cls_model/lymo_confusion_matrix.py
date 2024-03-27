
from pathlib import Path
import torch
import numpy as np
import collections
from lymonet.apis.yolov8_api import ConfusionMatrix

from lymonet.apis.yolov8_api import (
    ClassificationValidator, ClassifyMetrics, ConfusionMatrix, 
    ap_per_class, compute_ap, plot_mc_curve, plot_pr_curve, smooth
)



class LymoConfusionMatrix(ConfusionMatrix):

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        super().__init__(nc, conf, iou_thres, task)

        self.metrics = None

    def process_cls_preds(self, preds, targets):
        """update confusion matrix based on predictions and targets"""
        super().process_cls_preds(preds, targets)
        return self.matrix
    
    def compute_prf1(self, pred, targets, plot=False, on_plot=None, save_dir=Path(), names=(), eps=1e-16, prefix=""):
        
        # self.pred: list(tensor(32,3), ...) len()=num_batch
        # self.targets: list(tebsor(32), ...) len()=num_batch
        pred = torch.cat(pred).cpu().detach().numpy()   # (793, 3) 3是topk
        target = torch.cat(targets).cpu().detach().numpy()  # (793)
        pred_cls = pred[:, 0]  # (793)
        conf = np.ones_like(pred_cls)  # (793, )  # 1.0

        unique_classes, nt = np.unique(target, return_counts=True)
        nc = len(unique_classes)

        ap, p_curve, r_curve = np.zeros((nc, pred.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

        tp = np.zeros_like(pred)  # (793, 3)
        for i in range(pred.shape[1]):  # 遍历topk次
            tp_topk = pred[:, i] == target  # (793, ) bool
            tp[:, i] = tp_topk
        tp_cumsum = np.cumsum(tp, 1)  # (793, 3)  # 每一行累计，代表topk的累计
        # correct = pred_cls == target  # (793, ) bool

        x, prec_values = np.linspace(0, 1, 1000), []

        for ci, c in enumerate(unique_classes):  # per-class
            i = target == c  # 类型等于当前类型的索引
            n_l = nt[ci]  # number of labels
            n_p = i.sum()   # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            fpc = np.cumsum(1 - tp_cumsum[i], 0)  # (n_p, 3)
            tpc = np.cumsum(tp_cumsum[i], 0)

            # recall
            # recall = correct[i] / (n_l + eps)
            recall = tpc / (n_l + eps)  # recall for class c
            r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = dict(enumerate(names))  # to dict
        if plot:
            plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
            plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
            plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
            plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

        i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives

        # mean_precision 考虑样本均衡 mp = sum(p_i * n_i) / sum(n_i
        # mp = np.sum([p[i]*nt[i] for i in range(len(p))]) / np.sum(nt)
        # mr = np.sum([r[i]*nt[i] for i in range(len(r))]) / np.sum(nt)
        # mf1 = np.sum([f1[i]*nt[i] for i in range(len(f1))]) / np.sum(nt)

        acc = tp.sum() / (nt.sum() + eps)
        return tp, fp, p, r, f1, ap, acc, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values
        # return tp, fp, p, r, f1, ap, mp, mr, mf1, acc, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values
    
    def merge_classes(self, merge_type="normal_vs_abnormals"):
        """
        type: normal_vs_abnormal
            : metastatic_vs_unmetastatic
        """
        if merge_type is None:
            return self.matrix
        #把矩阵变成二分类的矩阵
        confusion = self.matrix
        assert confusion.shape[0] == confusion.shape[1] == 3, 'confusion matrix must be square'
        # nc = confusion.shape[0]  # number of classes

        new_cm = np.zeros((2, 2))
        if merge_type == "normal_vs_abnormals":
            new_cm[0, 0] = confusion[0, 0] + confusion[1, 1] + confusion[0, 1] + confusion[1, 0]
            new_cm[0, 1] = confusion[0, 2] + confusion[1, 2]
            new_cm[1, 0] = confusion[2, 0] + confusion[2, 1]
            new_cm[1, 1] = confusion[2, 2]
        elif merge_type == "metastatic_vs_unmetastatic":
            new_cm[0, 0] = confusion [1, 1]
            new_cm[0, 1] = confusion [1, 0]+confusion [1, 2]
            new_cm[1, 0] = confusion [0, 1]+confusion [2, 1]
            new_cm[1, 1] = confusion [0, 0]+confusion [0, 2]+confusion [2, 0]+confusion [2, 2]
        else:
            raise ValueError(f"type {merge_type} not supported")
        
        self.matrix = new_cm
        return new_cm


    def confusion2score(self, round=5, with_background=False, return_type='dict', names=None, percent=False):
        """
        input n*n confusion matrix, output: P R F1 score of each class and average ACC
        :param cm: confusion matrix, np.array, (n, n)
        :param round: default 5, decimal places
        :param with_backgroud: default False, if True, indicate the last row is background class
        :return: P, R, F1 and ACC
            P, R, F1: np.array (n,)
            acc: float32
        """
        confusion = self.matrix
        assert confusion.shape[0] == confusion.shape[1], 'confusion matrix must be square'
        nc = confusion.shape[0]  # number of classes
        actual_nc = nc - 1 if with_background else nc
        # sum of each row
        P_sum = np.repeat(np.sum(confusion, axis=1) + 1e-10, nc).reshape(nc, nc)
        P = confusion / P_sum
        R_sum = np.repeat(np.sum(confusion, axis=0) + 1e-10, nc).reshape(nc, nc).transpose()
        R = confusion / R_sum

        # 取对角
        P = np.diagonal(P, offset=0)
        R = np.diagonal(R, offset=0)
        F1 = 2 * P * R / (P + R + 1e-10)

        # acc
        correct = np.diagonal(confusion, offset=0)
        acc = np.sum(correct) / (np.sum(confusion) + 1e-10)
        # print(confusion, P, R, F1, acc)
        if percent:  # 变为百分比
            P = P * 100
            R = R * 100
            F1 = F1 * 100
            acc = acc * 100

        if isinstance(round, int):
            P = np.round(P, round)
            R = np.round(R, round)
            F1 = np.round(F1, round)
            acc = np.round(acc, round)

        if with_background:
            P = P[:-1]
            R = R[:-1]
            F1 = F1[:-1]

        mean_P = np.round(np.mean(P), round)
        mean_R = np.round(np.mean(R), round)
        mean_F1 = np.round(np.mean(F1), round)

        if return_type == 'list':
            if names is None:
                names = ['class_%d' % i for i in range(actual_nc)]
            assert len(names) == actual_nc, f'names must be same as number of classes {actual_nc} != {len(names)}'
            ret = list()
            head = [''] + names + ['mean']
            ret.append(head)
            ret.append(['P'] + list(P) + [mean_P])
            ret.append(['R'] + list(R) + [mean_R])
            ret.append(['F1'] + list(F1) + [mean_F1])
            ret.append(['ACC'] + [acc])

        elif return_type == 'dict':
            ret = collections.OrderedDict()
            ret['classes'] = names if names else ['class_%d' % i for i in range(actual_nc)]
            ret['precision'] = P
            ret['recall'] = R
            ret['F1'] = F1
            ret['accuracy'] = acc
            ret['mean_precision'] = mean_P
            ret['mean_recall'] = mean_R
            ret['mean_F1'] = mean_F1
        else:
            raise ValueError('return_type must be list or dict')
        self.metrics = ret
        return ret