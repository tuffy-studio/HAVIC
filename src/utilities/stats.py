import numpy as np
from sklearn import metrics

def calculate_stats(output, target):
    """
    计算 ACC, AP, AUC，兼容二分类和多分类。
    output: [N, C]  target: [N, C]，C=1 也支持
    """
    classes_num = output.shape[1]
    stats = []

    # 二分类情况下（只有一个类别输出），使用 0.5 阈值判断准确率
    if classes_num == 1:
        preds = (output > 0.5).numpy().astype(int)
        acc = metrics.accuracy_score(target, preds)
    else:
        acc = metrics.accuracy_score(np.argmax(target, axis=1), np.argmax(output, axis=1))

    for k in range(classes_num):
        try:
            ap = metrics.average_precision_score(target[:, k], output[:, k])
            auc = metrics.roc_auc_score(target[:, k], output[:, k])
        except:
            ap, auc = -1, -1
            print(f"[Warning] Class {k} cannot compute AP or AUC (possibly no positive samples)")

        stats.append({
            'ap': ap,
            'auc': auc,
            'acc': acc
        })

    return stats


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count