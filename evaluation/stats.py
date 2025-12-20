import numpy as np
from scipy import stats
from sklearn import metrics
import matplotlib

matplotlib.use('Agg')  # 使用非 GUI 后端，适用于服务器或远程环境
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import os

import seaborn as sns
from sklearn.metrics import confusion_matrix

class AverageMeter(object):
    '''
    用于计算和存储某个数值的当前值val、计数count、总和sum、平均值sum
    '''

    def __init__(self):
        self.reset()

    # 重置
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 更新
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_stats(output, target):
    """
    计算 ACC, F1, AP, AUC, TPR@FPR=1% (T1), TPR@FPR=0.1% (T0.1)
    
    output: [N, C] 或 [N,1] 或 [N]，模型输出的 logits 或概率
    target: [N, C] 或 [N,1] 或 [N]，标签或 one-hot
    """
    output = np.array(output)
    target = np.array(target)
    
    # 将一维 target 转为二维 shape [N,1] 方便统一处理
    if target.ndim == 1:
        target = target[:, None]
    if output.ndim == 1:
        output = output[:, None]

    classes_num = output.shape[1]
    stats = []

    # ------------------ 处理二分类单 logit ------------------
    if classes_num == 1:
        # 如果 target 本身就是 0/1
        preds = (output > 0.5).astype(int)
        acc = metrics.accuracy_score(target, preds)
        f1 = metrics.f1_score(target, preds)

        try:
            ap = metrics.average_precision_score(target, output)
            auc = metrics.roc_auc_score(target, output)
            fpr, tpr, _ = metrics.roc_curve(target, output)
            t1 = np.interp(0.01, fpr, tpr)
            t01 = np.interp(0.001, fpr, tpr)
        except Exception as e:
            ap, auc, t1, t01 = -1, -1, -1, -1
            print(f"[Warning] Cannot compute metrics for binary class. Error: {e}")

        stats.append({
            'AP': ap,
            'AUC': auc,
            'ACC': acc,
            'F1': f1,
            'T1': t1,
            'T0.1': t01
        })

    # ------------------ 多分类或二分类双 logit ------------------
    else:
        # 如果 target 是 one-hot，则取 argmax
        if target.shape[1] > 1:
            y_true = np.argmax(target, axis=1)
        else:
            y_true = target[:,0]

        y_pred = np.argmax(output, axis=1)
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        for k in range(classes_num):
            try:
                # 如果 target 是 one-hot 或单标签
                if target.shape[1] > 1:
                    ap = metrics.average_precision_score(target[:, k], output[:, k])
                    auc = metrics.roc_auc_score(target[:, k], output[:, k])
                    fpr, tpr, _ = metrics.roc_curve(target[:, k], output[:, k])
                else:
                    # 将单标签转成二分类格式
                    binary_true = (y_true == k).astype(int)
                    ap = metrics.average_precision_score(binary_true, output[:, k])
                    auc = metrics.roc_auc_score(binary_true, output[:, k])
                    fpr, tpr, _ = metrics.roc_curve(binary_true, output[:, k])

                t1 = np.interp(0.01, fpr, tpr)
                t01 = np.interp(0.001, fpr, tpr)
            except Exception as e:
                ap, auc, t1, t01 = -1, -1, -1, -1
                print(f"[Warning] Class {k} cannot compute metrics. Error: {e}")

            stats.append({
                'AP': ap,
                'AUC': auc,
                'ACC': acc,
                'F1': f1,
                'T1': t1,
                'T0.1': t01
            })

    return stats

def plot_classwise_logits_histogram_bce(output, labels, normalize=False, save_path=None):
    """
    绘制两类样本的 logits 分布直方图

    参数:
        output: Tensor，形状 [N, 1]，模型输出的 logits
        labels: Tensor，形状 [N]，对应的标签（0 或 1）
        normalize: 是否对 logits 进行 min-max 归一化
        save_path: 如果提供路径，则保存图片
    """
    output = output.detach().cpu()
    labels = labels.detach().cpu()

    logit_values = output.squeeze(-1)  # shape: [N]

    if normalize:
        min_val = logit_values.min()
        max_val = logit_values.max()
        logit_values = (logit_values - min_val) / (max_val - min_val + 1e-8)

    class0_logits = logit_values[labels == 0].numpy()
    class1_logits = logit_values[labels == 1].numpy()

    # 画图
    plt.figure(figsize=(7, 4))
    plt.hist(class0_logits, bins=80, alpha=0.6, label='Class 0', color='blue', edgecolor='black')
    plt.hist(class1_logits, bins=80, alpha=0.6, label='Class 1', color='red', edgecolor='black')
    plt.xlabel('Logit Value' + (' (normalized)' if normalize else ''))
    plt.ylabel('Number of Samples')
    plt.title('Logits Distribution by Class')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"类别分布图已保存至: {save_path}")
    else:
        plt.show()

def plot_classwise_logits_histogram(output, labels, normalize=False, save_path=None):
    """
    绘制两类样本的 logits 分布直方图

    参数:
        output: Tensor，形状 [N, 2]，模型输出的 logits
        labels: Tensor，形状 [N]，对应的标签（0 或 1）
        normalize: 是否对 logits 进行 min-max 归一化
        save_path: 如果提供路径，则保存图片
    """
    output = output.detach().cpu()
    labels = labels.detach().cpu()
    
    # 只取每个样本属于其真实类别的那个 logit 值
    # 即对于类别 0，取 output[i, 0]，类别 1，取 output[i, 1]
    logit_values = output[:, 1]  # 所有样本的 class 1 logits

    if normalize:
        min_val = logit_values.min()
        max_val = logit_values.max()
        logit_values = (logit_values - min_val) / (max_val - min_val + 1e-8)

    class0_logits = logit_values[labels == 0].numpy()
    class1_logits = logit_values[labels == 1].numpy()

    # 画图
    plt.figure(figsize=(7, 4))
    plt.hist(class0_logits, bins=80, alpha=0.6, label='Class 0', color='blue', edgecolor='black')
    plt.hist(class1_logits, bins=80, alpha=0.6, label='Class 1', color='red', edgecolor='black')
    plt.xlabel('Logit Value' + (' (normalized)' if normalize else ''))
    plt.ylabel('Number of Samples')
    plt.title('Logits Distribution by Class')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"类别分布图已保存至: {save_path}")
    else:
        plt.show()

def plot_precision_recall_curve(y_true, y_scores, class_name, save_dir):
    """
    绘制 Precision-Recall (PR) 曲线，并显示 AP (Average Precision)。

    :param y_true: 真实标签 (1D NumPy 数组)，二进制 (0 或 1)
    :param y_scores: 预测得分 (1D NumPy 数组)，通常是模型输出的概率
    :param class_name: 类别名称
    :param save_dir: 存放目录
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, marker='', markersize=4, color='#EB5757', label=f"AP = {ap_score:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve {class_name}")
    plt.legend(loc='lower right')  # 固定图例位置
    plt.grid()
    # plt.show()
    save_path = os.path.join(save_dir, f"PR_curve_{class_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)

def plot_roc_curve(y_true, y_scores, class_name, save_dir):
    """
    绘制 ROC (Receiver Operating Characteristic) 曲线，并显示 AUC (Area Under Curve)。

    :param y_true: 真实标签 (1D NumPy 数组)，二进制 (0 或 1)
    :param y_scores: 预测得分 (1D NumPy 数组)，通常是模型输出的概率
    :param class_name: 类别名称
    :param save_dir: 存放目录
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, marker='', markersize=4, color='#2E75B5', label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # 参考线（随机分类器）
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve {class_name}")
    plt.legend(loc='lower right')  # 固定图例位置
    plt.grid()
    # plt.show()
    save_path = os.path.join(save_dir, f"ROC_curve_{class_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir, class_names=None, normalize=False):
    """
    绘制混淆矩阵（支持归一化）

    参数:
        save_dir: 存放目录
        y_true (np.ndarray): 真实类别数组 (1D)
        y_pred (np.ndarray): 预测类别数组 (1D)
        class_names (list, 可选): 类别名称列表，默认为 None（自动生成 0,1,2,...）
        normalize (bool, 可选): 是否归一化（显示百分比）

    示例:
        y_true = np.array([0, 1, 2, 1, 2, 0])
        y_pred = np.array([0, 2, 2, 1, 0, 0])
        plot_confusion_matrix(y_true, y_pred, class_names=["A", "B", "C"], normalize=True)
    """

    # 计算混淆矩阵
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    # 是否归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)  # 每行归一化，计算分类准确率
        cm = np.nan_to_num(cm)  # 避免除零错误

    # 类别标签
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # 绘制热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    # 设置标签
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    titlename = f"Confusion Matrix {('(normalized)' if normalize else '')}"
    plt.title(titlename)
    # plt.show()
    save_path = os.path.join(save_dir, f"CM{'_normalized' if normalize else ''}.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)

if __name__ == "__main__":
    pass