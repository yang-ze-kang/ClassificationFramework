import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')

class MutiClassMetric(object):
    """多分类度量指标工具类"""

    def __init__(self, num_classes, labels_name) -> None:
        super().__init__()
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels_name = labels_name
        self.scores = []
        self.labels = []

    def update(self, scores, labels):
        """
        更新数据
        Args:
            scores:预测的各个类别的分数[num_samples,num_classes]
            labels:样本实际的标签，长度为num_samples
        """
        preds = np.argmax(scores, axis=1)
        for p, s, t in zip(preds, scores, labels):
            self.matrix[p, t] += 1
            self.scores.append(s)
            self.labels.append(t)

    def get_auc(self, type=None):
        """
        获取type类型的auc
        Args:
            type:可取每个类别、'macro'、'micro'，默认计算全部
        """
        bi_label = label_binarize(self.labels,
                                  classes=list(range(self.num_classes+1)))[:, :self.num_classes]
        scores = np.array(self.scores)
        if type is None:
            auc = {}
            for i in range(self.num_classes):
                tpr, fpr, _ = metrics.roc_curve(bi_label[:, i], scores[:, i])
                auc[self.labels_name[i]] = metrics.auc(tpr, fpr)
            auc['micro'] = metrics.roc_auc_score(bi_label, self.scores,
                                                 average='micro')
            auc['macro'] = metrics.roc_auc_score(bi_label, self.scores,
                                                 average='macro')
            return auc
        if type in self.labels_name:
            id = self.labels_name.index(type)
            tpr, fpr, _ = metrics.roc_curve(bi_label[:, id], scores[:, id])
            auc = metrics.auc(tpr, fpr)
            return auc
        assert type in ('macro', 'micro')
        auc = metrics.roc_auc_score(bi_label, self.scores, average=type)
        return auc

    def plot_roc(self, type=None):
        """
        绘制type的ROC曲线
        Args:
            type:绘制ROC的类型，可选每个类别、'micro'、'macro'，默认全部绘制
        """
        tpr = {}
        fpr = {}
        auc = {}
        bi_label = label_binarize(self.labels,
                                  classes=list(range(self.num_classes+1)))[:, :self.num_classes]
        scores = np.array(self.scores)
        # 计算每一类roc
        for i in range(self.num_classes):
            key = self.labels_name[i]
            tpr[key], fpr[key], _ = metrics.roc_curve(
                bi_label[:, i], scores[:, i])
            auc[key] = metrics.auc(tpr[key], fpr[key])
        # 计算micro roc
        tpr['micro'], fpr['micro'], _ = metrics.roc_curve(bi_label.ravel(),
                                                          scores.ravel())
        auc['micro'] = metrics.auc(tpr['micro'], fpr['micro'])
        # 计算macro roc
        all_fpr = np.unique(np.concatenate([fpr[self.labels_name[i]]
                                           for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr,
                                  fpr[self.labels_name[i]], tpr[self.labels_name[i]])
        mean_tpr /= self.num_classes
        tpr['macro'], fpr['macro'] = mean_tpr, all_fpr
        auc['macro'] = metrics.auc(tpr['macro'], fpr['macro'])
        # plot
        plt.figure()
        if type is not None:
            plt.plot(tpr[type], fpr[type],
                     label='%s ROC curve (area=%0.2f)' % (type, auc[type]))
        else:
            for key in tpr.keys():
                plt.plot(tpr[key], fpr[key],
                         label='%s ROC curve (area=%0.2f)' % (key, auc[key]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig('roc.png')

    def summary(self):
        """计算指标函数并打印"""
        # accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is ", acc)
        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        kappa = round((po - pe) / (1 - pe), 3)
        print("the model kappa is ", kappa)
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0

            table.add_row(
                [self.labels_name[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)
        plt.yticks(range(self.num_classes), self.labels_name)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        plt.savefig('confusion_matrix.png')

class BinaryClassMetric(object):
    """二分类度量指标工具类"""

    def __init__(self, num_classes, labels_name) -> None:
        super().__init__()
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels_name = labels_name
        self.scores = []
        self.labels = []

    def update(self, scores, labels):
        """
        更新数据
        Args；
            scores:预测的各个类别的分数[num_samples,num_classes]
            labels:样本实际的标签，长度为num_samples
        """
        preds = np.argmax(scores, axis=1)
        for p, s, t in zip(preds, scores, labels):
            self.matrix[p, t] += 1
            self.scores.append(s)
            self.labels.append(t)

    def get_auc(self, type=None):
        """
        获取type类型的auc
        Args:
            type:可取每个类别、'macro'、'micro'，默认计算全部
        """
        bi_label = label_binarize(self.labels,
                                  classes=list(range(self.num_classes+1)))[:, :self.num_classes]
        scores = np.array(self.scores)
        if type is None:
            auc = {}
            for i in range(self.num_classes):
                tpr, fpr, _ = metrics.roc_curve(bi_label[:, i], scores[:, i])
                auc[self.labels_name[i]] = metrics.auc(tpr, fpr)
            auc['micro'] = metrics.roc_auc_score(bi_label, self.scores,
                                                 average='micro')
            auc['macro'] = metrics.roc_auc_score(bi_label, self.scores,
                                                 average='macro')
            return auc
        if type in self.labels_name:
            id = self.labels_name.index(type)
            tpr, fpr, _ = metrics.roc_curve(bi_label[:, id], scores[:, id])
            auc = metrics.auc(tpr, fpr)
            return auc
        assert type in ('macro', 'micro')
        auc = metrics.roc_auc_score(bi_label, self.scores, average=type)
        return auc

    def plot_roc(self, type=None):
        """
        绘制type的ROC曲线
        Args:
            type:绘制ROC的类型，可选每个类别、'micro'、'macro'，默认全部绘制
        """
        tpr = {}
        fpr = {}
        auc = {}
        bi_label = label_binarize(self.labels,
                                  classes=list(range(self.num_classes+1)))[:, :self.num_classes]
        scores = np.array(self.scores)
        # 计算每一类roc
        for i in range(self.num_classes):
            key = self.labels_name[i]
            tpr[key], fpr[key], _ = metrics.roc_curve(
                bi_label[:, i], scores[:, i])
            auc[key] = metrics.auc(tpr[key], fpr[key])
        # 计算micro roc
        tpr['micro'], fpr['micro'], _ = metrics.roc_curve(bi_label.ravel(),
                                                          scores.ravel())
        auc['micro'] = metrics.auc(tpr['micro'], fpr['micro'])
        # 计算macro roc
        all_fpr = np.unique(np.concatenate([fpr[self.labels_name[i]]
                                           for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr,
                                  fpr[self.labels_name[i]], tpr[self.labels_name[i]])
        mean_tpr /= self.num_classes
        tpr['macro'], fpr['macro'] = mean_tpr, all_fpr
        auc['macro'] = metrics.auc(tpr['macro'], fpr['macro'])
        # plot
        plt.figure()
        if type is not None:
            plt.plot(tpr[type], fpr[type],
                     label='%s ROC curve (area=%0.2f)' % (type, auc[type]))
        else:
            for key in tpr.keys():
                plt.plot(tpr[key], fpr[key],
                         label='%s ROC curve (area=%0.2f)' % (key, auc[key]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig('roc.png')

    def summary(self):
        """计算指标函数并打印"""
        # accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is ", acc)
        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        kappa = round((po - pe) / (1 - pe), 3)
        print("the model kappa is ", kappa)
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0

            table.add_row(
                [self.labels_name[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)
        plt.yticks(range(self.num_classes), self.labels_name)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        # plt.tight_layout()
        plt.show()
        plt.savefig('confusion_matrix.png')