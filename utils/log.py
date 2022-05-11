from matplotlib import pyplot as plt
import os
import torch
import time
import scipy.signal
import matplotlib
matplotlib.use('Agg')


class Log():
    def __init__(self, log_dir, model_name):
        date = time.strftime('%Y-%m-%d %H:%M', time.localtime())
        self.logs_dir = 'logs'
        self.log_dir = log_dir
        self.save_path = os.path.join(
            self.logs_dir, self.log_dir+'_'+str(date))
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accuracy = []
        self.val_accuracy = []
        os.makedirs(self.save_path)
        os.makedirs(os.path.join(self.save_path, 'model'))

    def append_loss(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        with open(os.path.join(self.save_path, "train_loss.txt"), 'a') as f:
            f.write(str(train_loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.train_losses))

        plt.figure()
        plt.title(self.model_name)
        plt.plot(iters, self.train_losses, 'red',
                 linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral',
                 linewidth=2, label='val loss')
        try:
            if len(self.train_losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.train_losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_path, "loss.png"))
        plt.cla()
        plt.close("all")

    def append_accuracy(self, accuracy, val_accuracy):
        self.train_accuracy.append(accuracy)
        self.val_accuracy.append(val_accuracy)
        with open(os.path.join(self.save_path, "train_accuracy.txt"), 'a') as f:
            f.write(str(accuracy))
            f.write("\n")
        with open(os.path.join(self.save_path, "val_accuracy.txt"), 'a') as f:
            f.write(str(val_accuracy))
            f.write("\n")
        self.accuracy_plot()

    def accuracy_plot(self):
        iters = range(len(self.train_accuracy))
        plt.figure()
        plt.title(self.model_name)
        plt.plot(iters, self.train_accuracy, 'red',
                 linewidth=2, label='train accuracy')
        plt.plot(iters, self.val_accuracy, 'coral',
                 linewidth=2, label='val accuracy')
        try:
            if len(self.train_accuracy) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.train_accuracy, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_accuracy, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_path,
                    "epoch_accuracy.png"))
        plt.cla()
        plt.close("all")

    def save_model(self, model, name=None):
        if name is None:
            save_path = os.path.join(self.save_path, 'model', 'best.pth')
        else:
            save_path = os.path.join(self.save_path, 'model', name+'.pth')
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        torch.save(model.state_dict(), save_path)
