from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

from utils import log_string, loadData
from model import STGNN

parser = argparse.ArgumentParser()
# parser.add_argument('--time_slot', type = int, default = 5,
#                     help = 'a time step is 5 mins')
parser.add_argument('--P', type=int, default=12,
                    help='history steps')
parser.add_argument('--Q', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=16,
                    help='dims of each head attention outputs')

parser.add_argument('--train_ratio', type=float, default=0.6,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.2,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='epoch to run')
# parser.add_argument('--patience', type = int, default = 10,
#                     help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
# traffic flow data
parser.add_argument('--traffic_file', default='traffic_Data.npy',
                    help='traffic file')
parser.add_argument('--SE_file', default='**.npy',
                    help='spatial emebdding file')
parser.add_argument('--model_file', default='PEMS',
                    help='save the model to disk')
parser.add_argument('--log_file', default='log(PEMS)',
                    help='log file')

args = parser.parse_args()

# record the process
log = open(args.log_file, 'w')

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

trainX, trainY, valX, valY, testX, testY, mean, std = loadData(args)

# We can obmit this metric, because in this experiment we use a learnable ad
# SE = torch.from_numpy(SE).to(device)

log_string(log, "loading end....")


def res(model, valX, valY, mean, std):
    # 验证模式，会关闭dropout
    model.eval()

    # 验证集的个数，下面的解释同训练集
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():  # 不记录梯度
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx: end_idx]).float().to(device)
                y = valY[start_idx: end_idx]
                # te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat = model(X)

                # 预测的时候记得把归一化后的数据转化回来
                pred.append(y_hat.cpu().numpy() * std + mean)
                label.append(y)

    # pred是一个list，长度为num_batch，其中list的每一个元素是（batch_size of val_test,P,num of vertices）
    print(pred[0].shape)
    # 将每个val_batch中的样本连接起来
    pred = np.concatenate(pred, axis=0)
    print(pred.shape)
    label = np.concatenate(label, axis=0)

    # print(pred.shape, label.shape)
    maes = []
    rmses = []
    mapes = []
    wapes = []

    # 注意X 是 [val_num, P, num of vertices]

    # 这里的12是指预测的时间片的长度，对于滑动窗口中的每一个元素计算对应的精度指标
    for i in range(12):
        # 计算各类参数
        # pred[:, i, :]的shape就是（batch_size, num of vertices）
        mae, rmse, mape, wape = metric(pred[:, i, :], label[:, i, :])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        wapes.append(wape)

        log_string(log, 'step %d, mae: %.4f, rmse: %.4f, mape: %.4f, wape: %.4f' % (i + 1, mae, rmse, mape, wape))

    # 上面的是对每一个时间片进行单独分析，下面的对滑动窗口中整体进行分析
    mae, rmse, mape, wape = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    wapes.append(wape)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, wape: %.4f' % (mae, rmse, mape, wape))

    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)


def train(model, trainX, trainY, valX, valY, mean, std):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    # 将模型转为训练模式，启用Batch Normalization 和 Dropout
    model.train()
    # 构建优化器，Adam的参数为：模型的可学习的参数、学习率等超参数
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    # 根据网络性能调节学习率
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                              verbose=False, threshold=0.001, threshold_mode='rel',
                                                              cooldown=0, min_lr=2e-6, eps=1e-08)

    for epoch in tqdm(range(1, args.max_epoch + 1)):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        # 每个epoch对于数据进行打乱
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainY = trainY[permutation]
        # math.ceil 上取整 1.2->2
        # 计算出多少个batch
        num_batch = math.ceil(num_train / args.batch_size)
        mae, rmse, mape = res(model, valX, valY, mean, std)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                # 计算出每个batch对应的数据区间
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                # Tensor也可以在gpu上运行，numpy只能在cpu上运行
                # to.device 就让X，Y在gpu上跑
                X = torch.from_numpy(trainX[start_idx: end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx: end_idx]).float().to(device)
                # te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)

                # 梯度清0，这个需要在每个batch执行一次
                optimizer.zero_grad()

                # 正向传播
                y_hat = model(X)

                # 计算损失
                loss = _compute_loss(y, y_hat * std + mean)

                # 反向传播计算梯度
                loss.backward()
                # 进行梯度裁剪，防止梯度爆炸，阈值为5（梯度最大值）
                # 梯度裁剪放在计算梯度的后面，更新参数的前面
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                # 使用优化器进行参数更新
                optimizer.step()

                train_l_sum += loss.cpu().item()  # item()可以获取torch.Tensor的值。返回值为float类型

                # n += y.shape[0]

                # 一个batch结束，进行计数
                batch_count += 1
                # 进度条更新
                pbar.update(1)

        # 一个epoch结束，记录相关数据
        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                   % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        mae, rmse, mape = res(model, valX, valY, mean, std)
        # lr_scheduler.step()
        #根据mae[-1]的值决定是否要更新学习率
        lr_scheduler.step(mae[-1])
        if mae[-1] < min_loss:
            #保存mae值最小的模型
            min_loss = mae[-1]
            torch.save(model, args.model_file)


def test(model, valX, valY, mean, std):
    #用最好的模型test
    model = torch.load(args.model_file)
    mae, rmse, mape = res(model, valX, valY, mean, std)
    return mae, rmse, mape


def _compute_loss(y_true, y_predicted):
    # 计算损失
    return masked_mae(y_predicted, y_true, 0.0)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        # 处理无效值
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    # 对于无效数值就不求loss
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape


if __name__ == '__main__':
    maes, rmses, mapes = [], [], []
    for i in range(5):
        # i表示构建模型的次数，训练了多个模型
        log_string(log, "model constructed begin....")
        # 模型初始化构建
        model = STGNN(1, args.K * args.d, args.L, args.d).to(device)
        log_string(log, "model constructed end....")
        log_string(log, "train begin....")
        train(model, trainX, trainY, testX, testY, mean, std)
        log_string(log, "train end....")
        mae, rmse, mape = test(model, testX, testY, mean, std)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    log_string(log, "\n\nresults:")
    maes = np.stack(maes, 1)
    rmses = np.stack(rmses, 1)
    mapes = np.stack(mapes, 1)
    for i in range(12):
        log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (
            i + 1, maes[i].mean(), rmses[i].mean(), mapes[i].mean()))
        log_string(log,
                   'step %d, mae %.4f, rmse %.4f, mape %.4f' % (i + 1, maes[i].std(), rmses[i].std(), mapes[i].std()))
    log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].mean(), rmses[-1].mean(), mapes[-1].mean()))
    log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].std(), rmses[-1].std(), mapes[-1].std()))
