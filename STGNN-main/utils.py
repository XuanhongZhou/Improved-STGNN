import numpy as np
import pandas as pd


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape


def seq2instance(data, P, Q):
    """
    由于我们是根据前P个时间片的数据预测后Q个时间片的数据，因此这里对数据集进行了处理
    seq2instance
    """
    num_step, dims = data.shape
    # num_sample是新的样本总量，使用长度为P的滑动窗口，每次滑动一个数据项，因此最多有 num_step-P-Q+1 个样本
    num_sample = num_step - P - Q + 1
    # x.shape=(num_sample|新的样本总和, P|输入P个时间片的数据, dims|节点个数)
    # y.shape同理
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i: i + P]
        y[i] = data[i + P: i + P + Q]
    return x, y


def loadData(args):
    # np.squeeze 删除数组中维度为1的那一维，可以指定是第几维
    Traffic = np.squeeze(np.load(args.traffic_file))
    print("The shape of our traffic dataset:", end=' ')
    print(Traffic.shape)
    # train/val/test
    num_step = Traffic.shape[0]
    # num_step 即数据中的时间维度上（每5min的时间片有多少个）

    # 按照比例划分训练集、验证集和测试集
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps: train_steps + val_steps]
    test = Traffic[-test_steps:]
    # 我们的task的是给定P时间片，预测Q时间片，因此需要对数据处理
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # 归一化
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    return (trainX, trainY, valX, valY, testX, testY, mean, std)


"""
   这边的temporal embedding和spatial embedding 好像不需要

   # spatial embedding 
   f = open(args.SE_file, mode='r')
   lines = f.readlines()
   temp = lines[0].split(' ')
   N, dims = int(temp[0]), int(temp[1])
   SE = np.zeros(shape=(N, dims), dtype=np.float32)
   for line in lines[1:]:
       temp = line.split(' ')
       index = int(temp[0])
       SE[index] = temp[1:]

    # temporal embedding 
   Time = df.index
   dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
   timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
               // Time.freq.delta.total_seconds()
   timeofday = np.reshape(timeofday, newshape=(-1, 1))
   Time = np.concatenate((dayofweek, timeofday), axis=-1)
   # train/val/test
   train = Time[: train_steps]
   val = Time[train_steps: train_steps + val_steps]
   test = Time[-test_steps:]
   # shape = (num_sample, P + Q, 2)
   trainTE = seq2instance(train, args.P, args.Q)
   trainTE = np.concatenate(trainTE, axis=1).astype(np.int32)
   valTE = seq2instance(val, args.P, args.Q)
   valTE = np.concatenate(valTE, axis=1).astype(np.int32)
   testTE = seq2instance(test, args.P, args.Q)
   testTE = np.concatenate(testTE, axis=1).astype(np.int32)
   """