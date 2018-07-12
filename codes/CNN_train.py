import os
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import text_cnn
import pandas as pd
from sklearn.metrics import auc
import csv

def readcsvfile(filename):
    list1=[]
    list2=[]
    with open(filename) as f:
        for line in f:
            sl=line.split(',')
            example=[]
            if int(sl[-1])==0:
                label=[0,1]
            else:
                label=[1,0]
            for i in range(len(sl)-1):
                example.append(int(sl[i]))
            list1.append(example)
            list2.append(label)
    return list1,list2

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
    with tf.name_scope('stddev'):
        stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
    tf.summary.scalar('max',tf.reduce_max(var))
    tf.summary.scalar('min',tf.reduce_min(var))
    tf.summary.histogram('histogram',var)

cnn = text_cnn.TextCNN(
    sequence_num=30,
    num_classes=2,
    vocab_size=21,
    embedding_size=21,
    filter_sizes=[2, 3, 4, 5, 6],
    num_filters=64,
    flat_num=128
)
# Define Training procedure
with tf.name_scope('parameters'):
    # global_step=tf.Variable(0,trainable=False)
    # initial_lr = 0.1
    # lr=tf.train.exponential_decay(initial_lr,global_step,decay_steps=10,decay_rate=0.9)
    lr=tf.Variable(0.001,dtype=tf.float32)
    batch_size = 40
prediction = cnn.scores

with tf.name_scope('cross_entropy'):
    cross_entropy = cnn.loss
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

accuracy = cnn.accuracy


merged = tf.summary.merge_all()

def ROC(prob, test_label, drawROC, Is_ten_cross):
    num = len(prob)
    p = []
    t = []
    for i in range(num):
        p.append(prob[i][0])
        t.append(test_label[i][0])
    data = pd.DataFrame(index=range(0, num), columns=('probability', 'The true label'))
    data['The true label'] = t
    data['probability'] = p
    data.sort_values('probability', inplace=True, ascending=False)
    TPRandFPR = pd.DataFrame(index=range(len(data)), columns=('TP', 'FP'))

    for j in range(len(data)):
        data1 = data.head(n=j + 1)
        FP = len(data1[data1['The true label'] == 0][
                     data1['probability'] >= data1.head(len(data1))['probability']]) / float(
            len(data[data['The true label'] == 0]))
        TP = len(data1[data1['The true label'] == 1][
                     data1['probability'] >= data1.head(len(data1))['probability']]) / float(
            len(data[data['The true label'] == 1]))
        TPRandFPR.iloc[j] = [TP, FP]
    AUC = auc(TPRandFPR['FP'], TPRandFPR['TP'])
    if drawROC >= 1:
        plt.scatter(x=TPRandFPR['FP'], y=TPRandFPR['TP'], s=5, label='(FPR,TPR)', color='k')
        plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], 'k', label='AUC = %0.2f' % AUC)
        plt.legend(loc='lower right')
        plt.title('Receiver Operating Characteristic')
        plt.plot([(0, 0), (1, 1)], 'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 01.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        if Is_ten_cross == 0:
            plt.savefig("./Picture/CNN_ROC" + str(drawROC) + ".png")
        else:
            plt.savefig("./Picture/CNN_ROC_"+str(Is_ten_cross)+ "cross" + '_iter'+str(drawROC) +".png")
        plt.clf()
    return AUC, TPRandFPR['TP'], TPRandFPR['FP']


def JudgePositive(test_prob, th):
    IsPositive = []
    for i in range(len(test_prob)):
        if test_prob[i][0] >= th:
            IsPositive.append(1)
        else:
            IsPositive.append(0)
    return IsPositive


def Save_csv(data, name):
    csvfile = open('./Save_csv/' + name, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['TPR', 'FPR'])
    for i in range(len(data[0])):
        writer.writerow([data[0][i], data[1][i]])
    csvfile.close()


def Save_target(data,name):
    csvfile = open('./Save_csv/' + name, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['Accuracy', 'Specitivity', 'Sensitivity', 'AUC'])
    for i in range(len(data)):
        writer.writerow(data[i])
    csvfile.close()

def Metrix(test_prob_combine,data_label,is_cross,epoch):
    TP, TN, FP, FN = 0, 0, 0, 0
    test_pre_combine=JudgePositive(test_prob_combine, 0.5)
    for c in range(len(test_pre_combine)):
        if test_pre_combine[c] == data_label[c][0]:
            if test_pre_combine[c] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if test_pre_combine[c] == 1:
                FP += 1
            else:
                FN += 1
    Accuracy = (TP + TN) / (TP + TN + FN + FP)
    Specitivity = TN / (TN + FP)
    Sensitivity = TP / (TP + FN)
    if is_cross==0:
        AUC, TPR, FPR = ROC(test_prob_combine, data_label, epoch + 1, is_cross)
    else:
        AUC, TPR, FPR = ROC(test_prob_combine, data_label, epoch + 1, is_cross)

    metrix=[Accuracy, Specitivity, Sensitivity, AUC]
    roc=[TPR, FPR]
    return metrix,roc


def printmax(target,is_cross):
    allACC = []
    allAUC = []
    for x in target:
        allACC.append(x[0])
        allAUC.append(x[3])
    epochACC = allACC.index(max(allACC))
    epochAUC = allAUC.index(max(allAUC))
    resultACC=target[epochACC]
    resultAUC=target[epochAUC]
    if is_cross==0:
        Save_target(target, 'CNN_valid_target.csv')
        print("Valid_MaxACC: Iter: " + str(epochACC + 1) + "\nAccuracy:" + str(resultACC[0]) + "\nSpecitivity:" + str(
            resultACC[1]) + "\nSensitivity:" + str(resultACC[2]) + "\nAUC:" + str(resultACC[3]))
        print("Valid_MaxAUC: Iter: " + str(epochAUC + 1) + "\nAccuracy:" + str(resultAUC[0]) + "\nSpecitivity:" + str(
            resultAUC[1]) + "\nSensitivity:" + str(resultAUC[2]) + "\nAUC:" + str(resultAUC[3]))
    else:
        Save_target(target, 'CNN_'+str(is_cross)+'cross_target.csv')
        print("Cross_MaxACC: Iter: " + str(epochACC + 1) + "\nAccuracy:" + str(resultACC[0]) + "\nSpecitivity:" + str(
            resultACC[1]) + "\nSensitivity:" + str(resultACC[2]) + "\nAUC:" + str(resultACC[3]))
        print("Cross_MaxAUC: Iter: " + str(epochAUC + 1) + "\nAccuracy:" + str(resultAUC[0]) + "\nSpecitivity:" + str(
            resultAUC[1]) + "\nSensitivity:" + str(resultAUC[2]) + "\nAUC:" + str(resultAUC[3]))


def K_cross(train_file, k):
    (data_feature, data_label) = readcsvfile(train_file)
    n = len(data_feature) // k

    n_batch = ((k - 1) * n) // batch_size
    cross_target = []
    valid_target = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(10):
            sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
            test_pre_combine = []
            test_prob_combine = []
            for i in range(k):
                test_feature = data_feature[n * i:n * (i + 1)]
                test_label = data_label[n * i:n * (i + 1)]
                train_feature = data_feature[0:n * i] + data_feature[n * (i + 1):k * n]
                train_label = data_label[0:n * i] + data_label[n * (i + 1):k * n]
                for batch in range(n_batch):
                    batch_xs = train_feature[batch * batch_size:(batch + 1) * batch_size]
                    batch_ys = train_label[batch * batch_size:(batch + 1) * batch_size]
                    feed_dict={cnn.input_x: batch_xs,
                               cnn.input_y: batch_ys,
                               cnn.dropout_keep_prob: 0.7}
                    sess.run(train_step, feed_dict)
                test_prob = sess.run(prediction,
                                     feed_dict={
                                         cnn.input_x: test_feature,
                                         cnn.input_y: test_label,
                                         cnn.dropout_keep_prob: 1.0})

                test_prob_combine.extend(test_prob)
            (cross_metrix,cross_roc)=Metrix(test_prob_combine, data_label,k,epoch)

            Save_csv(cross_roc, "CNN_"+ str(k)+"_cross_iter" + str(epoch + 1) + '.csv')
            cross_target.append(cross_metrix)
            validtest_prob = sess.run(prediction,
                                 feed_dict={
                                     cnn.input_x: validtest_feature,
                                     cnn.input_y: validtest_label,
                                     cnn.dropout_keep_prob: 1.0})
            (valid_metrix,valid_roc)=Metrix(validtest_prob, validtest_label,0,epoch)
            Save_csv(valid_roc, "CNN_valid_iter" + str(epoch + 1) + '.csv')
            valid_target.append(valid_metrix)
        printmax(cross_target, k)
        printmax(valid_target, 0)


if __name__ == '__main__':
    is_ten_cross = 1
    train = 'Vocab_test.csv'
    K_cross(train, test, 10)