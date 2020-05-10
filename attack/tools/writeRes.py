#!/usr/bin/python

import os
import sys


class Params():
    def __init__(self, word2vec, file2store, model):
        self.word2vec = word2vec
        self.file2store = file2store
        self.model = model

def writeTestResults(str_Y_test, str_predictions, params, acc_list, avg_accuracy):
    #import pdb
    #pdb.set_trace()
    word2vecfile = params.word2vec
    file2store = params.file2store
    word2vecDict = parseWord2VecFile.loadData(word2vecfile)
    assert(len(str_Y_test) == len(str_predictions))
    tmpList = ['label\tprediction\tdistance\trank score']

    checkList1 = []
    checkList2 = []
    for item in str_Y_test:
        if item in word2vecDict.keys():
            continue
        checkList1.append(item)

    for item in word2vecDict.keys():
        if item in str_Y_test:
            continue
        checkList2.append(item)

    if len(checkList1)>0 or len(checkList2)>0:
        print('word in label but not in dict')
        print(checkList1)
        print('=====================')
        print('word in dict but not in label')
        print(checkList2)
        raise ValueError('checkList is not empty')

    scoreList = []
    rankscoreList = []
    for i in range(len(str_Y_test)):
        item_Y = str_Y_test[i]
        item_pred = str_predictions[i]
        vec_Y = word2vecDict[item_Y]
        vec_pred = word2vecDict[item_pred]
        score = computeDistance(vec_Y, vec_pred, 'cosin')
        scoreList.append(score)
        rankscore = computeRankScore(word2vecDict, item_pred, item_Y)
        rankscoreList.append(rankscore)
        tmpLine = '{}\t{}\t{}\t{}'.format(item_Y, item_pred, score, rankscore)
        tmpList.append(tmpLine)

    tmpLine = '\n\n=========== Statistics =========\n'
    tmpList.append(tmpLine)
    sumUp = sum(scoreList)
    average = sumUp/len(scoreList)
    tmpLine = 'total = {};\taverage = {}'.format(sumUp, average)
    tmpList.append(tmpLine)

    aveRankScore = np.mean(rankscoreList)
    tmpLine = 'average rank score is: {}'.format(str(aveRankScore))
    print(tmpLine)
    tmpList.append(tmpLine)

    varRankScore = np.std(rankscoreList)
    tmpLine = 'variance of rank score is: {}'.format(str(varRankScore))
    print(tmpLine)
    tmpList.append(tmpLine)

    tmpLine = 'acc result for each test round is: {}'.format(str(acc_list))
    tmpList.append(tmpLine)
    tmpLine = 'prediction with method {}, has a accuracy is: {}'.format(params.model, str(avg_accuracy))
    tmpList.append(tmpLine)
    tmpLine = 'accuracy variance is: {}'.format(np.std(acc_list))
    tmpList.append(tmpLine)

    content = '\n'.join(tmpList)
    print(content)
    with open(file2store, 'w') as f:
        f.write(content)

    return aveRankScore
