import csv
import math
import random
import operator

def loadTrainingDataset(filename, trainingSet = [], testSet = []):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            #Holdout
            if random.random() < 0.66:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
    print('Train set: ' + repr((len(trainingSet))))
    print('Test set: ' + repr((len(testSet))))

def euclidianDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclidianDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append((distances[x][0]))
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def main():
    #iteration for learning
    #ML
    trainingSet = []
    testSet = []
    loadTrainingDataset('iris.data', trainingSet, testSet)
    #KNN
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], 9)# k = 9
        result = getResponse(neighbors)
        predictions.append(result)
        print('>> actual = ' + repr(testSet[x][-1])+ '| prediction = ' + repr(result))


main()