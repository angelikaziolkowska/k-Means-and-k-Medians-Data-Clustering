import numpy as np
import matplotlib.pyplot as plt


def dataPrep(fileNames):
    features = []
    for file, i in zip(fileNames, range(len(fileNames))):
        for line in open(file, 'r'):
            ln = line.strip().split(' ')
            ln.append(i)
            features.append(ln[1:])
    features = np.asmatrix(features, dtype='float64')
    return features


def shuffleData(features):
    permutation = np.random.permutation(len(features))
    features = features[permutation]
    return features


def kClustering(k, data, kMeans, l2):
    lastClusterAssignments = []
    if l2:
        temp = []
        for item in data[:, :-1]:
            temp.append(np.array(item / np.linalg.norm(item)))
        features = np.array(temp)
    else:
        features = data[:, :-1]

    # Initialization phase: as data was shuffled we can simply select first k
    representatives = features[:k, :]

    looped = -1
    while 1:
        clusterAssignments, clusterAndActual, xInClusters, countInCluster = [], [], [], []
        looped += 1

        # Assignment phase
        for x in features:
            minDist = float(np.inf)
            minIndex = -1
            for y, i in zip(representatives, range(k)):
                if kMeans:  # euclidean distance
                    dist = np.linalg.norm(x - y)
                else:  # manhattan distance
                    dist = np.linalg.norm(x - y, ord=1)
                if dist < minDist:
                    minDist = dist
                    minIndex = i
            clusterAssignments.append(minIndex)

        for i in range(k):
            tempInCluster = []
            for x, actualLabel, cluster in zip(features, data[:, -1], clusterAssignments):
                if cluster == i:
                    tempInCluster.append(x)
                    clusterAndActual.append([cluster, int(actualLabel)])
            countInCluster.append(len(tempInCluster))
            xInClusters.append(tempInCluster)

        if lastClusterAssignments and lastClusterAssignments == clusterAssignments or looped > 50:
            # different cluster/actual pairs with their count
            clusterAndActualPairCount = getClusterAndActualPairCounts(clusterAndActual)
            countInA = getACounts(clusterAndActualPairCount)
            fScores, precisions, recalls = getAllPrecisionsRecallsFSores(clusterAndActualPairCount, countInA,
                                                                         countInCluster, k)
            # returning B-CUBED results
            return getBCubed(countInCluster, fScores, looped, k, precisions, recalls)
        else:
            lastClusterAssignments = clusterAssignments
            # Optimization phase
            representatives = []
            for i in xInClusters:
                if kMeans:
                    representatives.append(np.dot(1 / len(i), (np.sum(i, axis=0))))
                else:
                    representatives.append(np.median(i, axis=0))


def getBCubed(countInCluster, fScores, ite, k, precisions, recalls):
    itemCount = sum(countInCluster)
    averagePrecision = np.divide(sum(precisions), itemCount)
    print('PRECISION: %.2f' % averagePrecision)
    averageRecall = np.divide(sum(recalls), itemCount)
    print('RECALL: %.2f' % averageRecall)
    averageFScore = np.divide(sum(fScores), itemCount)
    print('F-SCORE: %.2f' % averageFScore)
    if ite <= 50:
        print('No objects moved clusters after %d iterations. CONVERGENCE REACHED!' % ite)
    else:
        print('Due to 50 iterations of loop, CONVERGENCE REACHED!')
    print('------------------------------------------------------------------')
    return [k, averagePrecision, averageRecall, averageFScore]


def getAllPrecisionsRecallsFSores(clusterAndActualPairCount, countInA, countInCluster, k):
    precisions, recalls, fScores = [], [], []
    for pair in clusterAndActualPairCount:
        count = clusterAndActualPairCount[pair]
        for cx, i in zip(countInCluster, range(k)):
            if pair[0] == i:  # if cluster is i
                precX = count / cx
                for j in range(count):  # for each item in cluster append precision to
                    precisions.append(precX)
        for ax, i in zip(countInA, range(4)):
            if pair[1] == i:
                recX = count / ax
                for j in range(count):
                    recalls.append(recX)
    for p, r in zip(precisions, recalls):
        fScores.append((2 * r * p) / (r + p))
    return fScores, precisions, recalls


def getACounts(clusterAndActualPairCount):
    countInA = []
    for i in range(4):
        c = 0
        for p in clusterAndActualPairCount:
            if p[1] == i:
                c += clusterAndActualPairCount[p]
        countInA.append(c)
    return countInA


def getClusterAndActualPairCounts(clusterAndActual):
    pairCounts = {}
    for ca in clusterAndActual:
        caTuple = tuple(ca)
        if pairCounts:
            found = False
            for i in pairCounts:
                if caTuple == i:
                    found = True
                    pairCounts[i] += 1
                    break
            if not found:
                pairCounts[caTuple] = 1
        else:
            pairCounts[caTuple] = 1
    return pairCounts


def plotGraph(averages, title):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_xlabel('k')
    ax.set_ylabel('Average')
    for i, t in zip(range(1, 4), ['Precision', 'Recall', 'F-Score']):
        ax.plot(averages[:, i], label=t)
    plt.xticks(range(0, 9), range(1, 10))
    ax.legend()
    plt.show()


if __name__ == '__main__':
    features = dataPrep(['animals', 'countries', 'fruits', 'veggies'])
    kType = ['k-Medians', 'k-Means']
    unitL2 = ['', 'with unit L2 norm']
    options = [(1, 0), (0, 1), (1, 1), (0, 0)]
    bCubed100 = []

    # run all results x100
    for i in range(1, 100):
        features = shuffleData(features)
        bCubed = {}
        for o in options:
            bCubed[o] = []

        for k in range(1, 10):
            for o in options:
                print('(loop %d) %s %s: k = %d' % (i, kType[o[0]], unitL2[o[1]], k))
                bCubed[o].append(kClustering(k, features, kMeans=o[0], l2=o[1]))

        for o in options:
            plotGraph(np.asmatrix(bCubed[o]), '(' + str(i) + ')' + kType[o[0]] + ' Clustering ' + unitL2[o[1]])
        bCubed100.append(bCubed)
