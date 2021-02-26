import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import pyplot as plt
import math


def set1():
    df = pd.read_csv("https://raw.githubusercontent.com/aharonk/Pandas-ML/main/Blank_Car_Info.csv") # 2
    df.columns = ["numDoors", "numCylinders", "mpgHighway", "mpgStreets", "currentMileage",
                  "avgMileagePerTuneUp", "qualityRating"] # 3
    df.isnull().sum() # 4

    kmeans = KMeans(n_clusters=4).fit(df)

    cc = kmeans.cluster_centers_
    print(cc) # 5
    print()

    print(kmeans.predict([[2, 4, 500, 300, 150, 760, .6]])) # 6
    print()

    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    xt = est.fit_transform(df)
    svm = SVC().fit(xt[:, :-1], xt[:, -1])
    newcar = svm.predict([[2, 3, 100, 900, 500, 45]]) # 7
    print(newcar)


def set2():
    df = pd.read_csv("https://raw.githubusercontent.com/aharonk/Pandas-ML/main/5%20year%20prices.csv") # 3
    df.describe() # 4

    covariance = df.cov() 
    correlation = df.corr()

    plt.rcParams.update({'font.size': 12, 'figure.figsize': (20, 20)})
    f3 = sns.heatmap(correlation, annot=True)
    plt.show() # 5

    numdays = len(df.index)

    gme = df.GME.to_numpy()
    model = LinearRegression().fit(np.arange(numdays).reshape((numdays, 1)), gme)
    print(model.predict([[numdays], [numdays+1], [numdays+2], [numdays+3], [numdays+4]])) # 6

    df = df.assign(GMELogChange=pd.Series([math.log(gme[x])/math.log(gme[x-1]) for x in range(1, numdays)])) # 7


if __name__ == '__main__':
    set1()
    set2()
