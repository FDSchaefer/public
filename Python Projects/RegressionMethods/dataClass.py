import os, sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class dataMethods():

    def get_df_color(self,color):
        if color == "white":
            return self._df_white
        elif color == "red":
            return self._df_red
        else:
            raise RuntimeError(f"Unkown Color")

    @property
    def get_df_raw(self):
        return self._df_raw

    @property
    def get_df_scaled(self):
        return self._df_scaled

    @property
    def get_scaler(self):
        return self._scaler

    def __init__(self, data_path):
        self._data_path = data_path
        
        self._df_white = pd.read_csv(data_path + "\winequality-red.csv",sep=";")
        self._df_white["type"] = "white"
        print("Loaded: White Wine Data\tLength: " + str(len(self._df_white)))

        self._df_red = pd.read_csv(data_path + "\winequality-white.csv",sep=";")
        self._df_red["type"] = "red"
        print("Loaded: Red Wine Data\tLength: " + str(len(self._df_red)))

        self._df_raw = pd.concat([self._df_white,self._df_red])
        print("\nUnified Data, with Length: " + str(len(self._df_raw)))

    def digitizeType(self,data,inverse=False):
        if not inverse:
            # Digitize the "type" feature. 
            data.loc[data["type"] == "white","type"] = 1
            data.loc[data["type"] == "red","type"] = 0
        if inverse:
            # Un-Digitize the "type" feature. 
            data.loc[data["type"] == 1,"type"] = "white"
            data.loc[data["type"] == 0,"type"] = "red"
        print("Completed")
        
    def scaleData(self,data,scaler_type = "standard"):
        if scaler_type == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaler_type == "maxmin":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()

        scaler.fit(data)
        self._df_scaled = scaler.transform(data)
        if isinstance(data,pd.DataFrame):
            self._df_scaled = pd.DataFrame(self._df_scaled, columns=data.columns)
        self._scaler = scaler

        print("Scaled data using method: "  + scaler_type)

    def elbowMethod(self,data,n_clusters = range(1,5)):

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        inertias = []
        for i in n_clusters:
            km = KMeans(n_clusters=i, random_state=12, n_init = 100)
            km.fit_predict(data)
            inertias.append(km.inertia_)

        plt.plot(n_clusters, inertias, 'x--')
        plt.xlabel('k')
        plt.ylabel('Cluster Inertias')
        plt.title('Determining Ideal Number Of Clusters')
        plt.show()


## dataMethods Plotting Functions 

def tileHistogram(data):
    # Histograms of all core features 
    fig,ax = plt.subplots(4,3, figsize=(15, 15))

    i,j = 0,0
    for feature in data.columns[:-1]:
        sns.histplot(data = data, x = feature, hue= "type", ax = ax[i,j],stat= "count", kde=True)
        if j < 2:
            j += 1
        else:
            j = 0
            i += 1

def correlationPlot(data):
    ## Correlation Matrix of all core features (red and white wine)
    fig,ax = plt.subplots(3,1, figsize=(10,20))

    corr_white = data.corr()
    corr_red = data.corr()
    corr_join = data.corr()

    sns.heatmap(corr_white, annot=True, ax = ax[0],cmap='coolwarm')
    ax[0].set_title("Feature Correlations - White Wine")
    sns.heatmap(corr_red, annot=True, ax = ax[1],cmap='coolwarm')
    ax[1].set_title("Feature Correlations - Red Wine")
    sns.heatmap(corr_join, annot=True, ax = ax[2],cmap='coolwarm')
    ax[2].set_title("Feature Correlations - All Wine")
    fig.tight_layout()