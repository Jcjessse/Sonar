import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/Toxi1.csv')
    print(dt_heart.head(5))
    
    dt_features = dt_heart.drop(['%Toxicos'], axis=1)
    dt_target = dt_heart['%Toxicos']
    
    imputer = SimpleImputer(strategy='mean')
    dt_features = imputer.fit_transform(dt_features)
    
    dt_features = StandardScaler().fit_transform(dt_features)
    dt_features = pd.DataFrame(dt_features, columns=dt_heart.drop(['%Toxicos'], axis=1).columns)
    
    # Eliminar características constantes
    dt_features = dt_features.loc[:, dt_features.var() != 0]
    
    # Normalizar los datos utilizando MinMaxScaler
    scaler = MinMaxScaler()
    dt_features = scaler.fit_transform(dt_features)
    
    # Discretizar los datos utilizando KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    dt_features = discretizer.fit_transform(dt_features)
    
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30, random_state=42)
    
    print(X_train.shape)
    print(y_train.shape)
    
    pca = PCA(n_components=3)
    pca.fit(X_train)
    
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()
    
    logistic = LogisticRegression(solver='lbfgs')
    
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA:", logistic.score(dt_test, y_test))
    
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA:", logistic.score(dt_test, y_test))

