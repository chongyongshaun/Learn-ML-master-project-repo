from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
# print(iris.metadata) 
  
# variable information 
print(iris.variables) 
print(X.head())
print(y["class"].unique())

# combine features and targets into one DataFrame with target column(s) last
df = pd.concat([X.reset_index(drop=True), pd.DataFrame(y).reset_index(drop=True)], axis=1)
#validating the combined df
print(df.head())
print(df.columns)

# 60% train, 20% valid, 20% test
#df.sample() shuffles the data, frac=1 means return all rows in random order 
train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

#check for imbalance in target classes
print("Training set class distribution:\n", train["class"].value_counts()) #iris dataset is quite balanced no need for over/under sampling

def plot_feature_distribution(data, feature, target):
    plt.figure(figsize=(10, 6))
    for cls in data[target].unique():
        subset = data[data[target] == cls]
        plt.hist(subset[feature], alpha=0.5, label=f'Class {cls}', bins=15)
    plt.title(f'Distribution of {feature} by {target}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
#plot all 4 features
# features = X.columns
# for feature in features:
#     plot_feature_distribution(df, feature, "class")

def scale_dataset(df, oversample = False):
    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if oversample:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        X_scaled, Y = smote.fit_resample(X_scaled, Y)

    data = np.hstack((X_scaled, Y.reshape(-1, 1)))
    return data, X_scaled, Y

train, X_train, y_train = scale_dataset(train, oversample=False)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

#svm Classifier
from sklearn.svm import SVC
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
