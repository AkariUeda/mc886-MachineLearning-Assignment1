import numpy as np
import pandas as pd


def normalize_features(features):
  for f in range(1, features.shape[1]):
    mini = features[:,f].min()
    features[:,f] = np.subtract(features[:,f],mini)
    max_value = features[:,f].max()
    features[:,f] = 1+np.divide(features[:,f],mini+max_value)
  return features

def replace_dummies(train_features):
    train_features.cut[train_features['cut'] == 'Fair'] = 1
    train_features.cut[train_features['cut'] == 'Good'] = 2
    train_features.cut[train_features['cut'] == 'Very Good'] = 3
    train_features.cut[train_features['cut'] == 'Premium'] = 4
    train_features.cut[train_features['cut'] == 'Ideal'] = 5
    train_features.clarity[train_features['clarity'] == 'I1'] = 1
    train_features.clarity[train_features['clarity'] == 'SI2'] = 2
    train_features.clarity[train_features['clarity'] == 'SI1'] = 3
    train_features.clarity[train_features['clarity'] == 'VS2'] = 4
    train_features.clarity[train_features['clarity'] == 'VS1'] = 5
    train_features.clarity[train_features['clarity'] == 'VVS2'] = 6
    train_features.clarity[train_features['clarity'] == 'VVS1'] = 7
    train_features.clarity[train_features['clarity'] == 'IF'] = 8
    train_features.color[train_features['color'] == 'D'] = 7
    train_features.color[train_features['color'] == 'E'] = 6
    train_features.color[train_features['color'] == 'F'] = 5
    train_features.color[train_features['color'] == 'G'] = 4
    train_features.color[train_features['color'] == 'H'] = 3
    train_features.color[train_features['color'] == 'I'] = 2
    train_features.color[train_features['color'] == 'J'] = 1

def main():
    #Reading the dataset into the training and validation sets.
    dataset = pd.read_csv('diamonds.csv')
    dataset = dataset.sample(frac=1)
    dataset_labels = dataset["price"]
    drop_features = ['Unnamed: 0', 'price']
    dataset_features = dataset.drop(drop_features, axis=1)
    replace_dummies(dataset_features)

    #features
    train_features = dataset_features[0:36679]
    valid_features = dataset_features[36679:45849]
    test_features = dataset_features[45849:len(dataset_features)]

    #adicionando o X0 = 1
    x0 = np.ones((len(train_features),1))
    train_features = np.append(x0, train_features, axis=1)
    x0 = np.ones((len(valid_features),1))
    valid_features = np.append(x0, valid_features, axis=1)
    x0 = np.ones((len(test_features),1))
    test_features = np.append(x0, test_features, axis=1)
    #targets
    train_labels = np.array(dataset_labels[0:36679])
    valid_labels = np.array(dataset_labels[36679:45849])
    test_labels = np.array(dataset_labels[45849:len(dataset_labels)])
    normalize_features(train_features)
    normalize_features(valid_features)
    normalize_features(test_features)
    
    np.savetxt("train_features.csv",train_features,delimiter=",")
    np.savetxt("train_labels.csv",train_labels,delimiter=",")
    np.savetxt("valid_features.csv",valid_features,delimiter=",")
    np.savetxt("valid_labels.csv",valid_labels,delimiter=",")
    np.savetxt("test_features.csv",test_features,delimiter=",")
    np.savetxt("test_labels.csv",test_labels,delimiter=",")


if __name__ == "__main__":
    main()
