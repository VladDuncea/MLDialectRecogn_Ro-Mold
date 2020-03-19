import math
import sys
import csv

import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time


def calc_accuracy(predicted, true):
    accuracy = (predicted == true).mean()
    return accuracy


# -----------------------------
# punctul 2
def normalize_data(data, type=None):
    if type == 'standard':
        stand = preprocessing.StandardScaler()
        stand.fit(data)
        data = stand.transform(data)
    elif type == 'min_max':
        x = 1 + 1
    # posibil sa trebuiasca sa fac l1,l2 cu numpy !!!
    elif type == 'l1':
        val = np.sum(abs(data), axis=1)
        norm_data = np.zeros(len(data))
        for i in range(len(data)):
            for j in range(len(data[i])):
                norm_data[i] += abs(data[i, j])
            if norm_data[i] != 0:
                data[i] /= norm_data[i]
    elif type == 'l2':
        norm_data = np.sqrt(np.sum(data ** 2, axis=1))
        norm_data = np.where(norm_data == 0, 1, norm_data)
        data /= norm_data[:, None]
    return data


class BagOfWords:

    def __init__(self):
        self.dictData = dict()
        self.word_list = []

    def build_vocabulary(self, data):
        index = 0
        for sentence in data:
            for word in sentence:
                if word not in self.dictData:
                    self.dictData[word] = index
                    self.word_list.append(word)
                    index += 1
        return self.dictData

    def get_features(self, data):
        features = np.zeros((len(data), len(self.dictData)))
        for i in range(len(data)):
            for word in data[i]:
                if word in self.dictData:
                    features[i, self.dictData[word]] += 1
        return features


def prep_data(data):
    new_data = []
    index = 0
    for sentence in data:
        words = list(filter(None, sentence.replace(',', '').replace('.', '').replace(')', '').replace('(', '')
                                    .replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4','')
                                    .replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '')
                                    .replace('‘','').replace(';', '').replace('%', '').replace('“', '').replace('_', '')
                                    .replace('@','').replace('”', '').replace('…', '').replace('{', '').replace('}', '')
                                    .replace('\'', '').replace('„','').replace('’', '')
                                    .replace('"', '').replace('|', '').replace('', '').replace('*', '').replace('«','')
                                    .replace('»', '').replace(':', '').replace('$NE$', '').upper().split(" ")))
        for word in words:
            if len(word)<2 or len(word)>20:
                words.remove(word)
        new_data.append(words)
    return new_data


# timing

start_time = time.time()

# load data

train_data = np.genfromtxt('data/train_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8')
train_labels = np.loadtxt('data/train_labels.txt')

validation_data1 = np.genfromtxt('data/validation_source_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'),
                                 encoding='utf-8')
validation_data2 = np.genfromtxt('data/validation_target_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'),
                                 encoding='utf-8')
validation_labels1 = np.loadtxt('data/validation_source_labels.txt')[:,1]
validation_labels2 = np.loadtxt('data/validation_target_labels.txt')[:,1]

test_data = np.genfromtxt('data/test_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8')

print("Done opening data")
print("--- %s seconds ---" % (time.time() - start_time))

# prepare data
train_sentences = prep_data(train_data['Text'])
# print(train_sentences[:])
validation_sentences1 = prep_data(validation_data1['Text'])
validation_sentences2 = prep_data(validation_data2['Text'])
test_sentences = prep_data(test_data['Text'])

# create class
bagofwords = BagOfWords()
# build train dict
dict_data = bagofwords.build_vocabulary(train_sentences)

# words = 0
# for key, val in bagofwords.dictData.items():
#     if words == 15:
#         print(key)
#         words =0
#     print(key, end =" ")
#     words+=1

print("Lungime dictionar:" + str(len(dict_data)))
print("--- %s seconds ---" % (time.time() - start_time))

# get features
features_train = bagofwords.get_features(train_sentences)
features_validation1 = bagofwords.get_features(validation_sentences1)
features_validation2 = bagofwords.get_features(validation_sentences2)
# features_test = bagofwords.get_features(test_sentences)

print("Done features")
print("--- %s seconds ---" % (time.time() - start_time))


normalized_train = normalize_data(features_train, "l2")
normalized_validation1 = normalize_data(features_validation1, "l2")
normalized_validation2 = normalize_data(features_validation2, "l2")
# normalized_test = normalize_data(features_test, "l2")

print("Done normalization")
print("--- %s seconds ---" % (time.time() - start_time))

# print(normalized_train)
# print(normalized_validation)
# print(normalized_test)

# SVM model
Neigh_vals = [1, 2, 3, 5]
# Neigh_vals = [15]
accuracy1 = np.zeros(len(Neigh_vals))
accuracy2 = np.zeros(len(Neigh_vals))
for i in range(len(Neigh_vals)):
    Neigh_param = Neigh_vals[i]
    kn_model = KNeighborsClassifier(n_neighbors=Neigh_param,algorithm='auto', leaf_size=30, n_jobs=-1)  # kneigh classifier
    # svm_model = svm.LinearSVC(C=C_param, verbose=0, max_iter=10000)  # kernel liniar
    kn_model.fit(normalized_train, train_labels[:, 1])  # train
    print("Done fitting")
    print("--- %s seconds ---" % (time.time() - start_time))

    predicted_val1_labels = kn_model.predict(normalized_validation1)  # predict
    predicted_val2_labels = kn_model.predict(normalized_validation2)  # predict
    # predicted_val1_labels = np.round(np.clip(predicted_val1_labels,1,2))
    # predicted_val2_labels = np.round(np.clip(predicted_val2_labels,1,2))

    print("Done predict validation")
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Neighbours: " + str(Neigh_param) + "Leaf size: " + str(30))
    if len(Neigh_vals) == 1:
        print("Accuracy1: " + str(calc_accuracy(predicted_val1_labels, validation_labels1)))
        print("Accuracy2: " + str(calc_accuracy(predicted_val2_labels, validation_labels2)))
    else:
        accuracy1[i] = calc_accuracy(predicted_val1_labels, validation_labels1)
        accuracy2[i] = calc_accuracy(predicted_val2_labels, validation_labels2)
        print("Accuracy1: " + str(accuracy1[i]))
        print("Accuracy2: " + str(accuracy2[i]))
    print("F1-Score1: " + str(sklearn.metrics.f1_score(predicted_val1_labels, validation_labels1)))
    print("F1-Score2: " + str(sklearn.metrics.f1_score(predicted_val2_labels, validation_labels2)))

    if len(Neigh_vals) == 1:
        predicted_test_labels = kn_model.predict(normalized_test)  # predict
        # write to file
        w = csv.writer(open("predictii" + str(Neigh_vals) + ".csv", "w", newline=''))
        w.writerow(["id", "label"])
        for i in range(len(predicted_test_labels)):
            w.writerow([test_data['ID'][i], predicted_test_labels[i].astype(int)])
        print("Done predict test")
        print("--- %s seconds ---" % (time.time() - start_time))
    print("---------------------------\n")

print("DONE")
print("--- %s seconds ---" % (time.time() - start_time))