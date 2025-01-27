import math
import sys
import csv

import sklearn
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import neural_network
import numpy as np
import time
from svm import SVM
import cupy as xp

from sklearn.neighbors import KNeighborsClassifier

MIN_USAGE = 2

def print_out(file, msg):
    print(msg)
    file.write(msg)
    file.write("\n")

def normlaize_d(data):
    norm_data = np.sum(abs(data), axis=1)
    norm_data = np.where(norm_data == 0, 1, norm_data)
    data /= norm_data[: ,None]
    return data

# -----------------------------
# punctul 2
def normalize_data(train_data, test_data,validation_1,validation_2, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer('l1')
    elif type == 'l2':
        scaler = preprocessing.Normalizer('l2')
    elif type == 'quantile_normal':
        scaler = preprocessing.QuantileTransformer(output_distribution='normal')

    if scaler is not None:
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        # validation_1 = scaler.transform(validation_1)
        validation_2 = scaler.transform(validation_2)
    else:
        print_out(f,"Invalid scaling method - no scaling has been done")

    return train_data, test_data, validation_1, validation_2


class BagOfWords:

    def __init__(self):
        self.dictData = dict()
        self.word_list = []

    def build_vocabulary(self, data):
        index = 0
        counter_dict = dict()
        for sentence in data:
            for word in sentence:
                if word not in counter_dict:
                    counter_dict[word] = 1
                else:
                    counter_dict[word] += 1

        for sentence in data:
            for word in sentence:
                if word not in self.dictData and counter_dict[word] >= MIN_USAGE:
                    self.dictData[word] = index
                    self.word_list.append(word)
                    index += 1
        self.word_list = np.array(self.word_list)
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
    for sentence in data:
        words = list(filter(None, sentence.replace(',', ' ').replace('.', ' ').replace(')', ' ').replace('(', ' ')
                            .replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '')
                            .replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '')
                            .replace('‘', ' ').replace(';', ' ').replace('%', '').replace('“', ' ').replace('_', ' ')
                            .replace('@', '').replace('”', '').replace('…', '').replace('{', '').replace('}', '')
                            .replace('\'', ' ').replace('„', ' ').replace('’', ' ').replace('<', ' ').replace('>', ' ')
                            .replace('"', ' ').replace('|', '').replace('', '').replace('*', '').replace('«', '')
                            .replace('»', '').replace(':', ' ').replace('►', ' ').replace('$NE$', '').replace('=', ' ')
                            .replace('˝', ' ').replace('$', ' ').replace('\\xad', '').replace('″', ' ')
                            .replace('‚', ' ').upper()
                            .replace('Ş', 'S').replace('Ã', 'A').replace('Â','A').replace('Ţ','T')
                            .replace('Ț', 'T').replace('Ș', 'S').replace('Ă', 'A')
                            .split(" ")))
        new_words = []
        for word in words:
            if word[0] == 'Î':
                word = 'I' + word[1:]
            word = word.replace('Î','A')
            if len(word) < 3 or len(word) > 13:
                kk = 1
            else:
                new_words.append(word)

        new_data.append(new_words)
    return new_data


#logs
f = open('logs.txt', 'a+')

# timing
start_time = time.time()

# load data

train_data = np.genfromtxt('data/train_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8')['Text']
train_labels = np.loadtxt('data/train_labels.txt')[:, 1]

validation_data1 = np.genfromtxt('data/validation_source_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'),
                                 encoding='utf-8')['Text']
validation_data2 = np.genfromtxt('data/validation_target_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'),
                                 encoding='utf-8')['Text']

validation_labels1 = np.loadtxt('data/validation_source_labels.txt')[:, 1]
validation_labels2 = np.loadtxt('data/validation_target_labels.txt')[:, 1]

train_data = np.concatenate((train_data, validation_data1))
train_labels = np.concatenate((train_labels, validation_labels1))

# train_data = np.concatenate((train_data, validation_data2))
# train_labels = np.concatenate((train_labels, validation_labels2))

validation_data1 = None
validation_labels1 = None

test_data = np.genfromtxt('data/test_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8')

print_out(f, "Done opening data")
print_out(f, "--- %s seconds ---" % (time.time() - start_time))

# prepare data
train_sentences = np.array(prep_data(train_data))
# validation_sentences1 = np.array(prep_data(validation_data1))
validation_sentences2 = np.array(prep_data(validation_data2))
test_sentences = np.array(prep_data(test_data['Text']))

train_data = None
validation_data1 = None
validation_data2 = None
# test_data = None

# create class
bagofwords = BagOfWords()
# build train dict
dict_data = bagofwords.build_vocabulary(train_sentences)

# indexes = np.argsort(bagofwords.word_list)
# print(bagofwords.word_list[indexes[:1000]])
print_out(f, "Lungime dictionar:" + str(len(dict_data)))
print_out(f, "--- %s seconds ---" % (time.time() - start_time))

# get features
features_train = bagofwords.get_features(train_sentences)
# features_validation1 = bagofwords.get_features(validation_sentences1)
features_validation2 = bagofwords.get_features(validation_sentences2)
features_test = bagofwords.get_features(test_sentences)

train_sentences = None
validation_sentences1 = None
validation_sentences2 = None
test_sentences = None

print_out(f,"Done features")
print_out(f,"--- %s seconds ---" % (time.time() - start_time))

norm_method = 'l1'
# normalized_train = normlaize_d(features_train)
# features_train = None
# normalized_test = normlaize_d(features_test)
# features_test = None
# normalized_validation2 = normlaize_d(features_validation2)
# features_validation2 = None
normalized_train,normalized_test,normalized_validation1,normalized_validation2 \
    = normalize_data(features_train,features_test,[],features_validation2, norm_method)

features_train = None
features_test = None
features_validation1 =None
features_validation2 = None

print_out(f,"Done normalization - METHOD: " + norm_method)
print_out(f,"--- %s seconds ---" % (time.time() - start_time))


# SVM model
# C_vals = [1, 5, 10,20,30,40,50,80]
C_vals = [10]

print_out(f, "C:" + str(C_vals))
accuracy1 = np.zeros(len(C_vals))
accuracy2 = np.zeros(len(C_vals))
for i in range(len(C_vals)):
    C_param = C_vals[i]
    # svm_model = svm.SVC(C=C_param, kernel='rbf', gamma='scale', verbose=1)  # kernel rbf # nu termina
    # svm_model = svm.LinearSVC(C=C_param, verbose=0, max_iter=15000)  # cel mai bun pe l1 momentan
    # svm_model = KNeighborsClassifier(n_neighbors=C_param, algorithm='auto', leaf_size=30, n_jobs=-1)  # pare slab si nu termina ever cu toate datele
    # svm_model = tree.DecisionTreeClassifier() # slab,dar termina ~30 min
    svm_model = neural_network.MLPClassifier(max_iter=200, verbose=1, hidden_layer_sizes=(20, 10), learning_rate_init = 0.01)

    # gpu
    # svm_model = SVM(kernel='linear', kernel_params={}, n_folds=3, use_optimal_lambda=True)
    # x_train = xp.asarray(normalized_train)
    # y_train = xp.asarray(train_labels)
    # svm_model.fit(x_train, y_train)

    svm_model.fit(normalized_train, train_labels)  # train
    print_out(f, "Done fitting")
    print_out(f, "--- %s seconds ---" % (time.time() - start_time))

    # words = np.array(bagofwords.word_list)
    # weights = np.squeeze(svm_model.coef_)
    # indexes = np.argsort(weights)
    # print("the most RO words are", words[indexes[-100:]])
    # print("the most MOLD words are", words[indexes[:100]])

    # predicted_val1_labels = svm_model.predict(normalized_validation1)  # predict
    predicted_val2_labels = svm_model.predict(normalized_validation2)  # predict

    print_out(f, "Done predict validation")
    print_out(f, "--- %s seconds ---" % (time.time() - start_time))

    if len(C_vals) == 1:
        predicted_test_labels = svm_model.predict(normalized_test)  # predict
        # write to file
        w = csv.writer(open("predictii_" + str(C_param) + "_" + norm_method + ".csv", "w", newline=''))
        w.writerow(["id", "label"])
        for i in range(len(predicted_test_labels)):
            w.writerow([test_data['ID'][i], predicted_test_labels[i].astype(int)])
        print_out(f, "Done predict test")
        print_out(f, "--- %s seconds ---" % (time.time() - start_time))

    if len(C_vals) == 1:
        # print_out(f, "Accuracy1: " + str(sklearn.metrics.accuracy_score(predicted_val1_labels, validation_labels1)))
        print_out(f, "Accuracy2: " + str(sklearn.metrics.accuracy_score(predicted_val2_labels, validation_labels2)))
    else:
        # accuracy1[i] = sklearn.metrics.accuracy_score(predicted_val1_labels, validation_labels1)
        accuracy2[i] = sklearn.metrics.accuracy_score(predicted_val2_labels, validation_labels2)
        print_out(f, "Accuracy1: " + str(accuracy1[i]))
        print_out(f, "Accuracy2: " + str(accuracy2[i]))

    # print_out(f, "F1-Score1: " + str(sklearn.metrics.f1_score(predicted_val1_labels, validation_labels1,average='macro')))
    print_out(f, "F1-Score2: " + str(sklearn.metrics.f1_score(predicted_val2_labels, validation_labels2,average='macro')))
    print_out(f, "---------------------------\n")

print("DONE")
print("--- %s seconds ---" % (time.time() - start_time))
