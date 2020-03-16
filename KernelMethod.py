import codecs
import math
from sklearn import preprocessing
from sklearn import svm
import numpy as np


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
        x = 1+1
    # posibil sa trebuiasca sa fac l1,l2 cu numpy !!!
    elif type == 'l1':
        for i in range(len(data)):
            norm_data = 0
            for j in range(len(data[i])):
                norm_data += abs(data[i, j])
            data[i] /= norm_data
    elif type == 'l2':
        for i in range(len(data)):
            norm_data = 0
            for j in range(len(data[i])):
                norm_data += math.sqrt((data[i, j])**2)
            if norm_data != 0:
                data[i] /= norm_data
    return data


class BagOfWords:

    def __init__(self):
        self.dictData = dict()
        self.word_list = []

    def build_vocabulary(self, data):
        index =0
        for sentence in data:
            for word in sentence:
                if (word != "$NE$") and (word not in self.dictData):
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


# load data
# np_load_old = np.load
# modify the default parameters of np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
encoded_train = codecs.open('data/train_samples.txt', encoding='utf-8')
train_sentences = np.genfromtxt(encoded_train, delimiter='\t', dtype=None, names=('ID', 'Text'),encoding='None')
train_labels = np.loadtxt('data/train_labels.txt')

encoded_validation = codecs.open('data/train_samples.txt', encoding='utf-8')
validation_sentences = np.genfromtxt(encoded_validation, delimiter='\t', dtype=None, names=('ID', 'Text'),encoding='None')
validation_labels = np.loadtxt('data/validation_source_labels.txt')

encoded_test = codecs.open('data/train_samples.txt', encoding='utf-8')
test_sentences = np.genfromtxt(encoded_test, delimiter='\t', dtype=None, names=('ID', 'Text'),encoding='None')

# np.load = np_load_old


# create class
bagofwords = BagOfWords()
# build train dict
dict_data = bagofwords.build_vocabulary(train_sentences)
print("Lungime dictionar:" + str(len(dict_data)))

# get features
features_train = bagofwords.get_features(train_sentences['Text'][:1000])
features_validation = bagofwords.get_features(validation_sentences['Text'][:1000])
features_test = bagofwords.get_features(test_sentences['Text'])

normalized_train = normalize_data(features_train, "l2")
normalized_validation = normalize_data(features_validation, "l2")
normalized_test = normalize_data(features_test, "l2")

# print(normalized_train)
# print(normalized_validation)
# print(normalized_test)

# SVM model
C_param = 1
svm_model = svm.SVC(C_param, "linear") # kernel liniar
svm_model.fit(normalized_train, train_labels[:1000, 1]) # train
predicted_val_labels = svm_model.predict(normalized_validation) # predict
predicted_test_labels = svm_model.predict(normalized_test) # predict

np.savetxt('predictii.txt', predicted_test_labels.astype(int))  # salveaza predictiile in fisier

print("Accuracy: " + str(calc_accuracy(predicted_val_labels, validation_labels[:1000, 1])))

