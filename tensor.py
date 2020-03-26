import time

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
from sklearn import preprocessing

MIN_USAGE = 1

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
        validation_1 = scaler.transform(validation_1)
        validation_2 = scaler.transform(validation_2)
    else:
        print_out(f,"Invalid scaling method - no scaling has been done")

    return train_data, test_data, validation_1, validation_2


class BagOfWords:

    def __init__(self):
        self.dictData = dict()
        self.word_list = []
        self.nr_words = 0

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
                    self.dictData[word] = self.nr_words
                    self.word_list.append(word)
                    self.nr_words += 1
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





def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()


def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec


def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)


f = open('logs-tensor.txt', 'a+')
# timing
start_time = time.time()

# load data
train_data = np.genfromtxt('data/train_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8')['Text']
train_labels = np.loadtxt('data/train_labels.txt')[:, 1]
validation_data1 = np.genfromtxt('data/validation_source_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'),
                                 encoding='utf-8')['Text']
validation_data2 = np.genfromtxt('data/validation_target_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'),
                                 encoding='utf-8')
validation_labels1 = np.loadtxt('data/validation_source_labels.txt')[:, 1]
validation_labels2 = np.loadtxt('data/validation_target_labels.txt')[:, 1]
# train_data = np.concatenate((train_data, validation_data1))
# train_labels = np.concatenate((train_labels, validation_labels1))
# validation_data1 = None
# validation_labels1 = None
test_data = np.genfromtxt('data/test_samples.txt', delimiter='\t', dtype=None, names=('ID', 'Text'), encoding='utf-8')

train_labels -= 1
validation_labels1 -= 1
validation_labels2 -= 1

print_out(f, "Done opening data")
print_out(f, "--- %s seconds ---" % (time.time() - start_time))

# prepare data
train_sentences = np.array(prep_data(train_data))
validation_sentences1 = np.array(prep_data(validation_data1))
validation_sentences2 = np.array(prep_data(validation_data2['Text']))
test_sentences = np.array(prep_data(test_data['Text']))

test_data = None
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
features_validation1 = bagofwords.get_features(validation_sentences1)
features_validation2 = bagofwords.get_features(validation_sentences2)
features_test = bagofwords.get_features(test_sentences)

train_sentences = None
validation_sentences1 = None
validation_sentences2 = None
test_sentences = None

print_out(f,"Done features")
print_out(f,"--- %s seconds ---" % (time.time() - start_time))

norm_method = 'standard'
normalized_train,normalized_test,normalized_validation1,normalized_validation2 \
    = normalize_data(features_train,features_test,features_validation1,features_validation2, norm_method)

# normalized_train = features_train
# normalized_test = features_test
# normalized_validation1 = features_validation1
# normalized_validation2 = features_validation2

features_train = None
features_test = None
features_validation1 =None
features_validation2 = None

print_out(f,"Done normalization - METHOD: " + norm_method)
print_out(f,"--- %s seconds ---" % (time.time() - start_time))



# train_examples, test_examples = dataset['train'], dataset['test']
# encoder = info.features['text'].encoder
# print('Vocabulary size: {}'.format(encoder.vocab_size))
#
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64
# train_dataset = (train_examples
#                  .shuffle(BUFFER_SIZE)
#                  .padded_batch(BATCH_SIZE, padded_shapes=([None],[])))
#
# test_dataset = (test_examples
#                 .padded_batch(BATCH_SIZE,  padded_shapes=([None],[])))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=bagofwords.nr_words, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.0, nesterov=False),
              metrics=['accuracy'])

model.fit(normalized_train, train_labels,
          validation_data=(normalized_validation2,validation_labels2)
          ,epochs=10 ,batch_size=1000)


test_loss, test_acc = model.evaluate(normalized_validation1,validation_labels1)
predictions = model.predict(normalized_test)
print(predictions)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))