import os
import json
import time
import pprint
import argparse
import datetime
import subprocess
import numpy as np
import seaborn as sns
import keras.callbacks as ckbs
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import mlflow
from dl_models import DlModels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
print("done")

# sunucuda calismak icin
plt.switch_backend('agg')

pp = pprint.PrettyPrinter(indent=4)

TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}

class PhishingUrlDetection:

    def __init__(self, epoch, architecture):
        self.params = {'loss_function': 'categorical_crossentropy',
                       'optimizer': 'adam',
                       'sequence_length': 200,
                       'batch_train': 5000,
                       'batch_test': 5000,
                       'categories': ['phishing', 'legitimate'],
                       'char_index': None,
                       'epoch': epoch,
                       'embedding_dimension': 100,
                       'architecture': architecture,
                       'result_dir': "/mnt/c/Users/dell/PycharmProjects/pro/test_results/",
                       'dataset_dir': "/mnt/c/Users/dell/PycharmProjects/pro/dataset/small_dataset/"}

        if not os.path.exists(self.params['result_dir']):
            os.makedirs(self.params['result_dir'])
            print("Directory ", self.params['result_dir'], " Created ")
        else:
            print("Directory ", self.params['result_dir'], " already exists")

        self.ml_plotter = Plotter()
        self.dl_models = DlModels(self.params['categories'], self.params['embedding_dimension'], self.params['sequence_length'])

    def model_sum(self, x):
        try:
            TEST_RESULTS['hiperparameter']["model_summary"] += x
        except:
            TEST_RESULTS['hiperparameter']["model_summary"] = x

    def load_and_vectorize_data(self):
        print("data loading")
        train = [line.strip() for line in open("{}/train.txt".format(self.params['dataset_dir']), "r").readlines()]
        test = [line.strip() for line in open("{}/test.txt".format(self.params['dataset_dir']), "r").readlines()]
        val = [line.strip() for line in open("{}/val.txt".format(self.params['dataset_dir']), "r").readlines()]

        mlflow.log_param("samples_train", len(train))
        mlflow.log_param("samples_test", len(test))
        mlflow.log_param("samples_val", len(val))
        mlflow.log_param("samples_overall", len(train) + len(test) + len(val))
        mlflow.log_param("dataset", self.params['dataset_dir'])

        raw_x_train = [line.split("\t")[1] for line in train]
        raw_y_train = [line.split("\t")[0] for line in train]

        raw_x_val = [line.split("\t")[1] for line in val]
        raw_y_val = [line.split("\t")[0] for line in val]

        raw_x_test = [line.split("\t")[1] for line in test]
        raw_y_test = [line.split("\t")[0] for line in test]

        tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        tokener.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
        open("{}/char_index.json".format(self.params['result_dir']), "w").write(json.dumps(tokener.word_index))
        print("char index saved.")
        self.params['char_index'] = tokener.word_index

        x_train = np.asanyarray(tokener.texts_to_sequences(raw_x_train))
        x_val = np.asanyarray(tokener.texts_to_sequences(raw_x_val))
        x_test = np.asanyarray(tokener.texts_to_sequences(raw_x_test))

        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        y_train = np_utils.to_categorical(encoder.transform(raw_y_train), num_classes=len(self.params['categories']))
        y_val = np_utils.to_categorical(encoder.transform(raw_y_val), num_classes=len(self.params['categories']))
        y_test = np_utils.to_categorical(encoder.transform(raw_y_test), num_classes=len(self.params['categories']))

        print("Data are loaded.")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def dl_algorithm(self, x_train, y_train, x_val, y_val, x_test, y_test):
        x_train = sequence.pad_sequences(x_train, maxlen=self.params['sequence_length'])
        x_test = sequence.pad_sequences(x_test, maxlen=self.params['sequence_length'])
        x_val = sequence.pad_sequences(x_val, maxlen=self.params['sequence_length'])

        print("train sequences: {}  |  test sequences: {} | val sequences: {}\n"
              "x_train shape: {}  |  x_test shape: {} | x_val shape: {}\n"
              "Building Model....".format(len(x_train), len(x_test), len(x_val), x_train.shape, x_test.shape, x_val.shape))

        # Build Deep Learning Architecture
        if self.params['architecture'] == "ann":
            model = self.dl_models.ann_complex(self.params['char_index'])
            mlflow.log_param('architecture', "ann")
        elif self.params['architecture'] == "cnn":
            model = self.dl_models.cnn_complex(self.params['char_index'])
            mlflow.log_param('architecture', "cnn")
        elif self.params['architecture'] == "rnn":
            model = self.dl_models.rnn_complex(self.params['char_index'])
            mlflow.log_param('architecture', "rnn")
        else:
            model = self.dl_models.cnn_complex3(self.params['char_index'])
            mlflow.log_param('architecture', "cnn3")

        model.compile(loss=self.params['loss_function'], optimizer=self.params['optimizer'], metrics=['accuracy'])

        model.summary()
        model.summary(print_fn=lambda x: self.model_sum(x + '\n'))

        hist = model.fit(x_train, y_train,
                         batch_size=self.params['batch_train'],
                         epochs=self.params['epoch'],
                         shuffle=True,
                         validation_data=(x_val, y_val))
                         #callbacks=[CustomCallBack()])

        t = time.time()
        score, acc = model.evaluate(x_test, y_test, batch_size=self.params['batch_test'])

        mlflow.log_metric("test_time", float("{0:.2f}".format(round(time.time() - t, 2))))
        TEST_RESULTS['test_result']['test_time'] = time.time() - t

        y_test = list(np.argmax(np.asanyarray(np.squeeze(y_test), dtype=int).tolist(), axis=1))
        y_pred = model.predict(x_test, batch_size=self.params['batch_test'], verbose=1).tolist()
        y_pred = list(np.argmax(np.asanyarray(np.squeeze(y_pred), dtype=int).tolist(), axis=1))

        report = classification_report(y_test, y_pred, target_names=self.params['categories'])
        print(report)
        TEST_RESULTS['test_result']['report'] = report
        TEST_RESULTS['epoch_history'] = hist.history
        TEST_RESULTS['test_result']['test_acc'] = acc
        TEST_RESULTS['test_result']['test_loss'] = score

        test_confusion_matrix = confusion_matrix(y_test, y_pred)
        TEST_RESULTS['test_result']['test_confusion_matrix'] = test_confusion_matrix.tolist()

        print('Test loss: {0}  |  test accuracy: {1}'.format(score, acc))
        self.save_results(model)

    def save_results(self, model):
        tm = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        tsm = tm.split("_")
        TEST_RESULTS['date'] = tsm[0]
        TEST_RESULTS['date_time'] = tsm[1]

        TEST_RESULTS['embedding']['vocabulary_size'] = len(self.params['char_index'])
        TEST_RESULTS["embedding"]['embedding_dimension'] = self.params['embedding_dimension']

        TEST_RESULTS['hiperparameter']['epoch'] = self.params['epoch']
        TEST_RESULTS['hiperparameter']['train_batch_size'] = self.params['batch_train']
        TEST_RESULTS['hiperparameter']['test_batch_size'] = self.params['batch_test']
        TEST_RESULTS['hiperparameter']['sequence_length'] = self.params['sequence_length']

        mlflow.log_param("epoch_number", self.params['epoch'])
        mlflow.log_param("train_batch_size", self.params['batch_train'])
        mlflow.log_param("test_batch_size", self.params['batch_train'])
        mlflow.log_param("embed_dim", self.params['embedding_dimension'])

        model_json = model.to_json()
        model.save("{}model_all.h5".format(self.params['result_dir']))
        open("{0}model.json".format(self.params['result_dir']), "w").write(json.dumps(model_json))
        model.save_weights("{0}weights.h5".format(self.params['result_dir']))

        open("{0}raw_test_results.json".format(self.params['result_dir']), "w").write(json.dumps(TEST_RESULTS))
        open("{0}model_summary.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['hiperparameter']["model_summary"])
        open("{0}classification_report.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['test_result']['report'])

        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'])
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'], normalized=True)

        embeddings = model.layers[0].get_weights()[0]
        words_embeddings = {w: embeddings[idx].tolist() for w, idx in self.params['char_index'].items()}
        open("{0}char_embeddings.json".format(self.params['result_dir']), "w").write(json.dumps(words_embeddings))

        # Log metrics and artifacts
        for t in TEST_RESULTS['epoch_history']['loss']:
            mlflow.log_metric("train_loss", t)
        for t in TEST_RESULTS['epoch_history']['acc']:
            mlflow.log_metric("train_acc", t)
        for t in TEST_RESULTS['epoch_history']['val_loss']:
            mlflow.log_metric("val_loss", t)
        for t in TEST_RESULTS['epoch_history']['val_acc']:
            mlflow.log_metric("val_acc", t)

        mlflow.log_artifact("{0}raw_test_results.json".format(self.params['result_dir']))
        mlflow.log_artifact("{0}model_all.h5".format(self.params['result_dir']))
        mlflow.log_artifact("{0}model_summary.txt".format(self.params['result_dir']))
        mlflow.log_artifact("{0}classification_report.txt".format(self.params['result_dir']))
        mlflow.log_artifact("{0}model.json".format(self.params['result_dir']))
        mlflow.log_artifact("{0}weights.h5".format(self.params['result_dir']))
        mlflow.log_artifact("{0}confusion_matrix.png".format(self.params['result_dir']))
        mlflow.log_artifact("{0}normalized_confusion_matrix.png".format(self.params['result_dir']))

class Plotter:
    def plot_graphs(self, train, val, save_to=None, name="accuracy"):
        if name == "accuracy":
            val, = plt.plot(val, label="val_acc")
            train, = plt.plot(train, label="train_acc")
        else:
            val, = plt.plot(val, label="val_loss")
            train, = plt.plot(train, label="train_loss")

        plt.ylabel(name)
        plt.xlabel("epoch")
        plt.legend(handles=[val, train], loc=2)

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))
        plt.close()

    def plot_confusion_matrix(self, confusion_matrix, categories, save_to=None, normalized=False):
        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)
        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=True, yticklabels=True)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to:
            if normalized:
                plt.savefig("{0}/{1}.png".format(save_to, "normalized_confusion_matrix"))
            else:
                plt.savefig("{0}/{1}.png".format(save_to, "confusion_matrix"))

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", default=10, help='The number of epoch', type=int)
    parser.add_argument("-arch", "--architecture", default="cnn", help='Architecture to be tested')
    parser.add_argument("-bs", "--batch_size", default=2000, help='batch size')

    args = parser.parse_args()

    return args

def main():
    args = argument_parsing()
    vc = PhishingUrlDetection(args.epoch, args.architecture)

    mlflow.start_run(experiment_id=1)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_and_vectorize_data()
    vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test)
    mlflow.end_run()

if __name__ == '__main__':
    main()