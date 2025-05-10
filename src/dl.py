import os
import json
import time
import pprint
import argparse
import datetime
import numpy as np
import seaborn as sns
import keras.callbacks as ckbs
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import mlflow
from dl_models import DlModels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
import tensorflow as tf
import psutil  # لمراقبة الموارد
import gc  # لتنظيف الذاكرة
from sklearn.utils.class_weight import compute_class_weight  # لمعالجة الـ imbalance

print("done")

# sunucuda calismak icin
plt.switch_backend('agg')

pp = pprint.PrettyPrinter(indent=4)

TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}

# تقليل استهلاك الذاكرة في TensorFlow
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None

class MemoryCallback(ckbs.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # مراقبة استهلاك الذاكرة
        memory_info = psutil.virtual_memory()
        print(f"Memory Usage after epoch {epoch + 1}: {memory_info.percent}% (Used: {memory_info.used / 1024**3:.2f} GB)")
        # تنظيف الذاكرة
        gc.collect()
        tf.keras.backend.clear_session()

class PhishingUrlDetection:

    def __init__(self, epoch, architecture):
        self.params = {'loss_function': 'categorical_crossentropy',
                       'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
                       'sequence_length': 200,
                       'batch_train': 64,
                       'batch_test': 64,
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
        train = [line.strip() for line in open("{}/train.txt".format(self.params['dataset_dir']), "r").readlines() if line.strip()]
        test = [line.strip() for line in open("{}/test.txt".format(self.params['dataset_dir']), "r").readlines() if line.strip()]
        val = [line.strip() for line in open("{}/val.txt".format(self.params['dataset_dir']), "r").readlines() if line.strip()]

        mlflow.log_param("samples_train", len(train))
        mlflow.log_param("samples_test", len(test))
        mlflow.log_param("samples_val", len(val))
        mlflow.log_param("samples_overall", len(train) + len(test) + len(val))
        mlflow.log_param("dataset", self.params['dataset_dir'])

        # التحقق من البيانات قبل التقسيم
        raw_x_train = []
        raw_y_train = []
        for line in train:
            parts = line.split()
            if len(parts) >= 2:
                raw_y_train.append(parts[0])
                raw_x_train.append(parts[1])
            else:
                print(f"Warning: Invalid line in train.txt: {line}")

        raw_x_val = []
        raw_y_val = []
        for line in val:
            parts = line.split()
            if len(parts) >= 2:
                raw_y_val.append(parts[0])
                raw_x_val.append(parts[1])
            else:
                print(f"Warning: Invalid line in val.txt: {line}")

        raw_x_test = []
        raw_y_test = []
        for line in test:
            parts = line.split()
            if len(parts) >= 2:
                raw_y_test.append(parts[0])
                raw_x_test.append(parts[1])
            else:
                print(f"Warning: Invalid line in test.txt: {line}")

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
            mlflow.log_param('architecture', "ann_complex")
        elif self.params['architecture'] == "cnn":
            model = self.dl_models.cnn_complex(self.params['char_index'])
            mlflow.log_param('architecture', "cnn_complex")
        elif self.params['architecture'] == "rnn":
            model = self.dl_models.rnn_complex(self.params['char_index'])
            mlflow.log_param('architecture', "rnn_complex")
        else:
            model = self.dl_models.cnn_complex3(self.params['char_index'])
            mlflow.log_param('architecture', "cnn3")

        model.compile(loss=self.params['loss_function'], optimizer=self.params['optimizer'], metrics=['accuracy'])

        model.summary()
        model.summary(print_fn=lambda x: self.model_sum(x + '\n'))

        # إضافة EarlyStopping وMemoryCallback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        memory_callback = MemoryCallback()

        # إضافة class_weight لمعالجة الـ imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
        class_weight_dict = dict(enumerate(class_weights))

        hist = model.fit(x_train, y_train,
                         batch_size=self.params['batch_train'],
                         epochs=self.params['epoch'],
                         shuffle=True,
                         validation_data=(x_val, y_val),
                         class_weight=class_weight_dict,
                         callbacks=[early_stopping, memory_callback])

        t = time.time()
        score, acc = model.evaluate(x_test, y_test, batch_size=self.params['batch_test'])

        mlflow.log_metric("test_time", float("{0:.2f}".format(round(time.time() - t, 2))))
        TEST_RESULTS['test_result']['test_time'] = time.time() - t

        # تحويل y_test و y_pred بشكل صحيح
        y_test = np.argmax(y_test, axis=1)
        y_pred = model.predict(x_test, batch_size=self.params['batch_test'], verbose=1)
        y_pred = np.argmax(y_pred, axis=1)

        # طباعة عينة للتأكد
        print("y_test sample:", y_test[:5])
        print("y_pred sample:", y_pred[:5])

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

        # إضافة اسم الموديل في أسماء الملفات
        model_name = self.params['architecture']  # ann, cnn, rnn
        model_filename = f"{model_name}_model_all.h5"
        json_filename = f"{model_name}_model.json"
        weights_filename = f"{model_name}_weights.h5"
        results_filename = f"{model_name}_raw_test_results.json"
        summary_filename = f"{model_name}_model_summary.txt"
        report_filename = f"{model_name}_classification_report.txt"
        embeddings_filename = f"{model_name}_char_embeddings.json"
        cm_filename = f"{model_name}_confusion_matrix.png"
        normalized_cm_filename = f"{model_name}_normalized_confusion_matrix.png"

        # حفظ الموديل بصيغة .h5
        model.save(f"{self.params['result_dir']}{model_filename}")

        # حفظ هيكل الموديل بصيغة JSON
        model_json = model.to_json()
        open(f"{self.params['result_dir']}{json_filename}", "w").write(json.dumps(model_json))

        # حفظ أوزان الموديل بصيغة .h5
        model.save_weights(f"{self.params['result_dir']}{weights_filename}")

        # حفظ بيانات أخرى (مثل النتائج والـ embeddings)
        open(f"{self.params['result_dir']}{results_filename}", "w").write(json.dumps(TEST_RESULTS))
        open(f"{self.params['result_dir']}{summary_filename}", "w").write(TEST_RESULTS['hiperparameter']["model_summary"])
        open(f"{self.params['result_dir']}{report_filename}", "w").write(TEST_RESULTS['test_result']['report'])

        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], save_to=self.params['result_dir'], filename=cm_filename)
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], save_to=self.params['result_dir'], normalized=True, filename=normalized_cm_filename)

        embeddings = model.layers[0].get_weights()[0]
        words_embeddings = {w: embeddings[idx].tolist() for w, idx in self.params['char_index'].items()}
        open(f"{self.params['result_dir']}{embeddings_filename}", "w").write(json.dumps(words_embeddings))

        # Log metrics and artifacts باستخدام المفتاح المناسب للـ accuracy
        accuracy_key = next((key for key in TEST_RESULTS['epoch_history'].keys() if 'acc' in key), None)
        if accuracy_key:
            for t in TEST_RESULTS['epoch_history'][accuracy_key]:
                mlflow.log_metric("train_acc", t)
            for t in TEST_RESULTS['epoch_history'].get(f'val_{accuracy_key}', []):
                mlflow.log_metric("val_acc", t)

        for t in TEST_RESULTS['epoch_history'].get('loss', []):
            mlflow.log_metric("train_loss", t)
        for t in TEST_RESULTS['epoch_history'].get('val_loss', []):
            mlflow.log_metric("val_loss", t)

        # Log artifacts لـ mlflow
        mlflow.log_artifact(f"{self.params['result_dir']}{results_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{model_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{summary_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{report_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{json_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{weights_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{cm_filename}")
        mlflow.log_artifact(f"{self.params['result_dir']}{normalized_cm_filename}")

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

    def plot_confusion_matrix(self, confusion_matrix, save_to=None, normalized=False, filename=None):
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

        g.set_yticklabels(["phishing", "legitimate"], rotation=0)
        g.set_xticklabels(["phishing", "legitimate"], rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to and filename:
            plt.savefig(f"{save_to}/{filename}")
        plt.close()

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", default=20, help='The number of epoch', type=int)
    parser.add_argument("-arch", "--architecture", default="ann", help='Architecture to be tested')
    parser.add_argument("-bs", "--batch_size", default=64, help='batch size')

    args = parser.parse_args()

    return args

def main():
    args = argument_parsing()

    # إنشاء أو استرجاع التجربة بناءً على الاسم
    experiment_name = "phishing_detection"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    vc = PhishingUrlDetection(args.epoch, args.architecture)

    # تشغيل التدريب
    mlflow.start_run(experiment_id=experiment_id)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_and_vectorize_data()
    vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test)
    mlflow.end_run()

if __name__ == '__main__':
    main()