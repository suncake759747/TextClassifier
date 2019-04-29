import config
import keras as k
import numpy as np
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.layers import Dropout
from keras.models import Model
from keras.engine.topology import Input

from sklearn.metrics import confusion_matrix, classification_report


def build_keras_model(batch_size, dropout_rate, embedding_size, max_len, num_epochs, train_steps, validation_steps,
                      vocab_processor, label_mapping, training_data_path, validation_data_path, model_name='model',
                      pretrained_embeddings=None):
    vocabulary_size = len(vocab_processor.vocabulary_)
    model = Sequential()
    if pretrained_embeddings:
        model.add(Embedding(vocabulary_size, embedding_size, weights=[final_embeddings], input_length=maxlen, trainable=False))
    else:
        model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len,
                            trainable=True))
    model.add(Bidirectional(LSTM(embedding_size, dropout=dropout_rate, recurrent_dropout=dropout_rate)))
    model.add(Dense(int(embedding_size/2), activation='selu'))
    model.add(Dropout(rate = dropout_rate))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(generate_keras_batches(training_data_path, num_epochs, vocab_processor, label_mapping), 
                        epochs=num_epochs, steps_per_epoch = train_steps, 
                        validation_data=generate_keras_batches(validation_data_path, num_epochs, vocab_processor, label_mapping),
                        validation_steps=validation_steps)
    model.save(model_name)
    return model

def generate_batches(file, vocab_processor, batch_size, label_mapping):
    batch_data = []
    label_data = []
    ids = []
    record_number = 0
    for record in record_iter(file):
        record_number += 1
        ids.append(record[0])
        batch_data.append(record[1])
        label_data.append(label_mapping[record[2].rstrip()])
        if record_number % batch_size == 0:
            x_np = np.array(np.array(list(vocab_processor.transform(batch_data))), dtype=np.float32)
            y_np = np.array(label_data)
            id_np = np.array(ids)
            batch_data = []
            label_data = []
            ids = []
            yield (x_np, y_np, id_np)
    x_np = np.array(np.array(list(vocab_processor.transform(batch_data))), dtype=np.float32)
    y_np = np.array(label_data)
    id_np = np.array(ids)
    yield (x_np, y_np, id_np)
    
def generate_keras_batches(file, num_epochs, vocab_processor, label_mapping, batch_size=config.batch_size,
                           max_words_per_doc=config.max_len):
    while 1:
        batches = generate_batches(file, vocab_processor, batch_size, label_mapping)
        for x_batch, y_batch, _ in batches:
                yield (x_batch[:,:max_words_per_doc], y_batch)

def record_iter(file):
    with open(file, 'r') as f:
        for row in f:
            yield row.split('\t')                

def use_lstm(model, data_path, vocab_processor, label_mapping):
    batch_size = 1024
    probs = np.empty((0,7))
    labels = np.empty((0,1))
    ids = np.empty((0,1))
    batches_processed = 0
    for batch_x, batch_y, batch_ids in generate_batches(data_path, vocab_processor, batch_size, label_mapping):
        batches_processed += 1
        if batches_processed % 20 == 0:
            print("%s records processed" % (batches_processed * batch_size))
        probs = np.vstack([probs, model.predict_on_batch(batch_x)])
        labels = np.vstack([labels, batch_y.reshape(-1,1)])
        ids = np.vstack([ids, batch_ids.reshape(-1,1)])
    preds = np.argmax(probs, axis = 1).reshape(-1,1)
    confusion_df = pd.DataFrame(data = confusion_matrix(labels, preds), 
                                columns = sorted(label_mapping.keys()), index=sorted(label_mapping.keys()))
    print("Confusion Matrix for data in %s" % data_path)
    print("PREDICTED CLASS ON X-AXIS. TRUE CLASS ON X AXIS.")
    print(confusion_df)
    print(classification_report(labels, preds, 
                                labels=list(label_mapping.values()), target_names=list(label_mapping.keys())))
    return probs, labels, preds, ids