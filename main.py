import libemg
import numpy as np
import time
import os 
import joblib

import sys, csv, time, math, threading, random
from datetime import datetime
# from PySide6.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
#     QSpinBox, QDoubleSpinBox, QCheckBox, QHBoxLayout, QLineEdit, QMessageBox
# )
# from PySide6.QtCore import Qt, QTimer
# from PySide6.QtGui import QPainter, QColor, QPen, QBrush

COLLECT = 0
NAME = 'amir'
dataset_folder = 'data/' + NAME + '/'
MODEL = 1
PATH = 'pickles/'

X = 0.0
Y = 0.0
probs = []
velocity = 0

kernel_size = 5
n_samples = 40
inc = 1
n_channels = 8
n_classes = 5
features = 2
epochs = 50    
decay = 10
lr_factor = 0.5
lr_patience = 1
init_learning_rate = 1e-3
min_learning_rate = 1e-5
dropout = 0.2
batch = 1024
patience = 5




if __name__ == "__main__":
    p, smm = libemg.streamers.myo_streamer() 
    odh = libemg.data_handler.OnlineDataHandler(smm)

    if not os.path.exists(dataset_folder) or not os.listdir(dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)
        args = {'num_reps': 5, 'rep_time': 5, 'rest_time': 2, 'media_folder': 'images/', 'data_folder': dataset_folder}
        ui = libemg.gui.GUI(odh, args=args, width=700, height=700)
        ui.download_gestures([1,2,3,6,7], "images/")
        ui.start_gui()

    filters = [
            libemg.data_handler.RegexFilter(left_bound="C_", right_bound="_R", values=["0","1","2","3","4"], description='classes'),
            libemg.data_handler.RegexFilter(left_bound="R_", right_bound="_emg.csv", values=["0", "1", "2", "3", "4"], description='reps'), 
    ]
    offline_dh = libemg.data_handler.OfflineDataHandler()
    offline_dh.get_data(folder_location=dataset_folder, regex_filters=filters, delimiter=',')
    train_windows, train_metadata = offline_dh.parse_windows(n_samples, inc)
    train_windows /= 128
    train_windows = joblib.load(PATH + 'valid_windows').astype(np.int16) / 128
    train_metadata = joblib.load(PATH + 'valid_meta')

    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = ['WENG']
    training_feats = fe.extract_features(feature_list, train_windows)

    dataset = {\
            'training_features': training_feats,
            'training_labels': train_metadata['classes'],
    }

    if MODEL:
        import torch
        print(torch.cuda.is_available())
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
        print(keras.backend.backend())

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_learning_rate)
        initializer = keras.initializers.HeNormal(seed=42)

        inp = keras.layers.Input((40, 8))

        x = keras.layers.Conv1D(16, 4, padding='same', kernel_initializer=initializer)(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Conv1D(32, 4, padding='same', kernel_initializer=initializer)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation='gelu', kernel_initializer=initializer)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Dense(32, activation='gelu', kernel_initializer=initializer)(x)
        out = keras.layers.Dense(5, activation='softmax', kernel_initializer=initializer)(x)
        model = keras.Model(inputs=inp, outputs=out)
        optimizer = keras.optimizers.Adam(learning_rate=init_learning_rate)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # loss=EqLoss(),
            optimizer=optimizer,
            metrics=['accuracy'])
        model.summary()
        model.load_weights(PATH + 'eq2_crop_f.weights.h5')

        # train_len = len(train_windows)
        # i = np.arange(train_len)
        # np.random.shuffle(i)
        # train_windows = train_windows[i]
        # train_metadata['classes'] = train_metadata['classes'][i]
        # history = model.fit(train_windows[:int(0.8 * train_len)], train_metadata['classes'][:int(0.8 * train_len)], 
        #                     epochs=50, shuffle=True, batch_size=batch, callbacks=[early_stop, reduce_lr],
        #                     validation_data=(train_windows[int(0.8 * train_len):], train_metadata['classes'][int(0.8 * train_len):]))

        # train_windows = joblib.load(PATH + 'train_windows').astype(np.int8) / 1.2
        # train_meta = joblib.load(PATH + 'train_meta')
        o_classifier = libemg.emg_predictor.EMGClassifier(model)
        # o_classifier.add_rejection(0.9)
        # o_classifier.add_majority_vote(10)
        o_classifier.add_velocity(train_windows, train_metadata['classes'])
        classifier = libemg.emg_predictor.OnlineEMGClassifier(o_classifier, n_samples, inc, odh, features=None, output_format='probabilities')
        classifier.run(block=False)

    else:
        o_classifier = libemg.emg_predictor.EMGClassifier("LDA")
        # o_classifier.fit(feature_dictionary=dataset)
        # o_classifier.add_rejection(0.8)
        # o_classifier.add_velocity(train_windows, train_metadata['classes'])
        # classifier = libemg.emg_predictor.OnlineEMGClassifier(o_classifier, n_samples, inc, odh, feature_list, output_format='probabilities')
        # classifier.run(block=False)

#     # Create while loop to send data 
    import socket 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 12346))

    time.sleep(1)

    state_lock = threading.Lock()
    def input_thread():
        global X, Y, speed, probs, velocity
        while True:
            data,_ = sock.recvfrom(1024)
            data = str(data.decode("utf-8")).split(' ')[:-1]

            if data:
                probs = np.array([float(p) for p in data[:-1]])
                gesture = np.argmax(probs)
                velocity = float(data[-1])
                speed = np.clip(velocity, 0.05, 1)

                # print(probs, gesture, velocity)
                # print(speed)
                
                message = ''
                if gesture == 1:
                    message = '0;1'
                elif gesture == 2:
                    message = '-1;0'
                elif gesture == 3:
                    message = '1;0'
                elif gesture == 4:
                    message = '0;-1'
                else:
                    message = '0;0'

            try:
                raw = message
                parts = raw.strip().split(";")
                if len(parts) != 2:
                    continue
                x, y = int(parts[0]), int(parts[1])
                with state_lock:
                    X = x * speed
                    Y = y * speed

            except:
                continue

    threading.Thread(target=input_thread, daemon=True).start()
    odh.log_to_file(file_path=f"{dataset_folder}fitts_online{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}")