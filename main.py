import libemg
import numpy as np
import time
from datetime import datetime
import csv
from dynamixel_sdk import *
import time
import threading
import sys
import socket 
import joblib
import os

COLLECT = 0
NAME = 'aibme'
PATH = 'pickles/'
MODEL = 1
dataset_folder = 'data/' + NAME + '/'
bypass_log = 'bypass_log/' + NAME + '/'

X = 0.0
Y = 0.0
velocity = 0
probs = []

kernel_size = 5
n_samples = 30
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

# ---- Control table address ----
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_GOAL_VEL = 32
ADDR_TORQUE_LIMIT = 34 

ADDR_PRESENT_POSITION = 36
ADDR_PRESENT_VELOCITY = 38
ADDR_PRESENT_TORQUE = 40

PROTOCOL_VERSION = 1.0
WRIST_ID = 0 
GRIP_ID = 1 
BAUDRATE = 3000000  
DEVICENAME = 'COM3' 

MOVING_THRESHOLD = 30   
WRIST_MAX_TORQUE = 500
GRIP_MAX_TORQUE = 400
WRIST_MAX_VEL = 300
GRIP_MAX_VEL = 200
WRIST_MIN_POS = 800
WRIST_MAX_POS = 2500
GRIP_MAX_POS = 2340
GRIP_MIN_POS = 1780
GRIP_POS = [GRIP_MIN_POS, int(np.mean([GRIP_MIN_POS, GRIP_MAX_POS])), GRIP_MAX_POS]
WRIST_POS = [WRIST_MIN_POS, int(np.mean([WRIST_MIN_POS, WRIST_MAX_POS])), WRIST_MAX_POS]

wrist_pos = 0
wrist_trq = WRIST_MAX_TORQUE
grip_pos = 0
grip_trq = GRIP_MAX_TORQUE
speed = 0
speedG = 0
speedW = 0
state_lock = threading.Lock()

params = {
    'frame_rate': 60,
    'physics': {
        'enabled': 0,
        'mass': 5,
        'max_acceleration': 100,
        'velocity_scale': 700,
        'damping': 1,
    }
}

if __name__ == "__main__":
    p, smm = libemg.streamers.myo_streamer() 
    odh = libemg.data_handler.OnlineDataHandler(smm)

    if not os.path.exists(dataset_folder) or not os.listdir(dataset_folder):
        os.mkdir(dataset_folder)
        args = {'num_reps': 3, 'rep_time': 5, 'rest_time': 2, 'media_folder': 'images/', 'data_folder': dataset_folder}
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

    fe = libemg.feature_extractor.FeatureExtractor()
    feature_list = ['WENG']
    training_feats = fe.extract_features(feature_list, train_windows)

    dataset = {\
            'training_features': training_feats,
            'training_labels': train_metadata['classes'],
    }

    # # ---- Models ----
    # if MODEL:
    #     import torch
    #     print(torch.cuda.is_available())
    #     os.environ["KERAS_BACKEND"] = "torch"
    #     import keras
    #     print(keras.backend.backend())

    #     early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
    #     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_learning_rate)
    #     initializer = keras.initializers.HeNormal(seed=42)
    #     inp = keras.layers.Input((n_channels, n_samples))
    #     x = keras.layers.Reshape((n_samples, 8))(inp)
    #     x = keras.layers.Conv1D(32, 4, kernel_initializer=initializer)(x)
    #     x = keras.layers.BatchNormalization()(x)
    #     x = keras.layers.ReLU()(x)
    #     x = keras.layers.Dropout(dropout)(x)
    #     x = keras.layers.Conv1D(16, 4, kernel_initializer=initializer)(x)
    #     x = keras.layers.BatchNormalization()(x)
    #     x = keras.layers.ReLU()(x)
    #     x = keras.layers.Dropout(dropout)(x)
    #     x = keras.layers.Flatten()(x)
    #     x = keras.layers.Dense(128, activation='relu', kernel_initializer=initializer)(x)
    #     x = keras.layers.Dropout(dropout)(x)
    #     x = keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)(x)
    #     out = keras.layers.Dense(5, activation='softmax', kernel_initializer=initializer)(x)
    #     model = keras.Model(inputs=inp, outputs=out)
    #     model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
    #                     optimizer=keras.optimizers.Adam(learning_rate=init_learning_rate),
    #                     metrics=['accuracy'])
        
    #     # model.load_weights(PATH + 'model_22.weights.h5')
        
    #     train_len = len(train_windows)
    #     i = np.arange(train_len)
    #     np.random.shuffle(i)
    #     train_windows = train_windows[i]
    #     train_metadata['classes'] = train_metadata['classes'][i]
    #     history = model.fit(train_windows[:int(0.8 * train_len)], train_metadata['classes'][:int(0.8 * train_len)], 
    #                         epochs=50, shuffle=True, batch_size=batch, callbacks=[early_stop, reduce_lr],
    #                         validation_data=(train_windows[int(0.8 * train_len):], train_metadata['classes'][int(0.8 * train_len):]))

    #     # train_windows = joblib.load(PATH + 'train_windows')
    #     # train_meta = joblib.load(PATH + 'train_meta')
    #     o_classifier = libemg.emg_predictor.EMGClassifier2(model)
    #     o_classifier.add_rejection(0.8)
    #     o_classifier.add_majority_vote(10)
    #     o_classifier.add_velocity(train_windows, train_metadata['classes'])
    #     classifier = libemg.emg_predictor.OnlineEMGClassifier2(o_classifier, n_samples, inc, odh, None, output_format='probabilities')
    #     classifier.run(block=False)

    # else:
    #     o_classifier = libemg.emg_predictor.EMGClassifier("LDA")
    #     o_classifier.fit(feature_dictionary=dataset)
    #     o_classifier.add_rejection(0.8)
    #     o_classifier.add_velocity(train_windows, train_metadata['classes'])
    #     classifier = libemg.emg_predictor.OnlineEMGClassifier(o_classifier, n_samples, inc, odh, feature_list, output_format='probabilities')
    #     classifier.run(block=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 12346))

    time.sleep(1)

    if os.name == 'nt':
        import msvcrt
        def getch():
            return msvcrt.getch().decode()
    else:
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        def getch():
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
     
    # ---- Input ----
    def input_thread():
        global wrist_pos, grip_pos, wrist_trq, grip_trq
        global WRIST_MAX_TORQUE, GRIP_MAX_TORQUE, speedG, speedW
        global probs, velocity

        while True:
            data,_ = sock.recvfrom(1024)
            data = str(data.decode("utf-8")).split(' ')[:-1]

            if data:
                # print(data)
                probs = np.array([float(p) for p in data[:-1]])
                gesture = np.argmax(probs)
                velocity = float(data[-1])
                speed = np.clip(velocity, 0.0, 1.0)
                # speed *= 1.2 if speed > 0.1 else 1
                # print(speed)
                
                # message = ''
                # if gesture == 1:
                #     message = '-1;0'
                # elif gesture == 2:
                #     message = '0;-1'
                # elif gesture == 3:
                #     message = '2;-1'
                # elif gesture == 4:
                #     message = '-1;2'
                # else:
                #     message = '-1;-1'

                message = ''
                if gesture == 0:
                    message = '-1;0'
                elif gesture == 1:
                    message = '-1;2'
                elif gesture == 4:
                    message = '2;-1'
                elif gesture == 3:
                    message = '0;-1'
                else:
                    message = '-1;-1'

            try:
                raw = message
                parts = raw.strip().split(";")
                if len(parts) != 2:
                    continue
                w, g = int(parts[0]), int(parts[1])
                with state_lock:
                    wrist_pos = w if w != -1 else wrist_pos
                    grip_pos = g if g != -1 else grip_pos
                    # wrist_trq = WRIST_MAX_TORQUE
                    # grip_trq = GRIP_MAX_TORQUE
                    speedG = 0 if g == -1 else speed
                    speedW = 0 if w == -1 else speed
                    # if w == 2:
                    #     wrist_pos = wrist_trq = 0
                    # if g == 2:
                        # grip_pos = grip_trq = 0
            except:
                continue

    while 1:
        print(odh.get_data(N=10)[0])
        print(odh.get_data()[1])
        time.sleep(1)
