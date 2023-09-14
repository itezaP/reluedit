import random
import math
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import KFold
from scipy.spatial import distance

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, add, ThresholdedReLU, Lambda, Reshape, concatenate, Cropping1D, Flatten, LeakyReLU, ELU
from tensorflow.python.client import device_lib

def transform_seqs_to_input(seqA, seqB):
    x = []
    y = []
    for i in range(len(seqA)):
        x.append(int(seqA[i]))
        y.append(int(seqB[i]))

    return [x, y]

def matching_module():
    epsilon = 1

    model = Sequential()
    model.add(Dense(units=2, activation='relu', use_bias=True, input_shape=(2,)))
    model.add(Dense(units=2, activation='relu', use_bias=True))
    model.add(Dense(units=1, activation='relu', use_bias=True))

    w1 = model.layers[0].get_weights()
    w1[0][0][0], w1[0][0][1] = 1.0, -1.0
    w1[0][1][0], w1[0][1][1] = -1.0, 1.0
    w1[1][0], w1[1][1] = 0, 0
    w2 = model.layers[1].get_weights()
    w2[0][0][0], w2[0][0][1] = 1.0, 1.0
    w2[0][1][0], w2[0][1][1] = 1.0, 1.0
    w2[1][0], w2[1][1] = epsilon, -1 * epsilon
    w3 = model.layers[2].get_weights()
    w3[0][0][0], w3[0][1][0] = (1.0/epsilon), -1.0 * (1.0/epsilon)
    w3[1][0] = -1

    model.layers[0].set_weights(w1)
    model.layers[1].set_weights(w2)
    model.layers[2].set_weights(w3)

    model.trainable = False

    return model

def hammingNet(seq_length):
    input = Input(shape=(2, seq_length), name='input')

    y = Cropping1D((1,0))(input)
    x = Cropping1D((0,1))(input)
    y = Flatten()(y)
    x = Flatten()(x)

    out = {}
    start_i = tf.Variable(0, name='start_i')
    for i in range(seq_length):
        a = int(start_i)
        layername = 'for_gen_dense_' + str(i+1)
        z = Dense(1, activation='relu', name=layername, use_bias=False)(y[:seq_length, a:a+1])
        out[layername] = z
        start_i = start_i + 1
    
    pair_i = tf.Variable(1, name='pair_i')
    matchings = []
    for i in range(seq_length):
        c = int(pair_i)

        y_dense_layer_name = 'for_gen_dense_' + str(c)
        densed_y = out[y_dense_layer_name]
        x_char = Flatten()(x[0, (c-1):c])
        concat_l_name = 'concat_' + str(c)
        pair = concatenate([x_char, densed_y], name=concat_l_name)
        matching = matching_module()(pair)
        matchings.append(matching)
        
        pair_i = pair_i + 1

    all_match = []
    if seq_length > 1:
        all_match = concatenate([*matchings], name='concat')
        calc_layer_name = 'all_dense'
        result = Dense(1, activation='relu', name=calc_layer_name ,use_bias=False)(all_match)
        return Model(inputs=input, outputs=result)
    else:
        return Model(inputs=input, outputs=matchings)

def set_weight_for_debug(model, seq_length):
    print('setting weigths ...')

    for i in range(seq_length):
        lname = 'for_gen_dense_' + str(i+1)
        weights = model.get_layer(lname).get_weights()
    
        # for calculating correct distance
        #weights[0][0][0] = 1

        # for learning 
        weights[0][0][0] = random.uniform(0,1)
        model.get_layer(lname).set_weights(weights)
    
    if seq_length > 1:
        lname = 'all_dense'
        w = model.get_layer(lname).get_weights()
        for i in range(seq_length):
            w[0][i][0] = 1
        model.get_layer(lname).set_weights(w)

def froozen_align_model(model):
    print('froozen parameters in a network for alignment ...')
    layers = model.layers
    for layer in layers:
        if 'for_gen_dense' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    #return model

def measure_cnst_model():
    seq_length = 5

    times = []
    init_times = []
    MODEL_N = 5
    average = 0.0
    for i in range(MODEL_N):
        cnst_model_s_time = time.perf_counter()
        model = hammingNet(seq_length)
        cnst_model_e_time = time.perf_counter() - cnst_model_s_time

        times.append(cnst_model_e_time)
        average += cnst_model_e_time
        
    average = average / MODEL_N
    print(MODEL_N, 'trial average construct model time:', average)
    print('min=', round(min(times),3), 'max=', round(max(times),3), 'median=', round(np.median(times),3), 'std=', round(np.std(times),3))

def measure():
    filename = './sampledata/predtime_length_5_hamming.csv'
    print('test file :', filename)

    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line.rstrip('\n')
            lines.append(line)
    
    confirm_l = lines[0]
    sp = confirm_l.split(',')
    x, y = sp[0], sp[1]
    seq_length = len(x)

    model = hammingNet(seq_length)
    set_weight_for_debug(model, seq_length)
    froozen_align_model(model)

    N = 500
    perform_data = []
    if len(lines) < 500:
        shortage_times = math.ceil(N / len(lines))
        perform_data = lines * shortage_times
        perform_data = perform_data[:N]
    else:
        perform_data = lines[:N]

    times = []
    average_time = 0
    correct = 0
    for data in perform_data:
        sp = data.split(',')
        x, y, c_score = sp[0], sp[1], int(sp[2])
        input = transform_seqs_to_input(x, y)

        x = tf.constant([input])
        s_time = time.perf_counter()
        p_score = model(x, training=False)
        e_time = time.perf_counter() - s_time

        average_time += e_time
        times.append(e_time)
    
        p_score = int(p_score[0][0])
        if int(p_score) == c_score:
            correct += 1
        else:
            print('valid :', x, y, p_score, c_score)

    print('correct :', correct)

    average_time = average_time / N
    print(N, 'samples average time:', round(average_time, 3))
    print('min=', min(times), 'max=', max(times), np.argmax(times), 'median=', np.median(times), 'std=', np.std(times))

def plot_progress_weights(progress_weights, epoch, desire):
    x = [*range(epoch+1)]

    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'v', '*']
    plt.figure(figsize=(18, 10), dpi=100)

    y_s = np.array(progress_weights).T
    w_i = 0
    for y in y_s:
        #plt.scatter(x, y)
        m = markers[w_i%len(markers)]
        n = color_cycle[w_i%len(color_cycle)]
        plt.plot(x, y, marker=m, label="weigth" + str(w_i+1), color=n)
        plt.hlines(desire[w_i], 0, epoch, color='black' ,linestyles='dotted')
        w_i += 1

    plt.xlabel('epochs')
    plt.ylabel('weight')
    plt.ylim(-0.5,1.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.savefig('weights.png')  

def training():
    # data and model setting
    lines = []
    EPOCHS = 20
    DESIRE = '01011'
    LEN = len(DESIRE)
    filename = './sampledata/desired_length_' + str(LEN) + '_hamming.csv'
    with open(filename, 'r') as f:
        for line in f:
            line.rstrip('\n')
            lines.append(line)

    model = hammingNet(LEN)
    set_weight_for_debug(model, LEN)
    froozen_align_model(model)

    init_trained_weights = []
    for layer in model.layers:
        if 'for_gen_dense' in layer.name:
            weight = layer.get_weights()
            init_trained_weights.append(float(weight[0][0]))
    init_trained_weights.reverse()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    progress_weights = []
    
    progress_weight = []
    for i in range(LEN):
        lname = 'for_gen_dense_' + str(i+1)
        weights = model.get_layer(lname).get_weights()
        progress_weight.append(weights[0][0][0])
    progress_weights.append(progress_weight)
    
    progress_grads = []
    for epoch in range(EPOCHS):
        print('Epoch', epoch)
        loss = tf.Variable(0.0, name='loss')
        with tf.GradientTape() as tape:
            for line in lines:
                sp = line.split(',')
                x, y, true_score = sp[0], sp[1], int(sp[2])
                input = transform_seqs_to_input(x, y)
                input = tf.constant([input])
                logit = model(input, training=True)
                loss = loss + loss_fn(true_score, logit)
            batch_loss = loss / len(lines)
            print(batch_loss)
            grads = tape.gradient(batch_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            progress_grad = []
            for grad in grads:
                progress_grad.append(float(grad[0][0]))
            progress_grads.append(progress_grad)

        progress_weight = []
        for i in range(LEN):
            lname = 'for_gen_dense_' + str(i+1)
            weights = model.get_layer(lname).get_weights()
            progress_weight.append(weights[0][0][0])
        progress_weights.append(progress_weight)
    
    raw_trained_weights = []
    for layer in model.layers:
        if 'for_gen_dense' in layer.name:
            weight = layer.get_weights()
            raw_trained_weights.append(float(weight[0][0]))
    raw_trained_weights.reverse()

    print('init weights:', init_trained_weights)
    print('raw trained weights:', raw_trained_weights)

    d = transform_seqs_to_input(DESIRE, DESIRE)[0]
    print(progress_weights)
    plot_progress_weights(progress_weights, EPOCHS, d)

def main():
    #measure()
    #measure_cnst_model()
    training()

if __name__ == "__main__":
    main()
