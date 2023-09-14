import random
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, add, ThresholdedReLU, Lambda, Reshape, concatenate, Cropping1D, Flatten, LeakyReLU, ELU
from tensorflow.python.client import device_lib

def transform_seqs_to_input(seqA, seqB):
    matching_pairs = []
    input_length_x = 0

    matching_pairs.append([int(seqA[0]), int(seqB[0])])
    if len(seqA) == 1 and len(seqB) == 1:
        return matching_pairs
    else:
        input_length_x = len(seqA)
        match_layers_i = (input_length_x * 2) - 1
    
    start_i = 1
    end_i = 2

    for l in range(match_layers_i):
        if l < input_length_x - 1:
            i, j = [*reversed(range(0, end_i))], [*range(0, end_i)]
            for n in range(len(i)):
                if j[n] < len(seqB):
                    pair = [int(seqA[i[n]]), int(seqB[j[n]])]
                    matching_pairs.append(pair)
            end_i += 1
        else:
            i, j = [*reversed(range(start_i, input_length_x))], [*range(start_i, input_length_x)]
            for n in range(len(i)):
                if j[n] < len(seqB):
                    pair = [int(seqA[i[n]]), int(seqB[j[n]])]
                    matching_pairs.append(pair)
            start_i += 1
            if start_i > len(seqB):
                break

    return matching_pairs

def transform_input_for_generate(input):
    x = []
    y = []
    for pair in input:
        x.append(pair[0])
        y.append(pair[1])
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

def min_module(i, j, k):
    input = Input(shape=(2,))
    x = Dense(2, activation='relu', use_bias=True)(input)
    combined = concatenate([x, input])

    layer_name = 'result_pixel_' + str(i) + str(j) + '_' + str(k)
    z = Dense(1, activation='relu', use_bias=True, name=layer_name)(combined)
    model = Model(inputs=input, outputs=z)

    w1 = model.layers[1].get_weights()
    w1[0][0], w1[0][1] = [-1.0, 1.0], [1.0, -1.0]
    w2 = model.layers[3].get_weights()
    w2[0][0], w2[0][1], w2[0][2], w2[0][3] = -0.5, -0.5, 0.5, 0.5

    model.layers[1].set_weights(w1)
    model.layers[3].set_weights(w2)

    model.trainable = False

    return model

def minimum(i, j):
    input = Input(shape=(3,))
    comp1_pair = Lambda(lambda x: x[:, :2], output_shape=(2,))(input)
    comp2_input = Lambda(lambda x: x[:, 2:], output_shape=(1,))(input)
    
    m = min_module(i, j, 1)(comp1_pair)
    comp2_pair = concatenate([comp2_input, m])
    output = min_module(i, j, 2)(comp2_pair)
    
    model = Model(inputs=input, outputs=output)
    model.trainable = False
    return model

def align_model_for_N(seq_length_x, seq_length_y, matching_pair_number):
    input = Input(shape=(2, matching_pair_number), name='input')

    y = Cropping1D((1,0))(input)
    x = Cropping1D((0,1))(input)
    y = Flatten()(y)
    x = Flatten()(x)

    out = {}
    start_i = tf.Variable(0, name='start_i')
    step = tf.Variable(2, name='step')
    for i in range(seq_length_y):
        a = int(start_i)
        layername = 'for_gen_dense_' + str(i+1)
        z = Dense(1, activation='relu', name=layername, use_bias=False)(y[:matching_pair_number, a:a+1])
        out[layername] = z
        start_i = start_i + int(step)
        step = step + 1

    comp_i, comp_j = tf.Variable(1, name='comp_i'), tf.Variable(2, name='comp_j')
    start_sentinel, end_sentinel = tf.Variable(1, name='start_sentinel'), tf.Variable(2, name='end_sentinel')
    pair_i = tf.Variable(1, name='pair_i')
    cropping_start_i, cropping_end_i = tf.Variable(0, name='cropping_start_i'), tf.Variable(matching_pair_number-1, name='cropping_end_i')

    calc_layer = (seq_length_x * 2) - 1
    test_dict = {}

    a = int(cropping_start_i)
    b = int(cropping_end_i)
    c = int(pair_i)
    y_dense_layer_name = 'for_gen_dense_1'
    densed_y = out[y_dense_layer_name]
    x_char = Lambda(lambda x: x[:, 0:1], output_shape=(None, 1))(x)
    debug_name = 'matching_debug_' + str(c)
    pair_11 = concatenate([x_char, densed_y], name=debug_name)
    ext_gaps = Dense(2, activation='relu', name='first_calc_gap_layer')(pair_11)
    cropping_start_i = cropping_start_i + 1
    cropping_end_i = cropping_end_i - 1

    min1 = min_module(1, 1, 1)(ext_gaps)
    matching1 = matching_module()(pair_11)
    combined = concatenate([min1, matching1])
    z = min_module(1, 1, 2)(combined)
    result_pixel_11 = concatenate([ext_gaps, z], name='input_pixel_1_1')
    pair_i = pair_i + 1

    if seq_length_x == 1 and seq_length_y == 1:
        output = z
        return Model(inputs=input, outputs=output)
    else:
        m = 'input_pixel_1_1'
        test_dict[m] = result_pixel_11
        n = 'result_pixel_1_1'
        test_dict[n] = z

        unbalance_flag = True
        for calc_layer_i in range(calc_layer):
            if calc_layer_i < seq_length_x - 1:
                comp_i, comp_j = start_sentinel, end_sentinel
                while comp_i <= end_sentinel:
                    if comp_i <= seq_length_y:
                        input_layer_name = 'input_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        before_input_layer_name = 'before_input_' + str(int(comp_i)) + '_' + str(int(comp_j)) 

                        a = int(cropping_start_i)
                        b = int(cropping_end_i)
                        pair_name = 'pair_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        c = int(pair_i)
                        y_i = int(comp_i)
                        y_dense_layer_name = 'for_gen_dense_' + str(y_i)
                        densed_y = out[y_dense_layer_name]
                        x_char = Flatten()(x[0, (c-1):c])
                        debug_name = 'matching_debug_' + str(c)
                        pair = concatenate([x_char, densed_y], name=debug_name)
                        matching = matching_module()(pair)  

                        if comp_i == 1:
                            previous_input_pixel_name = 'input_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j)-1)
                            previous_result_pixel_name = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j)-1)
                            previous_input = test_dict[previous_input_pixel_name]
                            previous_result = test_dict[previous_result_pixel_name]
                            g = Lambda(lambda x: x[:, 0:1], output_shape=(None, 1))(previous_input)
                            before_input = concatenate([g, previous_result, matching], name=before_input_layer_name)
                            
                            input_pixel = Dense(3, activation='relu', name=input_layer_name)(before_input)
                            result_pixel = minimum(int(comp_i), int(comp_j))(input_pixel)

                            m = 'input_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                            test_dict[m] = input_pixel                        
                            n = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                            test_dict[n] = result_pixel
                            if unbalance_flag == True:
                                unbalance_flag = False
                        elif comp_j == 1:
                            previous_input_pixel_name = 'input_pixel_' + str(int(comp_i)-1) + '_' + str(int(comp_j))
                            previous_result_pixel_name = 'result_pixel_' + str(int(comp_i)-1) + '_' + str(int(comp_j))
                            previous_input = test_dict[previous_input_pixel_name]
                            previous_result = test_dict[previous_result_pixel_name]
                            g = Lambda(lambda x: x[:, 0:1], output_shape=(None, 1))(previous_input)
                            before_input = concatenate([g, previous_result, matching], name=before_input_layer_name)

                            input_pixel = Dense(3, activation='relu', name=input_layer_name)(before_input)
                            result_pixel = minimum(int(comp_i), int(comp_j))(input_pixel)

                            m = 'input_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                            test_dict[m] = input_pixel 
                            n = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                            test_dict[n] = result_pixel
                            if unbalance_flag == True:
                                unbalance_flag = False
                        else:
                            previous_result_pixel_name1 = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j)-1)
                            previous_result1 = test_dict[previous_result_pixel_name1]
                            previous_result_pixel_name2 = 'result_pixel_' + str(int(comp_i)-1) + '_' + str(int(comp_j))
                            previous_result2 = test_dict[previous_result_pixel_name2]
                            previous_result_pixel_name3 = 'result_pixel_' + str(int(comp_i)-1) + '_' + str(int(comp_j)-1)
                            previous_result3 = test_dict[previous_result_pixel_name3]
                            before_input = concatenate([previous_result1, previous_result2, previous_result3, matching], name=before_input_layer_name)

                            input_pixel = Dense(3, activation='relu', name=input_layer_name)(before_input)
                            result_pixel = minimum(int(comp_i), int(comp_j))(input_pixel)

                            m = 'input_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                            test_dict[m] = input_pixel 
                            n = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                            test_dict[n] = result_pixel
                            
                            if unbalance_flag == True:
                                unbalance_flag = False
                        
                        cropping_start_i = cropping_start_i + 1
                        cropping_end_i = cropping_end_i - 1

                    comp_i, comp_j = (comp_i + 1), (comp_j - 1)
                    pair_i = pair_i + 1
                    if unbalance_flag == True:
                        pair_i = pair_i - 1
                    unbalance_flag = True
                if int(end_sentinel) + 1 <= seq_length_x:
                    end_sentinel = end_sentinel + 1
            else:
                start_sentinel = start_sentinel + 1
                comp_i, comp_j = start_sentinel, end_sentinel

                while comp_i <= end_sentinel:
                    if comp_i <= seq_length_y:
                        a = int(cropping_start_i)
                        b = int(cropping_end_i)
                        pair_name = 'pair_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        before_input_layer_name = 'before_input_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        input_layer_name = 'input_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        c = int(pair_i)
                        y_i = int(comp_i)
                        y_dense_layer_name = 'for_gen_dense_' + str(y_i)
                        densed_y = out[y_dense_layer_name]
                        x_char = Flatten()(x[0, (c-1):c])
                        debug_name = 'matching_debug_' + str(c)
                        pair = concatenate([x_char, densed_y], name=debug_name)
                        matching = matching_module()(pair)

                        previous_result_pixel_name1 = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j)-1)
                        previous_result1 = test_dict[previous_result_pixel_name1]
                        previous_result_pixel_name2 = 'result_pixel_' + str(int(comp_i)-1) + '_' + str(int(comp_j))
                        previous_result2 = test_dict[previous_result_pixel_name2]
                        previous_result_pixel_name3 = 'result_pixel_' + str(int(comp_i)-1) + '_' + str(int(comp_j)-1)
                        previous_result3 = test_dict[previous_result_pixel_name3]
                        before_input = concatenate([previous_result1, previous_result2, previous_result3, matching], name=before_input_layer_name)

                        input_pixel = Dense(3, activation='relu', name=input_layer_name)(before_input)
                        result_pixel = minimum(int(comp_i), int(comp_j))(input_pixel)

                        m = 'input_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        test_dict[m] = input_pixel 
                        n = 'result_pixel_' + str(int(comp_i)) + '_' + str(int(comp_j))
                        test_dict[n] = result_pixel

                        cropping_start_i = cropping_start_i + 1
                        cropping_end_i = cropping_end_i - 1

                        if unbalance_flag == True:
                            unbalance_flag = False

                    comp_i, comp_j = (comp_i + 1), (comp_j - 1)
                    pair_i = pair_i + 1
                    if unbalance_flag == True:
                        pair_i = pair_i - 1
                    unbalance_flag = True
                    
                    if start_sentinel == end_sentinel:
                        return Model(inputs=input, outputs=result_pixel)

def set_weight_for_debug(model, seq_len_x, seq_len_y, matching_pair):
    print('setting weigths ...')

    for i in range(seq_len_y):
        lname = 'for_gen_dense_' + str(i+1)
        weights = model.get_layer(lname).get_weights()
        #weights[0][0][0] = 1

        weights[0][0][0] = random.uniform(0,1)
        model.get_layer(lname).set_weights(weights)

    w = model.get_layer('first_calc_gap_layer').get_weights()
    w[0][0][0], w[0][0][1] = 0, 0
    w[0][1][0], w[0][1][1] = 0, 0
    w[1][0], w[1][1] = 2, 2
    model.get_layer('first_calc_gap_layer').set_weights(w)
    model.get_layer('first_calc_gap_layer').trainable = False

    if seq_len_x > 1:
        calc_layer = (seq_len_x * 2) - 1
        comp_i, comp_j = 1, 2
        start_sentinel, end_sentinel = 1, 2

        for calc_layer_i in range(calc_layer):
            if calc_layer_i < seq_len_x - 1:
                comp_i, comp_j = start_sentinel, end_sentinel
                while comp_i <= end_sentinel:
                    if comp_i <= seq_len_y:
                        input_layer_name = 'input_' + str(comp_i) + '_' + str(comp_j)
                        w = model.get_layer(input_layer_name).get_weights()
                        if comp_i == 1:
                            w[0][0][0], w[0][0][1], w[0][0][2] = 1, 0, 1
                            w[0][1][0], w[0][1][1], w[0][1][2] = 0, 1, 0
                            w[0][2][0], w[0][2][1], w[0][2][2] = 0, 0, 1
                            w[1][0], w[1][1], w[1][2] = 1, 1, -1
                        elif comp_j == 1:
                            w[0][0][0], w[0][0][1], w[0][0][2] = 1, 0, 1
                            w[0][1][0], w[0][1][1], w[0][1][2] = 0, 1, 0
                            w[0][2][0], w[0][2][1], w[0][2][2] = 0, 0, 1
                            w[1][0], w[1][1], w[1][2] = 1, 1, -1
                        else:
                            w[0][0][0], w[0][0][1], w[0][0][2] = 1, 0, 0
                            w[0][1][0], w[0][1][1], w[0][1][2] = 0, 1, 0
                            w[0][2][0], w[0][2][1], w[0][2][2] = 0, 0, 1
                            w[0][3][0], w[0][3][1], w[0][3][2] = 0, 0, 1
                            w[1][0], w[1][1], w[1][2] = 1, 1, 0 

                        model.get_layer(input_layer_name).set_weights(w)
                        model.get_layer(input_layer_name).trainable = False

                    comp_i, comp_j = (comp_i + 1), (comp_j - 1)
                if end_sentinel + 1 <= seq_len_x:
                    end_sentinel += 1
            else:
                start_sentinel = start_sentinel + 1
                comp_i, comp_j = start_sentinel, end_sentinel

                while comp_i <= end_sentinel:
                    if comp_i <= seq_len_y:
                        input_layer_name = 'input_' + str(comp_i) + '_' + str(comp_j)
                        
                        w = model.get_layer(input_layer_name).get_weights()
                        w[0][0][0], w[0][0][1], w[0][0][2] = 1, 0, 0
                        w[0][1][0], w[0][1][1], w[0][1][2] = 0, 1, 0
                        w[0][2][0], w[0][2][1], w[0][2][2] = 0, 0, 1
                        w[0][3][0], w[0][3][1], w[0][3][2] = 0, 0, 1
                        w[1][0], w[1][1], w[1][2] = 1, 1, 0 
                        model.get_layer(input_layer_name).set_weights(w)
                        model.get_layer(input_layer_name).trainable = False

                    comp_i, comp_j = (comp_i + 1), (comp_j - 1)

def froozen_align_model(model):
    print('froozen parameters in a network for alignment ...')
    layers = model.layers
    for layer in layers:
        if 'for_gen_dense' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

def measure_cnst_model():
    SEQ_LEN_X = 10
    SEQ_LEN_Y = 2
    PAIRS_LEN = SEQ_LEN_X * SEQ_LEN_Y

    N = 5
    times = []
    average = 0
    for i in range(N):
        cnst_model_s_time = time.perf_counter()
        model = align_model_for_N(SEQ_LEN_X, SEQ_LEN_Y, PAIRS_LEN)
        cnst_model_e_time = time.perf_counter() - cnst_model_s_time
        
        set_weight_for_debug(model, SEQ_LEN_X, SEQ_LEN_Y, PAIRS_LEN)

        average += cnst_model_e_time
        times.append(cnst_model_e_time)

    print(N, 'trial average construct model time:', average / N)
    print('min=', min(times), 'max=', max(times), np.argmax(times), 'median=', np.median(times), 'std=', np.std(times))

def measure():
    filename = './sampledata/predtime_length_10_2_levenshtein.csv'
    print('test file :', filename)

    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line.rstrip('\n')
            lines.append(line)
    
    confirm_l = lines[0]
    sp = confirm_l.split(',')
    x, y = sp[0], sp[1]
    pairs = transform_seqs_to_input(x, y)
    SEQ_LEN_X = len(x)
    SEQ_LEN_Y = len(y)
    PAIRS_LEN = len(pairs)

    model = align_model_for_N(SEQ_LEN_X, SEQ_LEN_Y, PAIRS_LEN)
    set_weight_for_debug(model, SEQ_LEN_X, SEQ_LEN_Y, PAIRS_LEN)
    froozen_align_model(model)

    N = 500
    perform_data = []
    if len(lines) < N:
        shortage_times = math.ceil(N / len(lines))
        perform_data = lines * shortage_times
        perform_data = perform_data[:N]
    else:
        perform_data = lines[:N]
    
    average_time = 0
    correct = 0
    times = []
    for data in perform_data:
        testdata = data
        sp = testdata.split(',')
        x, y, c_score = sp[0], sp[1], int(sp[2])
        input = transform_seqs_to_input(x, y)
        extra_input = [transform_input_for_generate(input)]

        x = tf.constant(extra_input)
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
        m = markers[w_i%len(markers)]
        plt.plot(x, y, marker=m, label="weigth" + str(w_i+1), color=color_cycle[w_i])
        plt.hlines(desire[w_i], 0, epoch, color=color_cycle[w_i] ,linestyles='dotted')
        w_i += 1

    plt.xlabel('epochs')
    plt.ylabel('weight')
    plt.ylim(-0.5,1.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.savefig('weights.png')  

def training():
    # data and model setting
    EPOCHS = 20
    DESIRE = '11'
    LEN = len(DESIRE)

    filename = './sampledata/desired_length_' + str(LEN) + '_levenshtein_2.csv'
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            line.rstrip('\n')
            lines.append(line)

    confirm_l = lines[0]
    sp = confirm_l.split(',')
    x, y = sp[0], sp[1]
    pairs = transform_seqs_to_input(x, y)
    SEQ_LEN_X = len(x)
    SEQ_LEN_Y = len(y)
    PAIRS_LEN = len(pairs)

    model = align_model_for_N(SEQ_LEN_X, SEQ_LEN_Y, PAIRS_LEN)
    set_weight_for_debug(model, SEQ_LEN_X, SEQ_LEN_Y, PAIRS_LEN)
    froozen_align_model(model)
    #model.summary()

    init_trained_weights = []
    for layer in model.layers:
        if 'for_gen_dense' in layer.name:
            weight = layer.get_weights()
            init_trained_weights.append(float(weight[0][0]))
    init_trained_weights.reverse()

    progress_weights = [] 
    progress_weight = []
    for i in range(LEN):
        lname = 'for_gen_dense_' + str(i+1)
        weights = model.get_layer(lname).get_weights()
        progress_weight.append(weights[0][0][0])
    progress_weights.append(progress_weight)

    #optimizer = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.1)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    progress_grads = []
    for epoch in range(EPOCHS):
        print('Epoch', epoch)

        loss = tf.Variable(0.0, name='loss')
        with tf.GradientTape() as tape:
            for line in lines:
                sp = line.split(',')
                x, y, true_score = sp[0], sp[1], int(sp[2])
                input = transform_seqs_to_input(x, y)
                input = transform_input_for_generate(pairs)
                input = tf.constant([input])
                logit = model(input, training=True)
                loss = loss + loss_fn(true_score, logit)
            batch_loss = loss / len(lines)
            print(batch_loss)
            grads = tape.gradient(batch_loss, model.trainable_weights)
            #print(grads)
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

    d = [int(i) for i in DESIRE]
    print(progress_weights)
    plot_progress_weights(progress_weights, EPOCHS, d)

def main():
    #measure_cnst_model()
    #measure()
    training()

if __name__ == "__main__":
    main()
