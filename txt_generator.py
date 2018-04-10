#reference1: https://gist.github.com/karpathy/d4dee566867f8291f086
#reference2: https://gist.github.com/mikalv/3947ccf21366669ac06a01f39d7cff05

import tensorflow as tf
import numpy as np

#Set Hyperparameters
seq_len = 25
learning_rate = 1e-1
epoch = 200
batch_size = 32
hidden_size = 128
tf.reset_default_graph()

def readData(file_name):
    data = open(file_name, 'r').read()
    unique_chars = list(set(data))
    vocab_len = len(unique_chars)
    data_len = len(data)
    
    #Build dict for "character to occurance" and "occurance to character" 
    char_to_ix = { ch:i for i,ch in enumerate(unique_chars) }
    ix_to_char = { i:ch for i,ch in enumerate(unique_chars) }
    inputs = [list() for _ in range(len(data)-seq_len)]
    targets = [list() for _ in range(len(data)-seq_len)]
    p = 0
    
    #build input and target vectors using the dict defined above
    for i in range(data_len-seq_len):
        inputs[i] = [char_to_ix[ch] for ch in data[i:i+seq_len]]
        targets[i] = [char_to_ix[data[i+seq_len]]]
    
    #One Hot encoding for inputs and targets
    inputs_1h = np.zeros((len(inputs),seq_len,vocab_len))
    targets_1h = np.zeros((len(inputs),vocab_len))
    for i, input_ch in enumerate(inputs):
        for j, ch in enumerate(input_ch):
            inputs_1h[i,j,[inputs[i][j]]] = 1
        targets_1h[i,[targets[i]]] = 1
    return inputs_1h, targets_1h, vocab_len, data_len, ix_to_char

#######################################################################################
#                                    Defining Model                                   #
#######################################################################################
def rnn_model(inputs_1h, targets_1h, vocab_len, ix_to_char):
    x = tf.placeholder(tf.float32,[None, seq_len, vocab_len])
    y = tf.placeholder(tf.float32,[None, vocab_len])
    weights = tf.Variable(tf.random_normal([hidden_size, vocab_len]))
    bias = tf.Variable(tf.random_normal([vocab_len]))
    
    cell_in = x
    cell_in = tf.transpose(cell_in, [1, 0, 2])
    cell_in = tf.reshape(cell_in, [-1, vocab_len])
    cell_in = tf.split(cell_in, seq_len, 0)
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    rnn_outputs, rnn_states = tf.contrib.rnn.static_rnn(cell, cell_in, dtype=tf.float32)
    logits = tf.matmul(rnn_outputs[-1],weights) + bias

    prediction = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cost = tf.reduce_mean(prediction)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    total_batches = int(data_len/batch_size)
    
    for i in range(epoch):
        count = 0
        for _ in range(total_batches):
            inputs_batch, targets_batch = inputs_1h[count:count+batch_size], targets_1h[count:count+batch_size]
            count +=batch_size
            session.run([optimizer], feed_dict = {x:inputs_batch, y:targets_batch})

        #take one list of training set as initial characters to be used to make the predictions
        initial = inputs_batch[:1:]

        #print the initial characters
        init_chars = ' '
        for j in initial[0]:
            ix = np.where(j == max(j))
            init_chars += ix_to_char[ix[0][0]]
        
        #predict the next characters
        for k in range(500):
            if k>0:
                initial_minus_firstChar = initial[:,1:,:]
                initial = np.append(initial_minus_firstChar, np.reshape(predictedProb,[1, 1, vocab_len]), axis=1)
            predicted = session.run([logits], feed_dict={x:initial})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            expo_pred = np.exp(predicted)
            pred = expo_pred/ np.sum(expo_pred)
            predictedProb = np.random.multinomial(1, pred, 1)
            init_chars += ix_to_char[np.argmax(predictedProb)]
        print("\n \n Result from epoch ", i," : ", init_chars)
    finalText = open("generatedText.txt",'w+')
    finalText.write(init_chars)
    session.close()

inputs_1h, targets_1h, vocab_len, data_len, ix_to_char = readData("input.txt")
rnn_model(inputs_1h, targets_1h, vocab_len, ix_to_char)
