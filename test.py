import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np
from IPython.display import Image



#input 設定
U1_input = keras.Input(shape = (4,),name = 'U1 input')
U2_input = keras.Input(shape = (4,),name = 'U2 input')
U3_input = keras.Input(shape = (4,),name = 'U3 input')
U4_input = keras.Input(shape = (4,),name = 'U4 input')
U5_input = keras.Input(shape = (4,),name = 'U5 input')
U6_input = keras.Input(shape = (4,),name = 'U6 input')
corruption_input = keras.Input(shape = (8,),name = 'corruption input')



#resource1 network
U2_to_R1 = layers.Dense(32,activation = 'relu')(U2_input)
U2_to_R1_2 = layers.Dense(32,activation = 'relu')(U2_to_R1)
U2_to_R1_3 = layers.Dense(32,activation = 'relu')(U2_to_R1_2)
U2_to_R1_4 = layers.Dense(32,activation = 'relu')(U2_to_R1_3)
U2_to_R1_5 = layers.Dense(32,activation = 'relu')(U2_to_R1_4)
U2_to_R1_6 = layers.Dense(2,activation = None)(U2_to_R1_5)            #activation 還沒定

U3_to_R1 = layers.Dense(32,activation = 'relu')(U3_input)
U3_to_R1_2 = layers.Dense(32,activation = 'relu')(U3_to_R1)
U3_to_R1_3 = layers.Dense(32,activation = 'relu')(U3_to_R1_2)
U3_to_R1_4 = layers.Dense(32,activation = 'relu')(U3_to_R1_3)
U3_to_R1_5 = layers.Dense(32,activation = 'relu')(U3_to_R1_4)
U3_to_R1_6 = layers.Dense(2,activation = None)(U3_to_R1_5)             #activation 還沒定

U6_to_R1 = layers.Dense(32,activation = 'relu')(U6_input)
U6_to_R1_2 = layers.Dense(32,activation = 'relu')(U6_to_R1)
U6_to_R1_3 = layers.Dense(32,activation = 'relu')(U6_to_R1_2)
U6_to_R1_4 = layers.Dense(32,activation = 'relu')(U6_to_R1_3)
U6_to_R1_5 = layers.Dense(32,activation = 'relu')(U6_to_R1_4)
U6_to_R1_6 = layers.Dense(2,activation = None)(U6_to_R1_5)              #activation 還沒定
#resource1 end

S_1 = keras.layers.Add()([U2_to_R1_6, U3_to_R1_6, U6_to_R1_6])

#resource2 network
U1_to_R2 = layers.Dense(32,activation = 'relu')(U1_input)
U1_to_R2_2 = layers.Dense(32,activation = 'relu')(U1_to_R2)
U1_to_R2_3 = layers.Dense(32,activation = 'relu')(U1_to_R2_2)
U1_to_R2_4 = layers.Dense(32,activation = 'relu')(U1_to_R2_3)
U1_to_R2_5 = layers.Dense(32,activation = 'relu')(U1_to_R2_4)
U1_to_R2_6 = layers.Dense(2,activation = None)(U1_to_R2_5)               #activation 還沒定

U3_to_R2 = layers.Dense(32,activation = 'relu')(U3_input)
U3_to_R2_2 = layers.Dense(32,activation = 'relu')(U3_to_R2)
U3_to_R2_3 = layers.Dense(32,activation = 'relu')(U3_to_R2_2)
U3_to_R2_4 = layers.Dense(32,activation = 'relu')(U3_to_R2_3)
U3_to_R2_5 = layers.Dense(32,activation = 'relu')(U3_to_R2_4)
U3_to_R2_6 = layers.Dense(2,activation = None)(U3_to_R2_5)                #activation 還沒定

U5_to_R2 = layers.Dense(32,activation = 'relu')(U5_input)
U5_to_R2_2 = layers.Dense(32,activation = 'relu')(U5_to_R2)
U5_to_R2_3 = layers.Dense(32,activation = 'relu')(U5_to_R2_2)
U5_to_R2_4 = layers.Dense(32,activation = 'relu')(U5_to_R2_3)
U5_to_R2_5 = layers.Dense(32,activation = 'relu')(U5_to_R2_4)
U5_to_R2_6 = layers.Dense(2,activation = None)(U5_to_R2_5)                 #activation 還沒定
#resource2 end

S_2 = keras.layers.Add()([U1_to_R2_6, U3_to_R2_6, U5_to_R2_6])

#resource3 network
U1_to_R3 = layers.Dense(32,activation = 'relu')(U1_input)
U1_to_R3_2 = layers.Dense(32,activation = 'relu')(U1_to_R3)
U1_to_R3_3 = layers.Dense(32,activation = 'relu')(U1_to_R3_2)
U1_to_R3_4 = layers.Dense(32,activation = 'relu')(U1_to_R3_3)
U1_to_R3_5 = layers.Dense(32,activation = 'relu')(U1_to_R3_4)
U1_to_R3_6 = layers.Dense(2,activation = None)(U1_to_R3_5)                  #activation 還沒定

U4_to_R3 = layers.Dense(32,activation = 'relu')(U4_input)
U4_to_R3_2 = layers.Dense(32,activation = 'relu')(U4_to_R3)
U4_to_R3_3 = layers.Dense(32,activation = 'relu')(U4_to_R3_2)
U4_to_R3_4 = layers.Dense(32,activation = 'relu')(U4_to_R3_3)
U4_to_R3_5 = layers.Dense(32,activation = 'relu')(U4_to_R3_4)
U4_to_R3_6 = layers.Dense(2,activation = None)(U4_to_R3_5)                  #activation 還沒定

U6_to_R3 = layers.Dense(32,activation = 'relu')(U6_input)
U6_to_R3_2 = layers.Dense(32,activation = 'relu')(U6_to_R3)
U6_to_R3_3 = layers.Dense(32,activation = 'relu')(U6_to_R3_2)
U6_to_R3_4 = layers.Dense(32,activation = 'relu')(U6_to_R3_3)
U6_to_R3_5 = layers.Dense(32,activation = 'relu')(U6_to_R3_4)
U6_to_R3_6 = layers.Dense(2,activation = None)(U6_to_R3_5)                  #activation 還沒定
#resource3 end

S_3 = keras.layers.Add()([U1_to_R3_6, U4_to_R3_6, U6_to_R3_6])

#resource4 network
U2_to_R4 = layers.Dense(32,activation = 'relu')(U2_input)
U2_to_R4_2 = layers.Dense(32,activation = 'relu')(U2_to_R4)
U2_to_R4_3 = layers.Dense(32,activation = 'relu')(U2_to_R4_2)
U2_to_R4_4 = layers.Dense(32,activation = 'relu')(U2_to_R4_3)
U2_to_R4_5 = layers.Dense(32,activation = 'relu')(U2_to_R4_4)
U2_to_R4_6 = layers.Dense(2,activation = None)(U2_to_R4_5)                  #activation 還沒定

U4_to_R4 = layers.Dense(32,activation = 'relu')(U4_input)
U4_to_R4_2 = layers.Dense(32,activation = 'relu')(U4_to_R4)
U4_to_R4_3 = layers.Dense(32,activation = 'relu')(U4_to_R4_2)
U4_to_R4_4 = layers.Dense(32,activation = 'relu')(U4_to_R4_3)
U4_to_R4_5 = layers.Dense(32,activation = 'relu')(U4_to_R4_4)
U4_to_R4_6 = layers.Dense(2,activation = None)(U4_to_R4_5)                  #activation 還沒定

U5_to_R4 = layers.Dense(32,activation = 'relu')(U5_input)
U5_to_R4_2 = layers.Dense(32,activation = 'relu')(U5_to_R4)
U5_to_R4_3 = layers.Dense(32,activation = 'relu')(U5_to_R4_2)
U5_to_R4_4 = layers.Dense(32,activation = 'relu')(U5_to_R4_3)
U5_to_R4_5 = layers.Dense(32,activation = 'relu')(U5_to_R4_4)
U5_to_R4_6 = layers.Dense(2,activation = None)(U5_to_R4_5)                  #activation 還沒定
#resource4 end

S_4 = keras.layers.Add()([U2_to_R4_6, U4_to_R4_6, U5_to_R4_6])


stacked_symbol = tf.keras.layers.Concatenate(axis=1)([S_1, S_2, S_3, S_4])

encoded_symbol_normalizing = tf.keras.layers.Lambda(lambda x:x**0.5)(tf.math.reduce_mean(tf.math.square(stacked_symbol)))

encoded_symbol_original = tf.keras.layers.Lambda(lambda x:1/np.sqrt(2) * x)(tf.math.divide(stacked_symbol, encoded_symbol_normalizing))


#encoded_symbol = encoded_symbol_original + corruption_input

encoded_symbol = keras.layers.Add()([encoded_symbol_original, corruption_input])

output_input = layers.Dense(512,activation = 'relu')(encoded_symbol)

output_h2 = layers.Dense(512,activation = 'relu')(output_input)

output_h3 = layers.Dense(512,activation = 'relu')(output_h2)

output = layers.Dense(24,activation = None)(output_h3)

model = keras.Model(inputs=[U1_input,U2_input,U3_input,U4_input,U5_input,U6_input,corruption_input], outputs=output)
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,loss='mean_squared_error',metrics='accuracy')


#checkpoint_filepath = r"C:\Users\Mayor\Desktop\LEARNet\checkpoint\{}".format(Name)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r"C:\Users\Mayor\Desktop\NEW\checkpoint\cp",
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=True)

#keras.utils.plot_model(model, "c.png", show_shapes=True)

#plot_model(model,to_file='c.png')

#model.summary()


batch_size = 400
modulation_level =4
lin_space = np.arange(0,13,2) #不確定為啥要回圈這個範圍
print (lin_space)
SNR_range = 10
cha_mag = 2.0  #這參數不知道是啥(rly用的參數)
err_rate =[]   #不知道這是裝啥資料
ber_rate =[]   #裝BER

#train
for iterN in range(len(lin_space)):  #不知道為啥要回圈這個範圍
    EbN0dB = lin_space[iterN]
    N0 = 1.0 / np.power(10.0, EbN0dB/10.0)
    #print(EbN0dB)
    #print(N0)
    #cost_plot =[]  #應該是收集深度學的cost值

    if lin_space[iterN] == SNR_range:
        training_epochs = 300000    #100+SNR_range*30000  #epochs
        for epoch in range(training_epochs):
            #avg_cost = 0
            batch_ys = np.random.randint(4, size=(batch_size, 6))
            batch_y = np.zeros((batch_size, 24))
            for n in range(batch_size):
                for m in range(6):
                    batch_y[n, m * 4 + batch_ys[n, m]] = 1         #還是不太懂這個是啥? 可能是QAM過後的資訊?
            #print(batch_y)
            for i in range(6):
                globals()['x'+str(i)] = batch_y[:,i*4:(i+1)*4]

            noise_batch_r = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(batch_size, 4))   #白高斯雜訊_實部
            noise_batch_i = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(batch_size, 4))   #白高斯雜訊_虛部

            corruption_batch = np.hstack((noise_batch_r, noise_batch_i))                        #雜訊實虛合併

            #下面就是丟資料給深度學習開始訓練
            model.fit(x = [x0,x1,x2,x3,x4,x5,corruption_batch], y = batch_y, verbose = 0, callbacks=[model_checkpoint_callback])

            if epoch % 1000 == 0:
                print(model.evaluate(x = [x0,x1,x2,x3,x4,x5,corruption_batch], y = batch_y))
                print(epoch)

print('Learning Finished!')

#test
for iterN in range(len(lin_space)):  #不知為啥要迴圈
    #message = np.zeros((batch_size, 4), dtype=complex)
    EbN0dB = lin_space[iterN]
    N0 = 1.0 / np.power(10.0, EbN0dB / 10.0)
    test_batch_size = 100000
    test_ys = np.random.randint(4, size=(test_batch_size,6))
    test_y = np.zeros((test_batch_size,24))
    for n in range(test_batch_size):
        for m in range(6):
            test_y[n, m * 4 + test_ys[n, m]] = 1
            #test_y[n, 0] = 1

    for i in range(6):
        globals()['x'+str(i)] = test_y[:,i*4:(i+1)*4]

    noise_batch_test_r = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(test_batch_size, 4))
    noise_batch_test_i = (np.sqrt(N0 / 2.0)) * np.random.normal(0.0, size=(test_batch_size, 4))
    corruption_test_batch = np.hstack((noise_batch_test_r, noise_batch_test_i))

    model.evaluate(x = [x0,x1,x2,x3,x4,x5,corruption_test_batch], y = test_y)
    y_pred = model.predict(x = [x0,x1,x2,x3,x4,x5,corruption_test_batch])
    #print(y_pred)
    correct_prediction1 = tf.equal(tf.argmax(y_pred[:, 0:4], axis=1), tf.argmax(test_y[:, 0:4], axis=1))
    correct_prediction2 = tf.equal(tf.argmax(y_pred[:, 4:8], axis=1), tf.argmax(test_y[:, 4:8], axis=1))
    correct_prediction3 = tf.equal(tf.argmax(y_pred[:, 8:12], axis=1), tf.argmax(test_y[:, 8:12], axis=1))
    correct_prediction4 = tf.equal(tf.argmax(y_pred[:, 12:16], axis=1), tf.argmax(test_y[:, 12:16], axis=1))
    correct_prediction5 = tf.equal(tf.argmax(y_pred[:, 16:20], axis=1), tf.argmax(test_y[:, 16:20], axis=1))
    correct_prediction6 = tf.equal(tf.argmax(y_pred[:, 20:24], axis=1), tf.argmax(test_y[:, 20:24], axis=1))

    correct_prediction = [correct_prediction1, correct_prediction2, correct_prediction3, correct_prediction4, correct_prediction5, correct_prediction6]
    #算準確度 後面會用到
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))


    #算BER
    graycoding = tf.constant([[False, False], [False, True], [True, True], [True, False]], dtype=tf.bool)
    bit_error = []
    for i in range(6):
        bit_error.append(tf.reduce_mean(tf.cast(tf.math.logical_xor(tf.gather(graycoding, tf.argmax(y_pred[:, i * modulation_level:(i + 1) * modulation_level], axis=1)), tf.gather(graycoding, tf.argmax(test_y[:, i * modulation_level:(i + 1) * modulation_level], axis=1))), dtype=tf.float32)))

    BER = tf.reduce_mean(bit_error)
    ber_rate.append(BER)
print (lin_space)
print (ber_rate)
print("Finished!")
