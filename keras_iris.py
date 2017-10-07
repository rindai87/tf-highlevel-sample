# tf.estimator Quickstartによってirisデータがすでにローカル環境に
# 存在している前提のプログラムです。
import numpy as np
training_data = np.loadtxt('./iris_training.csv', delimiter=',', skiprows=1)
train_x = training_data[:, :-1]
train_y = training_data[:, -1]

test_data = np.loadtxt('./iris_test.csv', delimiter=',', skiprows=1)
test_x = test_data[:, :-1]
test_y = test_data[:, -1]

# ここからがKerasの処理
import tensorflow as tf
import tensorflow.contrib.keras as keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
backend = keras.backend


num_classes = 3
train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

# モデルの定義
model = Sequential()

# ネットワークの定義
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# モデルのサマリの確認
model.summary()

# モデルのコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 学習
history = model.fit(train_x, train_y,
                    batch_size=20,
                    epochs=2000,
                    verbose=0,
                    validation_data=(test_x, test_y))

# 学習モデルの評価
score = model.evaluate(test_x, test_y, verbose=0)

print(score[1])

backend.clear_session()
