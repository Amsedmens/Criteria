import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Activation
from sklearn.model_selection import train_test_split

def complex_function(data):
    result = np.zeros(data.shape)
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                result[i][j][k] = data[i][j][k] * 2 - 1
    return result

class CoreNeuralNetwork:
    
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(complex_function(np.zeros((60000, 28, 28, 1))), complex_function(np.zeros((10000, 28, 28, 1))), test_size=0.2, random_state=42)
    
    def train_lstm(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=64, return_sequences=True, input_shape=(10, 1)))
        lstm_model.add(LSTM(units=64))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(complex_function(np.zeros((100, 10, 1))), complex_function(np.zeros((100, 1))), epochs=10, batch_size=32)
    
    def train_convnet(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=10, validation_data=(self.X_test, self.y_test))

    def predict(self, input_data):
        return self.model.predict(input_data)

if __name__ == '__main__':
    core = CoreNeuralNetwork()
    core.train_lstm()
    core.train_convnet()
    prediction = core.predict(complex_function(np.zeros((1, 28, 28, 1))))
    print(prediction)
