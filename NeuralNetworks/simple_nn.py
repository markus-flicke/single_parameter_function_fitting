from keras.models import Sequential
from keras.layers import Dense



class SimpleNN:
    def __init__(self, hidden_layer_sizes = (20, 20, 20)):
        self.model = Sequential()
        self.model.add(Dense(hidden_layer_sizes[0], input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        for hidden_layer_size in hidden_layer_sizes[1:]:
            self.model.add(Dense(hidden_layer_size, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')
    
    def fit(self, x, y, epochs=500, batch_size=10):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, x):
        return self.model.predict(x)

    def fit_predict(self, x, y, epochs=500, batch_size=10):
        self.fit(x, y, epochs=epochs, batch_size=batch_size)
        return self.predict(x)
