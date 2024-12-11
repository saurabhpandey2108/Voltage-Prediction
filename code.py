import numpy as np
import time
from scipy.interpolate import interp1d
import pandas as pd


class DeepNeuralNetwork():
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes

        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not supported, please use 'relu' or 'sigmoid' instead.")

    
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}

    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        output_layer = self.sizes[2]

        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(hidden_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((hidden_layer, 1)) * np.sqrt(1./hidden_layer),
            "W3": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b3": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        return params

    def initialize_adam_optimizer(self):
        adam_opt = {
            "m": {key: np.zeros_like(value) for key, value in self.params.items()},
            "v": {key: np.zeros_like(value) for key, value in self.params.items()},
        }
        return adam_opt

    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.activation(self.cache["Z2"])
        self.cache["Z3"] = np.matmul(self.params["W3"], self.cache["A2"]) + self.params["b3"]
        self.cache["A3"] = self.cache["Z3"].T  
        return self.cache["A3"]

    def back_propagate(self, y, output):
        current_batch_size = y.shape[0]

        dZ3 = output - y
        dW3 = (1./current_batch_size) * np.matmul(dZ3.T, self.cache["A2"].T)
        db3 = (1./current_batch_size) * np.sum(dZ3.T, axis=1, keepdims=True)

        dA2 = np.matmul(self.params["W3"].T, dZ3.T)
        dZ2 = dA2 * self.activation(self.cache["Z2"], derivative=True)
        dW2 = (1./current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1./current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1./current_batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1./current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        return self.grads

    def rmse_loss(self, y, output):
        """
            RMSE Loss:
            L(y, y_hat) = sqrt(mean((y - y_hat)^2))
        """
        error = y - output
        loss = np.sqrt(np.mean(error ** 2))
        return loss

    def optimize(self, l_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        t = 1
        for key in self.params:
            self.adam_opt['m'][key] = beta1 * self.adam_opt['m'][key] + (1 - beta1) * self.grads[key]
            self.adam_opt['v'][key] = beta2 * self.adam_opt['v'][key] + (1 - beta2) * (self.grads[key] ** 2)

            m_hat = self.adam_opt['m'][key] / (1 - beta1 ** t)
            v_hat = self.adam_opt['v'][key] / (1 - beta2 ** t)

            self.params[key] -= l_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def train(self, dataset, epochs=10, batch_size=64, optimizer='adam', l_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        if self.optimizer == 'adam':
            self.adam_opt = self.initialize_adam_optimizer()

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train loss={:.2f}"

        for i in range(self.epochs):
            total_loss = 0
            num_batches = len(dataset) // self.batch_size

            for j in range(num_batches):
                batch = dataset[j * self.batch_size:(j + 1) * self.batch_size]
                x = batch[:, :-8]  # First 15 columns as input
                y = batch[:, -8:]  # Last 8 columns as output

                output = self.feed_forward(x)
                grad = self.back_propagate(y, output)
                self.optimize(l_rate=l_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

                total_loss += self.rmse_loss(y, output)

            print(template.format(i + 1, time.time() - start_time, total_loss / num_batches))


    def CalVoltage(C1, C2, R0, R1, R2, gamma1, M0, M, i, ir1=0, ir2=0, z=100, h=0, s=0):
        Ts = 0.0001 
        Q = 10 * 3600  
        n = 1  

        z = z - Ts * n * i / Q
        ir1 = np.exp(-Ts / (R1 * C1)) * ir1 + (1 - np.exp(-Ts / (R1 * C1))) * i
        ir2 = np.exp(-Ts / (R2 * C2)) * ir2 + (1 - np.exp(-Ts / (R2 * C2))) * i
        u = np.exp(-abs(n * i * gamma1 * Ts / Q))
        h = u * h - (1 - u) * (1.0 if i > 0 else 0.0)
        s = 1 if i > 0 else s
        vh = M0 * s + M * h
        v = OCV(z) + vh - R1 * ir1 - R2 * ir2 - R0 * i
        return round(v, 4), ir1, ir2, z, h, s

    def OCV(x_val):
        y = np.array([2.5, 2.5999, 2.757, 3.0026, 3.1401, 3.2088, 3.2383, 
                    3.2726, 3.2972, 3.3119, 3.3119, 3.3365, 3.3709, 3.4887, 3.5])
        x = np.array([0, 0.18474, 0.71411, 3.5374, 7.243, 12.36, 20.124, 
                    32.3, 44.828, 60.0004, 70.591, 84.708, 97.413, 99.707, 100])
        ocv_interp = interp1d(x, y, kind='linear', fill_value="extrapolate")
        return float(ocv_interp(x_val))



    def train(self, dataset, epochs=10, batch_size=64, optimizer='adam', l_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        if self.optimizer == 'adam':
            self.adam_opt = self.initialize_adam_optimizer()

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train loss={:.4f}"

        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = len(dataset) // self.batch_size

            for batch_idx in range(num_batches):
                
                batch = dataset[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                x = batch[:, :]   
                actual_voltages = batch[:, 5]  

                predicted_params = self.feed_forward(x)  # Shape: (batch_size, 8)

                
                ir1 = np.zeros(batch.shape[0])  # R1
                ir2 = np.zeros(batch.shape[0])  # R2
                z = np.ones(batch.shape[0]) * 100  # SOC
                h = np.zeros(batch.shape[0])  # hysteresis
                s = np.zeros(batch.shape[0])  # polarity
                predicted_voltages = []

                
                for idx in range(batch.shape[0]):
                    current = x[idx, 0]  
                    
                    C1, C2, R0, R1, R2, gamma1, M0, M = predicted_params[idx]
                    v, ir1[idx], ir2[idx], z[idx], h[idx], s[idx] = CalVoltage(
                        C1, C2, R0, R1, R2, gamma1, M0, M, current, ir1[idx],
                        ir2[idx], z[idx], h[idx], s[idx]
                    )
                    predicted_voltages.append(v)

                
                predicted_voltages = np.array(predicted_voltages).reshape(-1, 1)
                actual_voltages = actual_voltages.reshape(-1, 1)  

            
                loss = self.rmse_loss(actual_voltages, predicted_voltages)
                total_loss += loss

                
                grad = self.back_propagate(predicted_voltages, predicted_params)
                self.optimize(l_rate=l_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

            print(template.format(epoch + 1, time.time() - start_time, total_loss / num_batches))





if __name__ == "__main__":
    

    dataset = pd.read_excel('modified10.xlsx')

    dataset = dataset.to_numpy()

    
    input_size = 15
    hidden_layer_size = 64
    output_size = 8

    sizes = [input_size, hidden_layer_size, output_size]

    
    dnn = DeepNeuralNetwork(sizes, activation='relu')
    dnn.train(dataset, epochs=50, batch_size=32, optimizer='adam', l_rate=0.001)



