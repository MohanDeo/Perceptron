# ## Perceptron


import numpy as np


class Perceptron:
    def __init__(self, number_of_input_units, number_of_hidden_units, number_of_output_units, learning_rate=0.5,
                 weight_dict=0):
        # We have two layers of weights, which we use when we pass from one layer into another
        self.input_layer_weights = np.reshape(
            np.random.rand(number_of_hidden_units * (number_of_input_units + 1)),
            (number_of_hidden_units, (number_of_input_units + 1))
        )
        self.hidden_layer_weights = np.reshape(
            np.random.rand(number_of_output_units * (number_of_hidden_units + 1)),
            (number_of_output_units, (number_of_hidden_units + 1))
        )

        self.bias = -1
        self.learning_rate = learning_rate

        # If weight_dict is not 0, then set weights accordingly and create a function to just output model predicitons,
        # not training the model

    def transfer_function(self, linear_result, is_image=False):
        if is_image:
            return 2 / (1 + np.exp(-linear_result)) - 1
        return 1 / (1 + np.exp(-linear_result))

    def transfer_function_derivative(self, linear_results, is_image=False):
        if is_image:
            return 2 * self.transfer_function(linear_results) * (1 - self.transfer_function(linear_results))
        return self.transfer_function(linear_results) * (1 - self.transfer_function(linear_results))

    def compute_linear_result(self, layer_weights, input_to_layer):
        return np.dot(layer_weights, input_to_layer)

    def train(self, training_data, target_data, number_of_epochs, is_image=False):
        # For the autoencoder, the training data is the target as well
        # Take an input, which is a row of the training data
        error_history = np.zeros(number_of_epochs)
        for n in range(number_of_epochs):
            input_layer_weight_differences = np.zeros(self.input_layer_weights.shape)
            hidden_layer_weight_differences = np.zeros(self.hidden_layer_weights.shape)
            number_of_columns_input_layer_weight_differences = input_layer_weight_differences.shape[1]
            number_of_columns_hidden_layer_weight_differences = hidden_layer_weight_differences.shape[1]

            epoch_error = 0
            for i in range(len(training_data)):
                training_data_point = training_data[i]
                # This is what we want the output to end up being for this training data point
                target = target_data[i]

                # Feedforward
                # Compute this separately, as we will use it in backpropagation
                linear_result_hidden_layer = self.compute_linear_result(self.input_layer_weights, training_data_point)

                activation_for_hidden_layer = self.transfer_function(
                    linear_result_hidden_layer, is_image
                )

                # Pass the input layer's activation to the hidden layer
                # Add the bias back in for the hidden layer
                activation_for_hidden_layer = np.append(activation_for_hidden_layer, self.bias)

                # Compute this separately, as we will use it in backpropagation
                linear_result_output_layer = self.compute_linear_result(self.hidden_layer_weights,
                                                                        activation_for_hidden_layer)
                activation_for_output_layer = self.transfer_function(
                    linear_result_output_layer, is_image
                )

                # Compare the output to the target, using the MSE and accumulate the error to the total error for this epoch
                error = sum(0.5 * (target - activation_for_output_layer) ** 2)
                epoch_error = epoch_error + error

                # Now, backpropagate
                delta_signal_output = self.transfer_function_derivative(linear_result_output_layer, is_image) * (
                        target - activation_for_output_layer)

                for column_number in range(number_of_columns_hidden_layer_weight_differences):
                    hidden_layer_weight_differences[:, column_number] = hidden_layer_weight_differences[:,
                                                                        column_number] + delta_signal_output * \
                                                                        activation_for_hidden_layer[
                                                                            column_number]

                delta_signal_hidden = self.transfer_function_derivative(linear_result_hidden_layer, is_image) * (
                    # Don't include the last column of the hidden layer, to match the dimensions of input_layer_weight_differences
                    (delta_signal_output * self.hidden_layer_weights[:, :-1].T).T.sum(axis=0)
                )
                for column_number in range(number_of_columns_input_layer_weight_differences):
                    input_layer_weight_differences[:, column_number] = input_layer_weight_differences[:,
                                                                       column_number] + delta_signal_hidden * \
                                                                       training_data_point[
                                                                           column_number
                                                                       ]
            # Update the weights now to finish backprop
            # Paper suggests altering learning rate like this
            if is_image and n >= 10000:
                self.learning_rate = 0.001

            self.input_layer_weights = self.input_layer_weights + self.learning_rate * input_layer_weight_differences
            self.hidden_layer_weights = self.hidden_layer_weights + self.learning_rate * hidden_layer_weight_differences

            # Store the error, to track it over time
            error_history[n] = epoch_error

        return error_history

    def generate_output(self, input_data, is_image=False):
        predictions = []
        # Feedforward input and generate the output
        for i in range(len(input_data)):
            input_data_point = input_data[i]

            # Feedforward
            # Compute this separately, as we will use it in backpropagation
            activation_for_hidden_layer = self.transfer_function(
                self.compute_linear_result(self.input_layer_weights, input_data_point), is_image
            )

            # Pass the input layer's activation to the hidden layer
            # Add the bias back in for the hidden layer
            activation_for_hidden_layer = np.append(activation_for_hidden_layer, self.bias)

            activation_for_output_layer = self.transfer_function(
                self.compute_linear_result(self.hidden_layer_weights, activation_for_hidden_layer), is_image
            )

            predictions.append(np.round(activation_for_output_layer))

        return predictions

# perceptron = Perceptron(2, 1, 2)

# input_xor1 = np.array([[0,0,-1], [0,1,-1], [1,0,-1], [1,1,-1]])
# input_xor2 = np.array([[0,0], [0,1], [1,0], [1,1]])
# output_xor = np.array([0,1,1,0])

# errors_from_training = perceptron.train(input_xor1, input_xor2, 5000)

# plt.plot(range(5000), errors_from_training)

# perceptron = Perceptron(2, 1, 2)

# input_xor1 = np.array([[0,0,-1], [0,1,-1], [1,0,-1], [1,1,-1]])
# input_xor2 = np.array([[0,0], [0,1], [1,0], [1,1]])
# output_xor = np.array([0,1,1,0])

# errors_from_training = perceptron.train(input_xor1, input_xor2, 20000)
# plt.plot(range(20000), errors_from_training)
# predicitions = perceptron.generate_output(input_xor1)

# print(predicitions)
