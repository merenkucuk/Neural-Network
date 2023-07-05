import numpy as np
import pandas as pd
import os
from cv2 import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from tabulate import tabulate

# First we open our folder with listdir
# Then, we open the train test validation folders in our folder
# Afterwards, add the image files inside the folders to the lists
# We create 2 lists for each folder for images and labels
folder_names = os.listdir('Vegetable Images')
train_label, test_label, valid_label, train_file, test_file, valid_file = [], [], [], [], [], [],
for k, folder in enumerate(folder_names):
    inner_folder = os.listdir('Vegetable Images/' + folder)
    for j, veg_folder in enumerate(inner_folder):
        inner_file = os.listdir('Vegetable Images/' + folder + "/" + veg_folder)
        for file in inner_file:
            if folder == "test":
                test_file.append("Vegetable Images/" + folder + "/" + veg_folder + "/" + file)
                test_label.append(j)
            elif folder == "train":
                train_file.append("Vegetable Images/" + folder + "/" + veg_folder + "/" + file)
                train_label.append(j)
            elif folder == "validation":
                valid_file.append("Vegetable Images/" + folder + "/" + veg_folder + "/" + file)
                valid_label.append(j)

# Next, we create separate dataframes for train test and validation
df_train = pd.DataFrame({
    'vegetable_name': train_file,
    'vegetable_label': train_label
})

df_test = pd.DataFrame({
    'vegetable_name': test_file,
    'vegetable_label': test_label
})

df_val = pd.DataFrame({
    'vegetable_name': valid_file,
    'vegetable_label': valid_label
})


# With getTrainTestValid function we convert our dataframes into image and label data.
# In the getTrainTestValid function we first read the images, convert them to grayscale, resize and flatten them.
# Then we shuffle the labels and indexes and return 2 different numpy arrays with images and labels.
def getTrainTestValid(df):
    images = []
    for i, file_path in enumerate(df.vegetable_name.values):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        img = img.flatten() / 255
        images.append(img)

    y = df['vegetable_label']
    label_num = len(y)

    # idx_rand = np.random.choice(range(label_num), label_num, replace=False)
    idx_rand = np.random.permutation(label_num)
    y = shuffle(y)
    names_list = []
    labels_list = []
    for i in range(label_num):
        names_list.append(images[idx_rand[i]])
        labels_list.append(y[idx_rand[i]])

    return np.array(names_list), np.array(labels_list)


# We get two numpy arrays each for train test and validation.
x_train, y_train = getTrainTestValid(df_train)
x_test, y_test = getTrainTestValid(df_test)
x_val, y_val = getTrainTestValid(df_val)


# In our neural network, first of all, we take the number of data, label and hidden layer as definite parameters.
# The rest of the parameters have default values because they are the parameters that we can change.
# The remaining parameters are activation function, number of classes, learning rate, hidden layer size, batch size and epoch value.
class NeuralNetwork(object):
    def __init__(self, train_names, train_labels, n_hidden_layers, size_hidden_layer=128, batch_size=32, epoch=101,
                 activation="relu", n_classes=15, learning_rate=0.001):
        self.X = train_names
        self.y = train_labels
        self.number_of_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.n_classes = n_classes
        self.size_hl = size_hidden_layer

        self.size_step = 5e-2
        self.regularization = 1e-3
        self.weight = []
        self.bias = []
        self.hl_score = []
        self.activation = activation

    # initialize last layer
    def last_layer(self):
        self.softmax_weights = np.random.randn(self.size_hl, self.n_classes)
        self.softmax_weights /= 100
        arr1 = []
        for i in range(0, self.n_classes):
            arr1.append(0.)
        arr2 = [arr1]
        biases = np.array(arr2)
        self.softmax_biases = biases

    # initialize hidden_layer
    def hidden_layer(self, n_inputs):
        weights = np.random.randn(n_inputs, self.size_hl)
        self.weight.append(weights * 1 / 100)
        arr1 = []
        for i in range(0, self.size_hl):
            arr1.append(0.)
        arr2 = [arr1]
        biases = np.array(arr2)
        self.bias.append(biases)

    def softmax(self, scores):
        # numerical stability
        numerator = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        denumerator = np.sum(numerator, axis=1, keepdims=True)
        res = numerator / denumerator
        return res

    def relu(self, data):
        res = data.copy()
        res[res < 0] = 0
        return res

    def sigmoid(self, data):
        denum = 1 + np.exp(-1 * data)
        return np.reciprocal(denum)

    def tanh(self, data):
        return np.tanh(data)

    def negativeloglikelihood(self, X, y):
        res = -1 * np.log(X + 1e-230)
        one_hot_y = np.zeros((y.size, 15))
        one_hot_y[np.arange(y.shape[0]), y] = 1
        return round(np.sum(one_hot_y * res) / 100, 7)

    def create_batch(self, batch_size=32):
        mini_batches = []
        no_of_batches = self.X.shape[0] // batch_size
        i = 0

        for i in range(no_of_batches):
            batch_idx_first = i * batch_size
            batch_idx_end = (i + 1) * batch_size
            X_mini = self.X[batch_idx_first:batch_idx_end]
            Y_mini = self.y[batch_idx_first:batch_idx_end]
            mini_batches.append((X_mini, Y_mini))

        if self.X.shape[0] % batch_size != 0:
            X_mini = self.X[(i + 1) * batch_size:]
            Y_mini = self.y[(i + 1) * batch_size:]
            mini_batches.append((X_mini, Y_mini))

        return mini_batches

    def backward(self, score):
        weight_list = []
        bias_list = []
        n_hl = self.number_of_hidden_layers
        weight_softmax = np.dot(self.hl_score[n_hl].transpose(), score)
        bias_softmax = np.sum(score, axis=0, keepdims=True)
        val_softmax = np.dot(score, self.softmax_weights.transpose())
        val_softmax[self.hl_score[n_hl] <= 0] = 0

        for i in range(self.number_of_hidden_layers):
            n_hl -= 1
            weight_list.insert(0, np.dot(self.hl_score[n_hl].transpose(), val_softmax))
            bias_list.insert(0, np.sum(val_softmax, axis=0, keepdims=True))
            val_softmax = np.dot(val_softmax, self.weight[n_hl].transpose())
            val_softmax[self.hl_score[n_hl] <= 0] = 0

        weight_softmax += self.regularization * self.softmax_weights
        for i in range(len(weight_list)):
            weight_list[i] += self.regularization * self.weight[i]

        for i in range(len(self.weight)):
            self.weight[i] += -self.size_step * weight_list[i]
            self.bias[i] += -self.size_step * bias_list[i]
        self.softmax_weights += -self.size_step * weight_softmax
        self.softmax_biases += -self.size_step * bias_softmax

    def gradient(self, X, y, data):
        X = self.X
        y = self.y
        grad_val = np.dot(X, data)
        grad_val = np.dot(X.transpose(), (grad_val - y))
        return grad_val

    def cost(self, X, y, data):
        X = self.X
        y = self.y
        cost = np.dot(X, data)
        cost_val = np.dot((cost - y).transpose(), (cost - y))
        cost_val /= 2
        return cost_val[0]

    def gradientDescent(self, learning_rate=0.001, batch_size=32):
        theta = np.zeros((self.X.shape[1], 1))
        error_list = []
        max_iters = 3
        for itr in range(max_iters):
            mini_batches = self.create_batch(batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                theta = theta - learning_rate * self.gradient(X_mini, y_mini, theta)
                error_list.append(self.cost(X_mini, y_mini, theta))

        return theta, error_list

    def train(self):
        for i in range(self.number_of_hidden_layers):
            if i == 0:
                self.hidden_layer(32 * 32)
            else:
                self.hidden_layer(self.size_hl)
        self.last_layer()
        batch_count = self.create_batch(self.batch_size)
        train_acc = []
        n_epoch = []
        train_loss = []
        for epoch_count in range(self.epoch):
            acc_bool = True
            loss_bool = True
            for batch in batch_count:
                batch_data = batch[0]
                batch_labels = batch[1]
                self.hl_score = []
                self.hl_score.append(batch_data)
                for i in range(self.number_of_hidden_layers):
                    if i == 0:
                        if self.activation == "relu":
                            target = self.relu(np.dot(batch_data, self.weight[i]) + self.bias[i])
                            self.hl_score.append(target)
                        elif self.activation == "sigmoid":
                            target = self.sigmoid(np.dot(batch_data, self.weight[i]) + self.bias[i])
                            self.hl_score.append(target)
                        elif self.activation == "tanh":
                            target = self.tanh(np.dot(batch_data, self.weight[i]) + self.bias[i])
                            self.hl_score.append(target)
                    else:
                        if self.activation == "relu":
                            target = self.relu(np.dot(target, self.weight[i]) + self.bias[i])
                            self.hl_score.append(target)
                        elif self.activation == "sigmoid":
                            target = self.sigmoid(np.dot(target, self.weight[i]) + self.bias[i])
                            self.hl_score.append(target)
                        elif self.activation == "tanh":
                            target = self.tanh(np.dot(target, self.weight[i]) + self.bias[i])
                            self.hl_score.append(target)
                target = np.dot(target, self.softmax_weights) + self.softmax_biases
                predictions = self.softmax(target)
                if loss_bool:
                    loss_bool = False
                if acc_bool and epoch_count % 10 == 0:
                    predicted_class = np.argmax(predictions, axis=1)
                    loss_value = self.negativeloglikelihood(predictions, batch_labels)
                    train_loss.append(loss_value)
                    n_epoch.append(epoch_count)
                    acc = (np.mean(predicted_class == batch_labels))
                    print("#################################")
                    print("Epoch", epoch_count, "/", self.epoch - 1)
                    print("loss ", loss_value)
                    print("training accuracy: ", acc)
                    train_acc.append(acc)
                    prec = precision_score(batch_labels, predicted_class, zero_division=1, average="weighted")
                    recall = recall_score(batch_labels, predicted_class, zero_division=1, average="weighted")
                    fscore = f1_score(batch_labels, predicted_class, zero_division=1, average="weighted")
                    print("Precision Score: ", prec)
                    print("Recall Score: ", recall)
                    print("F1 Score: ", fscore)
                    acc_bool = False
                back_prop = predictions
                back_prop[range(len(batch_labels)), batch_labels] -= 1
                back_prop /= len(batch_labels)
                self.backward(back_prop)
        train_loss = np.array(train_loss)
        train_acc = np.array(train_acc)
        epoch = np.array(n_epoch)
        # Make plot of train and valid accuracy
        # Set all features for plot
        plt.plot(epoch, train_acc)
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["Train Accuracy"], loc="lower right")
        plt.show()
        plt.plot(epoch, train_loss)
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Train Loss"], loc="lower right")
        plt.show()

    def test(self, test_set, test_label_set):
        for i in range(self.number_of_hidden_layers):
            if i == 0:
                if self.activation == "relu":
                    target = self.relu(np.dot(test_set, self.weight[i]) + self.bias[i])
                    self.hl_score.append(target)
                elif self.activation == "sigmoid":
                    target = self.sigmoid(np.dot(test_set, self.weight[i]) + self.bias[i])
                    self.hl_score.append(target)
                elif self.activation == "tanh":
                    target = self.tanh(np.dot(test_set, self.weight[i]) + self.bias[i])
                    self.hl_score.append(target)
            else:
                if self.activation == "relu":
                    target = self.relu(np.dot(target, self.weight[i]) + self.bias[i])
                    self.hl_score.append(target)
                elif self.activation == "sigmoid":
                    target = self.sigmoid(np.dot(target, self.weight[i]) + self.bias[i])
                    self.hl_score.append(target)
                elif self.activation == "tanh":
                    target = self.tanh(np.dot(target, self.weight[i]) + self.bias[i])
                    self.hl_score.append(target)
        target = np.dot(target, self.softmax_weights) + self.softmax_biases
        probabilities = self.softmax(target)
        predicted_class = np.argmax(probabilities, axis=1)
        loss_values = self.negativeloglikelihood(probabilities, test_label_set)
        acc = np.mean(predicted_class == test_label_set)
        loss_values /= 100
        print("Average Loss according to Each Epoch: ", loss_values)
        print("Average Test Accuracy according to Each Epoch: ", acc)
        return loss_values, acc


# For Part 1, you will implement a neural network which contains one hidden layer.
# You will change the mentioned parameters (unit number in the hidden layer, activations function etc.) and report the results. Obligatory
print("Effect of differences of number of hidden layers")
print()
model1 = NeuralNetwork(x_train, y_train, n_hidden_layers=1)
print("Training of model 1 begins")
model1.train()
print("Training of model 1 is over")
print("Average Loss and Average Test Accuracy values of model 1")
loss1, acc1 = model1.test(x_val, y_val)
print(round(loss1, 5))
###########################################################################
model2 = NeuralNetwork(x_train, y_train, n_hidden_layers=2)
print("Training of model 2 begins")
model2.train()
print("Training of model 2 is over")
print("Average Loss and Average Test Accuracy values of model 2")
loss2, acc2 = model2.test(x_val, y_val)
print()
print()
table = [['models', 'model1', 'model2'],
         ['Number of Hidden Layer', '1', '2'],
         ['Average Accuracy Value', acc1, acc2],
         ['Average Loss Value', loss1, loss2]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print()
# For Part 1, you will implement single layer neural network and run experiments on
# Vegetable Image Dataset. You will change parameters (activation func., objective
# func. etc.) and report results with a table format in your reports. Obligatory

# Effect of differences of size of hidden layers
print("Effect of differences of size of hidden layers")
print()
model3 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, size_hidden_layer=64)
print("Training of model 3 begins")
model3.train()
print("Training of model 3 is over")
print("Average Loss and Average Test Accuracy values of model 3")
loss3, acc3 = model3.test(x_val, y_val)
###########################################################################
model4 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, size_hidden_layer=128)
print("Training of model 4 begins")
model4.train()
print("Training of model 4 is over")
print("Average Loss and Average Test Accuracy values of model 4")
loss4, acc4 = model4.test(x_val, y_val)
print()
print()
table = [['models', 'model1', 'model2'],
         ['Size of Hidden Layer', '64', '128'],
         ['Average Accuracy Value', acc3, acc4],
         ['Average Loss Value', loss3, loss4]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print()

# Effect of differences of batch size
print("Effect of differences of batch size")
print()
model5 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, batch_size=16)
print("Training of model 5 begins")
model5.train()
print("Training of model 5 is over")
print("Average Loss and Average Test Accuracy values of model 5")
loss5, acc5 = model5.test(x_val, y_val)

model6 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, batch_size=32)
print("Training of model 6 begins")
model6.train()
print("Training of model 6 is over")
print("Average Loss and Average Test Accuracy values of model 6")
loss6, acc6 = model6.test(x_val, y_val)

model7 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, batch_size=64)
print("Training of model 7 begins")
model7.train()
print("Training of model 7 is over")
print("Average Loss and Average Test Accuracy values of model 7")
loss7, acc7 = model7.test(x_val, y_val)
print()
print()
table = [['models', 'model1', 'model2', 'model3'],
         ['Batch Size', '16', '32', '64'],
         ['Average Accuracy Value', acc5, acc6, acc7],
         ['Average Loss Value', loss5, loss6, loss7]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print()
# Effect of differences of activation function
print("Effect of differences of activation function")
print()
model9 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, activation="relu")
print("Training of model 9 begins")
model9.train()
print("Training of model 9 is over")
print("Average Loss and Average Test Accuracy values of model 9")
loss9, acc9 = model9.test(x_val, y_val)

model10 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, activation="tanh")
print("Training of model 10 begins")
model10.train()
print("Training of model 10 is over")
print("Average Loss and Average Test Accuracy values of model 10")
loss10, acc10 = model10.test(x_val, y_val)

model11 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, activation="sigmoid")
print("Training of model 11 begins")
model11.train()
print("Training of model 11 is over")
print("Average Loss and Average Test Accuracy values of model 11")
loss11, acc11 = model11.test(x_val, y_val)
print()
print()
table = [['models', 'model1', 'model2', 'model3'],
         ['Activation Function', 'ReLU', 'TanH', 'Sigmoid'],
         ['Average Accuracy Value', acc9, acc10, acc11],
         ['Average Loss Value', loss9, loss10, loss11]]
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
print()

# According to best consequences, the final best model is
print("According to best consequences, the final best model is")
print()
print()
model12 = NeuralNetwork(x_train, y_train, n_hidden_layers=2, batch_size=64, size_hidden_layer=128, activation="relu")
print("Training of model 12 begins")
model12.train()
print("Training of model 12 is over")
print("Average Loss and Average Test Accuracy values of model 12")
model12.test(x_val, y_val)
