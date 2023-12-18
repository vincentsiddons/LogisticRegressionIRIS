import os
import numpy as np
import functools
import operator
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

#This class impliments a Logistic Regression model for more than 2 classes.
#NOTE: csv class label must correspond to the next class label unless all instances of that class are exhausted
class LogisticRegressionModel:
    csv = ''
    training = 0.0
    dev = 0.0
    test = 0.0

    curr_directory = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, csv = os.path.join(curr_directory, 'IRIS.csv'), training = 0.8, dev = 0.0, test = 0.2):
        self.csv = csv
        self.training = training
        self.dev = dev
        self.test = test
    
    #no arguement constructor
    @classmethod
    def no_arg(cls):
        return cls()

    def __str__(self):
        return "The file you are processing is: " + self.csv + " and you train/dev/test split is: " + str(self.training) + "/" + str(self.dev) + "/" + str(self.test) + "\n"
    
    #Deletes incomplete features at end of array
    def delete_hanging_numbers(self, arr):
        indicies = []
        i = len(arr) - 1
        while i >= 0:
            if arr[i][0:1].isnumeric() == True:
                indicies.append(i)
            elif arr[i][0:1].isnumeric() == False:
                break
            i -= 1
        try:
            arr = arr[0:indicies[len(indicies) - 1]]
        except:
            return arr
        return arr
    
    #Deletes incomplete features at start of array
    def delete_starting_numbers(self, arr):
        indicies = []
        i = 0
        while i < len(arr):
            if arr[i][0:1].isnumeric() == True:
                indicies.append(i)
            elif arr[i][0:1].isnumeric() == False:
                break
            i += 1
        try:
            arr = arr[indicies[len(indicies) - 1] + 2:len(arr)]
        except:
            return arr
        return arr

    
    #Stores the index of the start of each class in the csv arr in an arr
    def find_class_indicies(self):
        csv = open(self.csv, 'r')
        lines_arr = csv.readlines()

        #split wherever there is a comma and turn resulting 2D array into a 1D one
        csv_arr = []
        for i in range(0, len(lines_arr)):
            csv_arr.append(lines_arr[i].split(','))
        csv_arr = functools.reduce(operator.iconcat, csv_arr, [])

        curr_class = csv_arr[0]
        class_indicies = []
        for i in range(0, len(csv_arr)):
            if curr_class != csv_arr[i] and (i + 1) != 5 and (i + 1) % 5 == 0:
                curr_class = csv_arr[i]
                class_indicies.append(i)
                i += 1
        class_indicies.append(len(csv_arr) - 1)
        csv.close()

        return lines_arr, class_indicies
    
    #adjusts values of weights via stochastic gradient descent
    #new weights are created element-wise, L2 regularization
    def gradient_descent_w(self, x, arr, hyp, y, alpha):
        grad = np.zeros((len(arr), len(x)))
        for j in range(0, len(grad)):
            arr = arr - np.max(arr)
            diff = (-hyp*((np.exp(arr[j])/sum(np.exp(arr)))) - y[j])
            diff = diff + alpha*(diff**2)
            for i in range(0, len(grad[j])):
                grad[j][i] = diff[j]*x[i]
        return grad
    
    #adjusts values of biases via stochastic gradient descent, L2 regularization
    def gradient_descent_b(self, x, arr, hyp, y, alpha):
        grad = np.zeros((len(arr), 1))
        for j in range(0, len(grad)):
            arr = arr - np.max(arr)
            diff = (-hyp*((np.exp(arr[j])/sum(np.exp(arr)))) - y[j])
            diff = diff + alpha*(diff**2)
            for i in range(0, len(grad[j])):
                grad[j][i] = diff[j]*x[i]
        return grad
    
    #Returns softmax outputs of each element in an array
    def softmax(self, arr):
        softmax_arr = np.zeros(len(arr))
        for i in range(0, len(arr)):
            softmax_arr[i] = (np.exp(arr[i]))/(np.sum(np.exp(arr)))
        return softmax_arr
    
    #Calculates the F1 score for each class
    def calcualte_F1(self, x, w, b, y):
        outputs = []
        confusion_matrix = np.zeros((len(y[0]), len(y[0])))
        for i in range(0, len(x)):
            outputs.append(self.softmax(np.matmul(w, x[i]) + b))
        outputs = np.array(outputs)

        for i in range(0, len(outputs)):
            classif = np.argmax(outputs[i])
            gold_classif = np.argmax(y[i])
            confusion_matrix[classif][gold_classif] += 1

        F1 = []
        precisions = []
        recalls = []

        precisions.append(np.diagonal(confusion_matrix)/np.sum(confusion_matrix, axis=1))
        recalls.append(np.diagonal(confusion_matrix)/np.sum(confusion_matrix, axis=0))

        for i in range(0, len(precisions)):
            F1.append((2*precisions[i]*recalls[i])/(precisions[i]+ recalls[i]))
        F1 = np.array(F1)
        return F1
    
    #Scales x and performs one-hot encoding on y
    def x_y_processing(self, arr, class_arr):
        #creates input array of 4 features and output for each class
        x = []
        y = []
        i = 0
        while i < len(arr) - 1:
            x.append(arr[i: i + 4].astype(float))
            y.append([arr[i + 4][:len(arr[i + 4]) - 1]])
            i += 5

        x = np.array(x)
        y = np.array(y)
        

        #One-hot encoding
        y_embedded = np.zeros((len(y) - 1, class_arr))
        class_indicies = [0]
        class_num = 0
        for j in range(0, len(y) - 1):
            for i in range(0, len(y[j])):
                y_embedded[j][class_num] = 1
                #even out the number of examples for each class
                if (y[j][i] != y[j + 1][i]):
                    class_indicies.append(j - class_indicies[len(class_indicies) - 1])
                    class_indicies.append(j)
                    class_num += 1
                    j += 1

        #standard scale x
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        
        # Under-sampling (too many instances of class 1)
        under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        x, y_embedded = under_sampler.fit_resample(x, y_embedded)

        y = y_embedded

        return x, y

    #Creates array of each training feature value, initalizes weights and bias, creates output array, and also returns dev and test arrays
    def preprocessing(self):
        
        lines = self.find_class_indicies()
        indicies = lines[1]
        num_classes = len(indicies) - 1

        #Splits each line of the csv into features and outputs
        lines = lines[0]
        for i in range(0, len(lines)):
            lines[i] = lines[i].split(',')
        lines = functools.reduce(operator.iconcat, lines, [])

        #fills each array with their respective percentages of each class
        training_arr = []
        dev_arr = []
        test_arr = []
        for j in range(0, len(indicies) - 1):
            if j == 0:
                training_arr = lines[indicies[0] + 1:int(self.training*indicies[j + 1])]
                training_arr = self.delete_hanging_numbers(training_arr)
                dev_arr = lines[int(self.training*indicies[j + 1]) + 1:int(self.dev*indicies[j + 1] + int(self.training*indicies[j+1]))]
                dev_arr = self.delete_hanging_numbers(dev_arr)
                test_arr = lines[int(self.dev*indicies[j + 1] + int(self.training*indicies[j+1])) + 1:int(self.test*indicies[j+1] + int(self.training*indicies[j+1] + int(self.dev*indicies[j+1])))]
                test_arr = self.delete_starting_numbers(test_arr)
                test_arr = self.delete_hanging_numbers(test_arr)
            else:
                training_arr = training_arr + lines[indicies[j] + 1:int(self.training*indicies[j + 1])]
                training_arr = self.delete_hanging_numbers(training_arr)
                dev_arr = dev_arr + lines[int(self.training*indicies[j+1]) + 1:int(self.dev*indicies[j + 1] + int(self.training*indicies[j+1]))]
                dev_arr = self.delete_hanging_numbers(dev_arr)
                #The bug happens here
                needed_lines = lines[int(self.training*indicies[j+1] + int(self.dev*indicies[j+1])) + 1:int(self.test*indicies[j+1] + int(self.training*indicies[j+1] + int(self.dev*indicies[j+1])))]
                needed_lines = self.delete_starting_numbers(needed_lines)
                needed_lines = self.delete_hanging_numbers(needed_lines)
                #Sometimes the test array begins with a class
                if needed_lines[0].isnumeric() == False:
                    del needed_lines[0]
                #Sometimes the test array has <4 features at the start
                for i in range(0, 4):
                    if needed_lines[i][0:1].isnumeric() == False:
                        del needed_lines[0:i + 1]
                test_arr = test_arr + needed_lines
                test_arr = self.delete_starting_numbers(test_arr)
                test_arr = self.delete_hanging_numbers(test_arr)

        training_arr = np.array(training_arr)
        dev_arr = np.array(dev_arr)
        test_arr = np.array(test_arr)

        x_y = self.x_y_processing(training_arr, num_classes)
        x = x_y[0]
        y = x_y[1]

        rng = np.random.default_rng(seed=42)

        o = (rng.random((num_classes, 4)),  rng.random(num_classes))

        return x, o, y, dev_arr, test_arr, num_classes

    #Creates weight and bias values via logistic regression
    def train(self, hyperperameter, alpha):
        preprocessed = self.preprocessing()
        x = preprocessed[0]
        w = preprocessed[1][0]
        b = preprocessed[1][1][:, np.newaxis]
        y = preprocessed[2]
        for j in range(0, len(x)):
            for i in range(0, len(x[j])):
                output = np.matmul(w, x[j])
                w -= self.gradient_descent_w(x[j], output , hyperperameter, y, alpha)
                b -= self.gradient_descent_b(x[j], b, hyperperameter, y, alpha)
        b = b.flatten()
        return w,b
    
    #Returns an array of the F1 score for each class in the testing set
    def testing(self, hyperperameter, alpha):
        data = self.preprocessing()
        test_arr = data[4]
        num_classes = data[5]
        w_b = self.train(hyperperameter, alpha)
        x_y = self.x_y_processing(test_arr, num_classes)
        x = x_y[0]
        y = x_y[1]  

        if len(x) > len(y):
            x = x[0:len(y)]

        return self.calcualte_F1(x, w_b[0], w_b[1], y)

model1 = LogisticRegressionModel.no_arg()

print(model1.testing(0.2, 0.9))
        