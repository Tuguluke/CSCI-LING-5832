import csv
import regex as re
import numpy as np
import random

# part1 data cleaning functions
with open('positive-words.txt', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
poswords_list = []
for i in data:
    poswords_list.append(i[0])


with open('negative-words.txt', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
negwords_list = []
for i in data:
    negwords_list.append(i[0])

with open('hotelPosT-train.txt', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    data = list(reader)
hotel_pos_list = []
for i in data:
    hotel_pos_list.append(i)

with open('hotelNegT-train.txt', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    data = list(reader)
hotel_neg_list = []
for i in data:
    hotel_neg_list.append(i)


def counting_pos_rows(a):
    counter = 0
    for i in a.split():
        if re.sub(r'[!,.]', '', i) in poswords_list:
            counter += 1
    return counter


def counting_neg_rows(a):
    counter = 0
    for i in a.split():
        if re.sub(r'[!,.]', '', i) in negwords_list:
            counter += 1
    return counter


def counting_no_rows(a):
    counter = 0
    for i in a.split():
        if i in ['no']:
            counter = 1
    return counter


def counting_pronouns_rows(a):
    counter = 0
    for i in a.split():
        if i in ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]:
            counter += 1
    return counter


def counting_exclam_rows(a):
    counter = 0
    for i in a.split():
        if i[-1] == '!':
            counter = 1
    return counter


def counting_loglen_rows(a):
    i = a.split()
    counter = np.round(np.log(len(i)), 2)
    return counter


def extract_features(a):
    return [a[0].replace('\'', ''), counting_pos_rows(a[1].lower()),
            counting_neg_rows(a[1].lower()), counting_no_rows(a[1].lower()),
            counting_pronouns_rows(
                a[1].lower()), counting_exclam_rows(a[1].lower()),
            counting_loglen_rows(a[1].lower())]


# part2 SGD
with open('tuguluke-abulitibu-assgn2-part1.csv', newline='') as f:
    reader = csv.reader(f)
    train_data_all = list(reader)

def split_train_set(train_data_all):
    train_matrix = []
    for data in train_data_all:
        train_matrix.append([float(i)
                            for i in data[0][8:].replace(',', " ").split()])

    random.shuffle(train_matrix)
    train_data = train_matrix[:int((len(train_matrix)+1)*.80)]
    test_data = train_matrix[int((len(train_matrix)+1)*.80):]
    return train_data, test_data


train_data, test_data = split_train_set(train_data_all)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x, y = np.array([i[:-1] for i in train_data]), np.array([i[-1]
                                                         for i in train_data])


def stochastic_gradient_des(learn_rate=0.1, iter_max=50000):
    """
    sigmoid is a function parameterized by theta
    x is set of trains x_1, ..., x_m
    y is set of training output (labels)
    """
    theta = np.array([0]*6)   # theta <- 0

    x, y = np.array([i[:-1] for i in train_data]), np.array([i[-1]
                                                             for i in train_data])

    for i in range(iter_max):
        weights = np.array([0]*6)
        # in random order
        random_train = random.choice(test_data)
        x_random, y_random = np.array(
            random_train[:-1]), np.array(random_train[-1])
        # what is the estimate output for y_hat
        y_hat = sigmoid(np.dot(weights, x_random))
        # theta = theta - eta*g
        theta = theta - learn_rate * \
            ((learn_rate*((y_hat - y_random)*x_random))/len(train_data))
       # TODO: bias seems not contributing to the result improvement.
    return theta


w = stochastic_gradient_des()

with open('HW2-testset.txt', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    data = list(reader)
testset_list = []
for i in data:
    testset_list.append(i)

hw_list = []
for i in testset_list:
        if sigmoid(np.dot(w,extract_features(i)[1:])) > .5:
            hw_list.append(i[0]+' '+ 'POS')
        elif sigmoid(np.dot(w,extract_features(i)[1:])) < .5:
            hw_list.append(i[0]+' '+ 'NEG')


textfile = open("tuguluke-abulitibu-assgn2-out.txt", "w")
for element in hw_list:
    textfile.write(element + "\n")
textfile.close()


x_test, y_test = np.array([i[:-1] for i in test_data]
                          ), np.array([i[-1] for i in test_data])
counter_acc = 0
for (i, j) in zip(x_test, y_test):
    new_weight = sigmoid(np.dot(w, i)) #TODO:
    if new_weight > .5 and j == 1:
        counter_acc += 1
    elif new_weight < 5 and j == 0:
        counter_acc += 1

print("acc:", counter_acc/len(x_test))
