import sys
import math


def x_function(inputs, type):
    if type == "T1":
        return inputs
    elif type == "T2":
        for a in range(0, len(inputs)):
            if inputs[a] <= 0:
                inputs[a] = 0
    elif type == "T3":
        for b in range(0, len(inputs)):
            inputs[b] = 1 / (1 + math.exp(-1 * inputs[b]))
    elif type == "T4":
        for c in range(0, len(inputs)):
            inputs[c] = -1 + 2 / (1 + math.exp(-1 * inputs[c]))


input_values = []
for a in range(3, len(sys.argv)):
    input_values.append(sys.argv[a])
file = open(sys.argv[1], "r").read().splitlines()
for line in range(0, len(file)):
    weights = file[line].split()
    if line != len(file) - 1:
        layers = len(weights) // len(input_values)
        new_input = []
        for b in range(0, layers):
            new_input.append(0)
        count = 0
        input_layer = 0
        for c in range(0, len(weights)):
            new_input[input_layer] += float(input_values[count]) * float(weights[c])
            count += 1
            if count >= len(input_values):
                count = 0
                input_layer += 1
        x_function(new_input, sys.argv[2])
        input_values = new_input[:]
        print(new_input)
    else:
        for d in range(0, len(weights)):
            input_values[d] = input_values[d] * float(weights[d])
# python NN_Lab_1.py weights.txt T3 5 2 3 1 4
