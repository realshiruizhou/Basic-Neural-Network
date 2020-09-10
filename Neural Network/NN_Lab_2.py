import math
import sys
import random
t_func = {"T1": lambda x: x, "T2": lambda x: x if x > 0 else 0, "T3": lambda x: 1/(1+math.exp(-x)), "T4": lambda x: -1+2/(1+math.exp(-x))}
dervs = {"T1": lambda y: 1, "T2": lambda y: 1 if y > 0 else 0, "T3": lambda y: y*(1-y), "T4": lambda y: (1-y*y)/2}
weights = [[], [], []]
alpha = .01


def error(output, expected):
    return .5 * ((expected - output) ** 2)


def feed_forward(weight, input):
    layer_1 = []
    layer_2 = []
    output = []
    s = 0
    for a in range(0, len(weight[0])):
        if a % len(input) == 0 and a != 0:
            layer_1.append(t_func["T3"](s))
            s = 0
        s += weight[0][a] * input[a % len(input)]
    layer_1.append(t_func["T3"](s))
    layer_2.append(t_func["T3"](layer_1[0] * weight[1][0] + layer_1[1] * weight[1][1]))
    output.append(weight[2][0] * layer_2[0])
    return [input, layer_1, layer_2, output]


def back_propagation(output, expected):
    first = expected - output[3][0]
    last_weight = [first * output[2][0]]
    layer_2 = [weights[2][0] * first * dervs["T3"](output[2][0])]
    weight_2 = [layer_2[0] * output[1][0], layer_2[0] * output[1][1]]
    layer_1 = [weights[1][0] * layer_2[0] * dervs["T3"](output[1][0]), weights[1][1] * layer_2[0] * dervs["T3"](output[1][1])]
    weight_1 = []
    for a in range(0, len(output[0])):
        weight_1.append(layer_1[0] * output[0][a])
    for b in range(0, len(output[0])):
        weight_1.append(layer_1[1] * output[0][b])
    return [weight_1, weight_2, last_weight]


def update_weights(old_weights, gradient, alpha):
    new_weights = old_weights[:]
    for a in range(0, 3):
        for b in range(0, len(old_weights[a])):
            new_weights[a][b] = old_weights[a][b] + gradient[a][b] * alpha
    return new_weights


file = open(sys.argv[1], "r").read().splitlines()
inputs = []
for a in file:
    b = a.split()[:-2]
    b.append(a.split()[-1])
    temp = []
    for num in b:
        temp.append(int(num))
    inputs.append(temp)
length = len(inputs[0])
for a in range(0, length * 2):
    weights[0].append(random.uniform(-2, 2))
weights[1].append(random.uniform(-2, 2))
weights[1].append(random.uniform(-2, 2))
weights[2].append(random.uniform(-2, 2))
e = 1.0
while e > .001:
    s = 0.0
    for a in inputs:
        output = feed_forward(weights, a)
        gradient = back_propagation(output, a[-1])
        s += .5 * ((a[-1] - output[-1][0]) ** 2)
        weights = update_weights(weights, gradient, .1)
    e = s/len(inputs)
print("Layer counts: [" + str(len(inputs[0])) + ", 2, 1, 1]")
for a in weights:
    print(a)
# python NN_Lab_2.py sample_input.txt
