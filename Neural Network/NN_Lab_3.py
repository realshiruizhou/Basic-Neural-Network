import math
import sys
import random
t_func = {"T1": lambda x: x, "T2": lambda x: x if x > 0 else 0, "T3": lambda x: 1/(1+math.exp(-x)), "T4": lambda x: -1+2/(1+math.exp(-x))}
dervs = {"T1": lambda y: 1, "T2": lambda y: 1 if y > 0 else 0, "T3": lambda y: y*(1-y), "T4": lambda y: (1-y*y)/2}
weights = [[-0.5182376095658087, -1.9913405023502904, -0.3743290179070571, 0.21981307228739977, 0.7795213557897814, -3.459645838814721],
[-3.153536558025982, -5.0379076944479895],
[2.729804264196657]]

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


inputs = [1, 1, 1]
print(feed_forward(weights, inputs))
