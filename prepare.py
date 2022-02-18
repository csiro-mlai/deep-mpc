#!/usr/bin/python3

out = open('/tmp/cifar10-Input-P0-0', 'w')

def f(x):
    for a in x:
        aa = (a / 255 * 2 - 1)
        line = ''
        for i in range(1024):
            for j in range(3):
                line += '%.6f ' % aa[i + 1024 * j]
        line.strip()
        out.write(line)
        out.write('\n')

def g(x):
    for a in x:
        for i in range(10):
            out.write(str(int(i == a)))
            out.write(' ')
    out.write('\n')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

labels = []
data = []

for i in range(1, 6):
    part = unpickle('cifar-10-batches-py/data_batch_%d' % i)
    labels.extend(part[b'labels'])
    data.append(part[b'data'])

g(labels)

for x in data:
    f(x)

data = unpickle('cifar-10-batches-py/test_batch')
g(data[b'labels'])
f(data[b'data'])
