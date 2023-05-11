#!/usr/bin/python3

import sys
import numpy

binary = 'binary' in sys.argv

bout = open('/tmp/cifar10-Binary-P0-0', 'wb')
out = open('/tmp/cifar10-Input-P0-0', 'w')

if binary:
    def ff(x):
        bout.write(x.astype(numpy.single).tobytes())
else:
    def ff(x):
        numpy.savetxt(out, x.reshape(x.shape[0], -1), '%.6f')

def f(x):
    x = numpy.reshape(x, (x.shape[0], 3, 32, 32))
    x = numpy.moveaxis(x, 1, -1)
    print(x.shape)
    x = (x / 255 * 2 - 1)
    ff(x)
    print (x.max(), x.min(), x.sum())

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
