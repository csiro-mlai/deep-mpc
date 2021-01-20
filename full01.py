#!/usr/bin/python3

import struct, sys

try:
    digits = [int(x) for x in sys.argv[1]]
except:
    digits = 0, 1

print('digits: %s' % str(digits), file=sys.stderr)

w = lambda x: struct.unpack('>i', x.read(4))[0]
b = lambda x: struct.unpack('B', x.read(1))[0]

for s in 'train', 't10k':
    labels = open('%s-labels-idx1-ubyte' % s, 'rb')
    images = open('%s-images-idx3-ubyte' % s, 'rb')

    assert w(labels) == 2049
    n_labels = w(labels)

    assert w(images) == 2051
    n_images = w(images)
    assert n_labels == n_images
    assert w(images) == 28
    assert w(images) == 28

    print ('%d total examples' % n_images, file=sys.stderr)

    data = []
    n = 0

    for i in range(n_images):
        label = b(labels)
        image = [b(images) / 256 for j in range(28 ** 2)]
        if label in digits:
            data.append(image)
            n += label == digits[1]
            print(int(label == digits[1]), end=' ')
    print()

    print ('%d used examples (%d, %d)' % (len(data), len(data) - n, n),
           file=sys.stderr)

    for x in data:
        for y in x:
            print(y, end=' ')
        print()
