Machine Learning
----------------

The purpose of this document is to demonstrate the machine learning
functionality of MP-SPDZ, a software implementing multi-party
computation, one of the most important privacy-enhancing
techniques. Please see `this gentle introduction
<https://eprint.iacr.org/2020/300>`_ for more information on
multi-party computation and the `installation instructions
<readme.html#tl-dr-binary-distribution-on-linux-or-source-distribution-on-macos>`_
on how to install the software.

MP-SPDZ supports a number of machine learning algorithms such as
logistic and linear regression, decision trees, and some common deep
learning functionality. The latter includes the SGD and Adam
optimizers and the following layer types: dense, 2D convolution, 2D
max-pooling, and dropout.

The machine learning code only works in with arithmetic machines, that
is, you cannot compile it with ``-B``.

This document explains how to input data, how to train a model, and
how to use an existing model for prediction.


Data Input
~~~~~~~~~~

It's easiest to input data if it's available during compilation,
either centrally or per party. Another way is to only define the data
size in the high-level code and put the data independently into the
right files used by the virtual machine.


Integrated Data Input
=====================

If the data is available during compilation, for example as a PyTorch
or numpy tensor, you can use
:py:func:`Compiler.types.sfix.input_tensor_via` and
:py:func:`Compiler.types.sint.input_tensor_via`. Consider the
following code from ``breast_logistic.mpc`` (requiring
`scikit-learn <https://scikit-learn.org>`_)::

  from sklearn.datasets import load_breast_cancer
  from sklearn.model_selection import train_test_split

  X, y = load_breast_cancer(return_X_y=True)

  # normalize column-wise
  X /= X.max(axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

  X_train = sfix.input_tensor_via(0, X_train)
  y_train = sint.input_tensor_via(0, y_train)

This downloads the Wisconsin Breast Cancer dataset, normalizes the
sample data, splits it into a training and a test set, and then
converts it to an the relevant MP-SPDZ data structures. Under the
hood, the data is stored in ``Player-Data/Input-Binary-P0-0``, which
is where binary-encoded inputs for player 0 are read from. You
therefore have to copy said file if you execute it in another place
than where you compiled it.

MP-SPDZ also allows splitting the data input between parties, for
example horizontally::

  a = sfix.input_tensor_via(0, X_train[len(X_train) // 2:])
  b = sfix.input_tensor_via(1, X_train[:len(X_train) // 2])
  X_train = a.concat(b)

  a = sint.input_tensor_via(0, y_train[len(y_train) // 2:])
  b = sint.input_tensor_via(1, y_train[:len(y_train) // 2])
  y_train = a.concat(b)

The concatenation creates a unified secret tensor that can be used for
training over the whole dataset. Similarly, you can split a dataset
vertically::

  a = sfix.input_tensor_via(0, X_train[:,:X_train.shape[1] // 2])
  b = sfix.input_tensor_via(1, X_train[:,X_train.shape[1] // 2:])
  X_train = a.concat_columns(b)

The three approaches in this section can be run as follows::

  Scripts/compile-run.py -E ring breast_logistic
  Scripts/compile-run.py -E ring breast_logistic horizontal
  Scripts/compile-run.py -E ring breast_logistic vertical

In the last variants, the labels are all input via party 0.

Finally, MP-SPDZ also facilitates inputting data that is also
available party by party. Party 0 can run::

  a = sfix.input_tensor_via(0, X_train[:,:X_train.shape[1] // 2])
  b = sfix.input_tensor_via(1, shape=X_train[:,X_train.shape[1] // 2:].shape)
  X_train = a.concat_columns(b)
  y_train = sint.input_tensor_via(0, y_train)

while party 1 runs::

  a = sfix.input_tensor_via(0, shape=X_train[:,:X_train.shape[1] // 2].shape)
  b = sfix.input_tensor_via(1, X_train[:,X_train.shape[1] // 2:])
  X_train = a.concat_columns(b)
  y_train = sint.input_tensor_via(0, shape=y_train.shape)

Note that that the respective party only accesses the shape of data
they don't input.

You can run this case by running on one hand:

.. code-block:: console

  ./compile.py breast_logistic party0
  ./semi-party.x 0 breast_logistic-party0

and on the other (but on the same host):

.. code-block:: console

  ./compile.py breast_logistic party1
  ./semi-party.x 1 breast_logistic-party1

The compilation will output a hash at the end, which has to agree
between the parties. Otherwise the virtual machine will abort with an
error message. To run the two parties on different hosts, use the
:ref:`networking options <networking>`.


Data preprocessing
""""""""""""""""""

Sometimes it's necessary to preprocess data. We're using the following
code from ``torch_mnist_dense.mpc`` to demonstrate this::

  ds = torchvision.datasets.MNIST(root='/tmp', train=train, download=True)
  # normalize to [0,1] before input
  samples = sfix.input_tensor_via(0, ds.data / 255)
  labels = sint.input_tensor_via(0, ds.targets, one_hot=True)

This downloads the default training or the test set of MNIST
(depending on :py:obj:`train`) and then processes it to make it
usable. The sample data is normalized from an 8-bit integer to the
interval :math:`[0,1]` by dividing by 255. This is done within PyTorch
for efficiency. Then, the labels are encoded as one-hot vectors
because this is necessary for multi-label training in MP-SPDZ.


Independent Data Input
======================

The example code in
``keras_mnist_dense.mpc`` trains a dense neural network for
MNIST. It starts by defining tensors to hold data::

  training_samples = sfix.Tensor([60000, 28, 28])
  training_labels = sint.Tensor([60000, 10])

  test_samples = sfix.Tensor([10000, 28, 28])
  test_labels = sint.Tensor([10000, 10])

The tensors are then filled with inputs from party 0 in the order that
is used by ``convert.sh`` in `the preparation code
<https://github.com/csiro-mlai/deep-mpc>`_::

  training_labels.input_from(0)
  training_samples.input_from(0)

  test_labels.input_from(0)
  test_samples.input_from(0)

The virtual machine then expect the data as whitespace-separated text
in ``Player-Data/Input-P0-0``. If you use ``binary=True`` with
:py:func:`input_from`, the input is expected in
``Player-Data/Input-Binary-P0-0``, value by value as single-precision
float or 64-bit integer in the machine byte order (most likely
little-endian these days).


Training
~~~~~~~~

There are a number of interfaces for different algorithms.


Logistic regression with SGD
============================

This is available via :py:class:`~Compiler.ml.SGDLogistic`. We will
use ``breast_logistic.mpc`` as an example.

After inputting the data as above, you can call the following::

  log = ml.SGDLogistic(20, 2, program)
  log.fit(X_train, y_train)

This trains a logistic regression model in secret for 20 epochs with
mini-batches of size 2. Adding the :py:obj:`program` object as a
parameter uses further command-line parameters. Most notably, you can
add ``approx`` to use a three-piece approximate sigmoid function:

.. code-block:: console

  Scripts/compile-emulate.py breast_logistic approx

Omitting it invokes the default sigmoid function.

To check accuracy during training, you can call the following instead
of :py:func:`~Compiler.ml.SGDLogistic.fit`::

  log.fit_with_testing(X_train, y_train, X_test, y_test)

This outputs losses and accuracy for both the training and test set
after every epoch.

You can use :py:func:`~Compiler.ml.SGDLogistic.predict` to predict
labels and :py:func:`~Compiler.ml.SGDLogistic.predict_proba` to
predict probabilities. The following outputs the correctness (0 for
correct, :math:`\pm 1` for incorrect) and a measure of how much off
the probability estimate is::

  print_ln('%s', (log.predict(X_test) - y_test.get_vector()).reveal())
  print_ln('%s', (log.predict_proba(X_test) - y_test.get_vector()).reveal())


Linear regression with SGD
==========================

This is available via :py:class:`~Compiler.ml.SGDLinear`. It
implements an interface similar to logistic regression. The main
difference is that there is only
:py:func:`~Compiler.ml.SGDLinear.predict` for prediction as there is
no notion of labels in this case. See ``diabetes.mpc`` for an example
of linear regression.


PyTorch interface
=================

MP-SPDZ supports importing sequential models from PyTorch using
:py:func:`~Compiler.ml.layers_from_torch` as shown in
this code snippet in ``torch_mnist_dense.mpc``::

  import torch.nn as nn

  net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
  )

  from Compiler import ml

  ml.set_n_threads(int(program.args[2]))

  layers = ml.layers_from_torch(net, training_samples.shape, 128)

  optimizer = ml.SGD(layers)
  optimizer.fit(
    training_samples,
    training_labels,
    epochs=int(program.args[1]),
    batch_size=128,
    validation_data=(test_samples, test_labels),
    program=program
  )

This trains a network with three dense layers on MNIST using SGD,
softmax, and cross-entropy loss. The number of epochs and threads is
taken from the command line. For example, the following trains the
network for 10 epochs using 4 threads::

  Scripts/compile-emulate.py torch_mnist_dense 10 4

See ``Programs/Source/torch_*.mpc`` for further examples of the
PyTorch functionality, :py:func:`~Compiler.ml.Optimizer.fit` for
further training options, and :py:class:`~Compiler.ml.Adam` for an
alternative Optimizer.


Keras interface
===============

The following Keras-like code sets up a model with three dense layers
and then trains it::

  from Compiler import ml
  tf = ml

  layers = [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax')
  ]

  model = tf.keras.models.Sequential(layers)

  optim = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.01)

  model.compile(optimizer=optim)

  opt = model.fit(
    training_samples,
    training_labels,
    epochs=1,
    batch_size=128,
    validation_data=(test_samples, test_labels)
  )

See ``Programs/Source/keras_*.mpc`` for further examples using the
Keras interface.


Decision trees
==============

MP-SPDZ can train decision trees for binary labels by using the
algorithm by `Hamada et al.`_ The following example in
``breast_tree.mpc`` trains a tree of height five before outputting the
difference between the prediction on a test set and the ground truth::

  from Compiler.decision_tree import TreeClassifier
  tree = TreeClassifier(max_depth=5)
  tree.fit(X_train, y_train)
  print_ln('%s', (tree.predict(X_test) - y_test.get_vector()).reveal())

You can run the example as follows:

.. code-block:: console

  Scripts/compile-emulate.py breast_tree

It is also possible to output the accuracy after every level::

  tree.fit_with_testing(X_train, y_train, X_test, y_test)

You can output the trained tree as follows::

  tree.output()

The format of the output follows the description of `Hamada et al.`_

MP-SPDZ by default uses probabilistic rounding for fixed-point
division, which is used to compute Gini coefficients in decision tree
training. This has the effect that the tree isn't deterministic. You
can switch to deterministic rounding as follows::

  sfix.round_nearest = True

The ``breast_tree.mpc`` uses the following code to allow switching on
the command line::

  sfix.set_precision_from_args(program)

Nearest rounding can then be activated as follows:

.. code-block:: console

  Scripts/compile-emulate.py breast_tree nearest

.. _`Hamada et al.`: https://arxiv.org/abs/2112.12906


Data preparation
""""""""""""""""

MP-SPDZ currently support continuous and binary attributes but not
discrete non-binary attributes. However, such attributes can be
converted as follows using the `pandas <https://pandas.pydata.org>`_
library::

  import pandas
  from sklearn.model_selection import train_test_split
  from Compiler import decision_tree

  data = pandas.read_csv(
    'https://datahub.io/machine-learning/adult/r/adult.csv')

  data, attr_types = decision_tree.preprocess_pandas(data)

  # label is last column
  X = data[:,:-1]
  y = data[:,-1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

This downloads the adult dataset and convert discrete attributes to
binary using one-hot encoding. See ``easy_adult`` for the full
example. :py:obj:`attr_types` has to be used to indicates the
attribute types during training::

  tree.fit(X_train, y_train, attr_types=attr_types)


Loading pre-trained models
~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to import pre-trained from PyTorch as shown in
``torch_mnist_lenet_predict.mpc``::

  net = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20, 50, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(800, 500),
    nn.ReLU(),
    nn.Linear(500, 10)
  )

  # train for a bit
  transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])
  ds = torchvision.datasets.MNIST(root='/tmp', transform=transform, train=True)
  optimizer = torch.optim.Adam(net.parameters(), amsgrad=True)
  criterion = nn.CrossEntropyLoss()

  for i, data in enumerate(torch.utils.data.DataLoader(ds, batch_size=128)):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

This trains LeNet on MNIST for one epoch. The model can then be input
and used in MP-SPDZ::

  from Compiler import ml
  layers = ml.layers_from_torch(net, training_samples.shape, 128, input_via=0)
  optimizer = ml.Optimizer(layers)
  n_correct, loss = optimizer.reveal_correctness(test_samples, test_labels, 128, running=True)
  print_ln('Secure accuracy: %s/%s', n_correct, len(test_samples))

This outputs the accuracy of the network. You can use
:py:func:`~Compiler.ml.Optimizer.eval` instead of
:py:func:`~Compiler.ml.Optimizer.reveal_correctness` to retrieve
probability distributions or top guessess (the latter with ``top=True``)
for any sample data.


Storing and loading models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the Keras interface and the native
:py:class:`~Compiler.ml.Optimizer` class support an interface to
iterate through all model parameters. The following code from
``torch_mnist_dense.mpc`` uses it to store the model on disk in
secret-shared form::

  for var in optimizer.trainable_variables:
    var.write_to_file()

The example code in ``torch_mnist_dense_predict.mpc`` then uses the
model stored above for prediction. Much of the setup is the same, but
instead of training it reads the model from disk::

  optimizer = ml.Optimizer(layers)

  start = 0
  for var in optimizer.trainable_variables:
    start = var.read_from_file(start)

Then it runs the accuracy test::

  n_correct, loss = optimizer.reveal_correctness(test_samples, test_labels, 128)
  print_ln('Accuracy: %s/%s', n_correct, len(test_samples))

Using ``var.input_from(player)`` instead the model would be input
privately by a party.


Exporting models
~~~~~~~~~~~~~~~~

Models can be exported as follows::

  optimizer.reveal_model_to_binary()

if :py:obj:`optimizer` is an instance of
:py:class:`Compiler.ml.Optimizer`. The model parameters are then
stored in ``Player-Data/Binary-Output-P<playerno>-0``. They can be
imported for use in PyTorch::

  f = open('Player-Data/Binary-Output-P0-0')

  state = net.state_dict()

  for name in state:
      shape = state[name].shape
      size = numpy.prod(shape)
      var = numpy.fromfile(f, 'double', count=size)
      var = var.reshape(shape)
      state[name] = torch.Tensor(var)

  net.load_state_dict(state)

if :py:obj:`net` is a PyTorch module with the correct meta-parameters.
This demonstrates that the parameters are stored with double precision
in the canonical order.

There are a number of scripts in ``Scripts``, namely
``torch_cifar_alex_import.py``, ``torch_mnist_dense_import.py``, and
``torch_mnist_lenet_import.py``, which import the models output by
``torch_alex_test.mpc``, ``torch_mnist_dense.mpc``, and
``torch_mnist_lenet_predict.mpc``. For example you can run:

.. code-block:: console

  $ Scripts/compile-emulate.py torch_mnist_lenet_predict
  ...
  Secure accuracy: 9822/10000
  ...
  $ Scripts/torch_mnist_lenet_import.py
  Test accuracy of the network: 98.22 %

The accuracy values might vary as the model is freshly trained, but
they should match.
