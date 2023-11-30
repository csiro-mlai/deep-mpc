.. _ml-quickstart:

Machine Learning Quickstart
---------------------------

This document is a short introduction to running privacy-preserving
logistic regression in MP-SPDZ. It assumes that you have the framework
already installed as explained in the `installation instructions
<https://mp-spdz.readthedocs.io/en/latest/readme.html#tl-dr-binary-distribution-on-linux-or-source-distribution-on-macos>`_.
For more information on how to run machine learning algorithms in MP-SPDZ,
see the `full machine learning section
<https://mp-spdz.readthedocs.io/en/latest/machine-learning.html>`_.

The easiest way to use is to put Python code in an ``.mpc`` in
``Programs/Source``, for example ``Programs/Source/foo.mpc``. Put the
following code there to use the breast cancer dataset::

  X = sfix.input_tensor_via(0, [[1, 2, 3], # 2 samples
                                [11, 12, 13]])
  y = sint.input_tensor_via(0, [0, 1]) # 2 labels

  from Compiler import ml
  log = ml.SGDLogistic(100)
  log.fit(X, y)

  print_ln('%s', log.predict(X).reveal())

The first two lines make the data available to the secure
computation. The next lines create a logistic regression instance and
train it (for one hundred epochs). Finally, the last line uses the
instances for predictions and outputs the results.

After adding all the above code to ``Programs/Source/foo.mpc``, you
can run it either insecurely:

.. code-block:: console

  Scripts/compile-emulate.py foo

or securely with three parties on the same machine:

.. code-block:: console

  Scripts/compile-run.py -E ring foo

The first call should give the following output:

.. code-block:: console

  $ Scripts/compile-emulate.py foo
  Default bit length: 63
  Default security parameter: 40
  Compiling file Programs/Source/foo.mpc
  Writing binary data to Player-Data/Input-Binary-P0-0
  Setting learning rate to 0.01
  Using SGD
  Initializing dense weights in [-1.224745,1.224745]
  Writing to Programs/Bytecode/foo-multithread-1.bc
  2 runs per epoch
  Writing to Programs/Bytecode/foo-multithread-3.bc
  Writing to Programs/Bytecode/foo-multithread-4.bc
  Writing to Programs/Bytecode/foo-multithread-5.bc
  Initializing dense weights in [-1.224745,1.224745]
  Writing to Programs/Bytecode/foo-multithread-7.bc
  Writing to Programs/Bytecode/foo-multithread-8.bc
  Writing to Programs/Bytecode/foo-multithread-9.bc
  Writing to Programs/Schedules/foo.sch
  Writing to Programs/Bytecode/foo-0.bc
  Hash: 33f8d22d99960897f41fb2da31e7f5a0501d2e1071789e52d73b4043e5343831
  Program requires at most:
             8 integer inputs from player 0
         61054 integer bits
        190109 integer triples
           200 matrix multiplications (1x3 * 3x1)
           200 matrix multiplications (3x1 * 1x1)
             1 matrix multiplications (2x3 * 3x1)
         28406 virtual machine rounds
  Using security parameter 40
  Trying to run 64-bit computation
  Using SGD
  done with epoch 99
  [0, 1]
  The following benchmarks are including preprocessing (offline phase).
  Time = 0.0250086 seconds 

See `the documentation
<https://mp-spdz.readthedocs.io/en/latest/readme.html#running-computation>`_
for further
options such as different protocols or running remotely and `the
machine learning section
<https://mp-spdz.readthedocs.io/en/latest/machine-learning.html>`_ for
other machine learning methods.
