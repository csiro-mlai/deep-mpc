.. _lowest-level:

Lowest-Level Interface
----------------------

In the following, we will introduce the most protocol-independent
interfaces by walking through `Utils/protocol-tutorial
<../Utils/protocol-tutorial.cpp>`_. It implements the Rep3
multiplication protocol independently of the usual protocol interface
for illustration purposes.

.. default-domain:: cpp

.. code-block:: cpp

    // set up networking on localhost
    int my_number = atoi(argv[1]);
    int port_base = 9999;
    Names N(my_number, 3, "localhost", port_base);
    CryptoPlayer P(N);

This sets up pairwise encrypted connections as in :ref:`the low-level
example <low-level>`.

.. code-block:: cpp

    // correlated randomness for resharing
    SeededPRNG G[2];

The protocol requires every pair of parties to have a common PRNG, so
we need two instances. We use :class:`SeededPRNG` to make sure to
never use an uninitialized one.

.. code-block:: cpp

    // synchronize with other parties
    octetStream os;
    os.append(G[0].get_seed(), SEED_SIZE);

:class:`octetStream` is generally used to serialize and aggregate
network communication. In this case, we use it to store the seed of
one of the PRNGs.

.. code-block:: cpp

    P.pass_around(os, os, 1);

:func:`Player::pass_around` allows simultaneous sending to the "next" party
and receiving from the "previous" party. We use this with the buffer
holding the seed. As we don't need the send buffer afterwards, we can
use the same buffer for receiving.

.. code-block:: cpp

    G[1].SetSeed(os.consume(SEED_SIZE));

We seed the second PRNG using the received data. :func:`PRNG::SetSeed`
implicitly uses the required number of bits.

.. code-block:: cpp

    // simplify code
    typedef Z2<64> Z;

In this example, we use integers modulo :math:`2^{64}`, but the
protocol also works for any modulus, so we could also use
:class:`gfp_`.

.. code-block:: cpp

    // start with same shares on all parties for simplicity
    // replicated secret sharing of 3
    Z a[2] = {1, 1};
    // and 6
    Z b[2] = {2, 2};

For every secret number in Rep3, every party holds a pair of numbers
in the domain such that every pair of parties has the same number. The
sum of the unique numbers is the secret.

.. code-block:: cpp

    // compute an additive sharing of the product
    Z cc = a[0] * (b[0] + b[1]) + a[1] * b[0];

In a first step, every party computes an additive share of the
product. See `Araki et al. <https://eprint.iacr.org/2016/768>`_ for
details. All domain classes support the standard operators.

.. code-block:: cpp

    // result shares
    Z c[2];

    // re-randomize
    c[0] = cc + G[0].get<Z>() - G[1].get<Z>();

Sending the computed additive secret sharing directly to another party
to get back to a replicative secret sharing would be
insecure. Therefore, we randomize it using random numbers from the two
PRNGs.

.. code-block:: cpp

    // send and receive share
    os.reset_write_head();
    c[0].pack(os);
    P.pass_around(os, os, 1);
    c[1].unpack(os);

We clear the buffer, serialize our share, send it to the "next" party,
and receive one from the "previous" party. This concludes the
multiplication protocol. :func:`Z2::pack` and :func:`Z2::unpack` are
main methods for (de-)serialization. All domain classes support
this. You can use :func:`octetStream::output` to write the buffer to a
C++ output stream.

.. code-block:: cpp

    // open value to party 0
    if (P.my_num() == 1)
    {
        os.reset_write_head();
        c[0].pack(os);
        P.send_to(0, os);
    }

To allow party 0 to output the result, party 1 serializes one of their
shares and sends it to party 0.

.. code-block:: cpp

    // output result on party 0, which should be 18
    if (P.my_num() == 0)
    {
        P.receive_player(1, os);
        cout << "My shares: " << c[0] << ", " << c[1] << endl;
        cout << "Result: " << (os.get<Z>() + c[0] + c[1]) << endl;
    }

Party 0 receives the missing share from party 1 and reconstructs the
secret by summing up.

You can run the example as follows in the main directory:

.. code-block:: sh

    make protocol-tutorial.x
    for i in 0 1 2; do ./protocol-tutorial.x $i & true; done
