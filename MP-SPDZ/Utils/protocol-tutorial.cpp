/*
 * protocol-tutorial.cpp
 *
 * This file demonstrates the use of the low-level capabilities
 * to build a protocol, namely Rep3 multiplication and opening.
 *
 */

#include "Networking/CryptoPlayer.h"
#include "Math/Z2k.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <playerno>" << endl;
        exit(1);
    }

    // set up networking on localhost
    int my_number = atoi(argv[1]);
    int port_base = 9999;
    Names N(my_number, 3, "localhost", port_base);
    CryptoPlayer P(N);

    // correlated randomness for resharing
    SeededPRNG G[2];

    // synchronize with other parties
    octetStream os;
    os.append(G[0].get_seed(), SEED_SIZE);
    P.pass_around(os, os, 1);
    G[1].SetSeed(os.consume(SEED_SIZE));

    // simplify code
    typedef Z2<64> Z;

    // start with same shares on all parties for simplicity
    // replicated secret sharing of 3
    Z a[2] = {1, 1};
    // and 6
    Z b[2] = {2, 2};

    // compute an additive sharing of the product
    Z cc = a[0] * (b[0] + b[1]) + a[1] * b[0];

    // result shares
    Z c[2];

    // re-randomize
    c[0] = cc + G[0].get<Z>() - G[1].get<Z>();

    // send and receive share
    os.reset_write_head();
    c[0].pack(os);
    P.pass_around(os, os, 1);
    c[1].unpack(os);

    // open value to party 0
    if (P.my_num() == 1)
    {
        os.reset_write_head();
        c[0].pack(os);
        P.send_to(0, os);
    }

    // output result on party 0, which should be 18
    if (P.my_num() == 0)
    {
        P.receive_player(1, os);
        cout << "My shares: " << c[0] << ", " << c[1] << endl;
        cout << "Result: " << (os.get<Z>() + c[0] + c[1]) << endl;
    }
}
