/*
 * Client.cpp
 *
 */

#include "Client.h"

inline
Client::Client(const vector<string>& hostnames, int port_base,
        int my_client_id) :
        ctx("C" + to_string(my_client_id))
{
    bigint::init_thread();

    // Setup connections from this client to each party socket
    int nparties = hostnames.size();
    plain_sockets.resize(nparties);
    sockets.resize(nparties);
    for (int i = 0; i < nparties; i++)
    {
        set_up_client_socket(plain_sockets[i], hostnames[i].c_str(), port_base + i);
        octetStream(to_string(my_client_id)).Send(plain_sockets[i]);
        sockets[i] = new client_socket(io_service, ctx, plain_sockets[i],
                "P" + to_string(i), "C" + to_string(my_client_id), true);
        if (i == 0)
            specification.Receive(sockets[0]);
    }
}

inline
Client::~Client()
{
    for (auto& socket : sockets)
    {
        delete socket;
    }
}

// Send the private inputs masked with a random value.
// Receive shares of a preprocessed triple from each SPDZ engine, combine and check the triples are valid.
// Add the private input value to triple[0] and send to each spdz engine.
template<class T>
void Client::send_private_inputs(const vector<T>& values)
{
    int num_inputs = values.size();
    octetStream os;
    vector< vector<T> > triples(num_inputs, vector<T>(3));
    vector<T> triple_shares(3);
    bool active = true;

    // Receive num_inputs triples from SPDZ
    for (size_t j = 0; j < sockets.size(); j++)
    {
#ifdef VERBOSE_COMM
        cerr << "receiving from " << j << endl << flush;
#endif

        os.reset_write_head();
        os.Receive(sockets[j]);

#ifdef VERBOSE_COMM
        cerr << "received " << os.get_length() << " from " << j << endl << flush;
#endif

        if (j == 0)
        {
            if (os.get_length() == 3 * values.size() * T::size())
                active = true;
            else
                active = false;
        }

        int n_expected = active ? 3 : 1;
        if (os.get_length() != n_expected * T::size() * values.size())
            throw runtime_error("unexpected data length in sending");

        for (int j = 0; j < num_inputs; j++)
        {
            for (int k = 0; k < n_expected; k++)
            {
                triple_shares[k].unpack(os);
                triples[j][k] += triple_shares[k];
            }
        }
    }

    if (active)
        // Check triple relations (is a party cheating?)
        for (int i = 0; i < num_inputs; i++)
        {
            if (T(triples[i][0] * triples[i][1]) != triples[i][2])
            {
                cerr << triples[i][2] << " != " << triples[i][0] << " * " << triples[i][1] << endl;
                cerr << "Incorrect triple at " << i << ", aborting\n";
                throw mac_fail();
            }
        }

    // Send inputs + triple[0], so SPDZ can compute shares of each value
    os.reset_write_head();
    for (int i = 0; i < num_inputs; i++)
    {
        T y = values[i] + triples[i][0];
        y.pack(os);
    }

    for (auto& socket : sockets)
        os.Send(socket);
}

// Receive shares of the result and sum together.
// Also receive authenticating values.
template<class T, class U>
vector<U> Client::receive_outputs(int n)
{
    vector<T> triples(3 * n);
    octetStream os;
    bool active = true;
    for (auto& socket : sockets)
    {
        os.reset_write_head();
        os.Receive(socket);
#ifdef VERBOSE_COMM
        cout << "received " << os.get_length() << endl << flush;
#endif

        if (socket == sockets[0])
        {
            if (os.get_length() == (size_t) 3 * n * T::size())
                active = true;
            else
                active = false;
        }

        int n_expected = n * (active ? 3 : 1);
        if (os.get_length() != (size_t) n_expected * T::size())
            throw runtime_error("unexpected data length in receiving");

        for (int j = 0; j < n_expected; j++)
        {
            T value;
            value.unpack(os);
            triples[j] += value;
        }
    }

    if (active)
    {
        vector<U> output_values;
        for (int i = 0; i < 3 * n; i += 3)
        {
            if (T(triples[i] * triples[i + 1]) != triples[i + 2])
            {
                cerr << "Unable to authenticate output value as correct, aborting." << endl;
                throw mac_fail();
            }
            output_values.push_back(triples[i]);
        }

        return output_values;
    }
    else
    {
        triples.resize(n);
        return triples;
    }
}
