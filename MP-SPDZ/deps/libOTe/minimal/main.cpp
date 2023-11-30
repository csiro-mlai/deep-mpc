/*
 * minimal.cpp
 *
 *  Created on: 10 Jun 2022
 *      Author: marcel
 */

#include <assert.h>
#include <iostream>

//using namespace std;
#include "tests_cryptoTools/UnitTests.h"
#include "libOTe_Tests/UnitTests.h"
#include <cryptoTools/Common/Defines.h>

using namespace osuCrypto;

#include <string.h>
#include <stdio.h>

#include <cryptoTools/Network/Channel.h>
#include <cryptoTools/Network/Session.h>
#include <cryptoTools/Network/IOService.h>
#include <numeric>
#include <cryptoTools/Common/Timer.h>

#include <iomanip>
#include "frontend/util.h"

#include "frontend/ExampleBase.h"
#include "frontend/ExampleTwoChooseOne.h"
#include "frontend/ExampleNChooseOne.h"
#include "frontend/ExampleSilent.h"
#include "frontend/ExampleVole.h"
#include "libOTe/Tools/LDPC/LdpcImpulseDist.h"

int main(int argc, char** argv)
{
    assert(argc > 1);

    // Setup networking. See cryptoTools\frontend_cryptoTools\Tutorials\Network.cpp
    IOService ios;

    // The number of OTs.
    int n = 100;

    if (argc > 2)
        n = atoi(argv[2]);

    // The code to be run by the OT receiver.

    if (atoi(argv[1]) == 1)
    {
        Channel recverChl = Session(ios, "localhost:1212", SessionMode::Client).addChannel();
        PRNG prng(sysRandomSeed());
        SoftSpokenOT::TwoOneMaliciousReceiver recver(2);

        // Choose which messages should be received.
        BitVector choices(n);
        choices[0] = 1;
        //...

        // Receive the messages
        std::vector<block, AlignedBlockAllocator> messages(n);
        recver.receive(choices, messages, prng, recverChl);

        // messages[i] = sendMessages[i][choices[i]];
    }
    else
    {
        Channel senderChl = Session(ios, "localhost:1212", SessionMode::Server).addChannel();
        PRNG prng(sysRandomSeed());
        SoftSpokenOT::TwoOneMaliciousSender sender(2);

        // Choose which messages should be sent.
        auto sendMessages = allocAlignedBlockArray<std::array<block, 2>>(n);
        sendMessages[0] = { toBlock(54), toBlock(33) };
        //...

        // Send the messages.
        sender.send(span(sendMessages.get(), n), prng, senderChl);
    }
}
