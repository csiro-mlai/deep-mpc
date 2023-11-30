/*
 * atlas-ecdsa-party.cpp
 *
 */

#define NO_MIXED_CIRCUITS

#include "Machines/Atlas.hpp"

#include "hm-ecdsa-party.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    ShamirOptions(opt, argc, argv);
    run<AtlasShare>(argc, argv);
}
