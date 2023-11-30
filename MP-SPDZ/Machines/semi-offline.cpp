
#include "GC/TinierSecret.h"
#include "GC/Semi.h"
#include "GC/SemiPrep.h"

#include "Semi.hpp"
#include "Math/gfp.hpp"
#include "Processor/FieldMachine.hpp"
#include "Processor/OfflineMachine.hpp"
#include "Protocols/MascotPrep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityFieldMachine<SemiShare, SemiShare, gf2n,
            OfflineMachine<DishonestMajorityMachine>>(argc, argv, opt);
}
