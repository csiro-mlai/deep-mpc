
#include "GC/TinierSecret.h"
#include "GC/Semi.h"
#include "GC/SemiPrep.h"
#include "Protocols/Semi2kShare.h"
#include "Protocols/SemiPrep2k.h"
#include "Math/gf2n.h"

#include "Semi.hpp"
#include "Math/gfp.hpp"
#include "Processor/RingMachine.hpp"
#include "Processor/OfflineMachine.hpp"
#include "Protocols/MascotPrep.hpp"
#include "Protocols/RepRingOnlyEdabitPrep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityRingMachine<Semi2kShare, SemiShare,
            OfflineMachine<DishonestMajorityMachine>>(argc, argv, opt);
}
