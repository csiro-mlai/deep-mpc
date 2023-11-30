
#include "GC/TinierSecret.h"
#include "GC/Semi.h"
#include "GC/SemiPrep.h"
#include "Protocols/HemiShare.h"

#include "Semi.hpp"
#include "Math/gfp.hpp"
#include "Processor/FieldMachine.hpp"
#include "Processor/OfflineMachine.hpp"
#include "Protocols/MascotPrep.hpp"
#include "Protocols/HemiPrep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityFieldMachine<HemiShare, HemiShare, gf2n_short,
            OfflineMachine<DishonestMajorityMachine>>(argc, argv, opt);
}
