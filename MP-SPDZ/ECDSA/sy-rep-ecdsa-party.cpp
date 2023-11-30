/*
 * sy-ecdsa-party.cpp
 *
 */

#include "Protocols/SpdzWiseShare.h"
#include "Protocols/MaliciousRep3Share.h"
#include "Protocols/MAC_Check.h"
#include "Protocols/SpdzWiseMC.h"
#include "Protocols/SpdzWisePrep.h"
#include "Protocols/SpdzWiseInput.h"
#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "Tools/ezOptionParser.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/SemiHonestRepPrep.h"

#include "Processor/FieldMachine.hpp"
#include "Protocols/Replicated.hpp"
#include "Protocols/Share.hpp"
#include "Protocols/fake-stuff.hpp"
#include "Protocols/SpdzWise.hpp"
#include "Protocols/SpdzWisePrep.hpp"
#include "Protocols/SpdzWiseInput.hpp"
#include "Protocols/SpdzWiseShare.hpp"
#include "Processor/Data_Files.hpp"
#include "Processor/Instruction.hpp"
#include "Processor/Machine.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/RepPrep.hpp"
#include "GC/ThreadMaster.hpp"
#include "Math/gfp.hpp"
#include "Machines/MalRep.hpp"

#include "hm-ecdsa-party.hpp"

template<class T>
using SpdzWiseRep3Share = SpdzWiseShare<MaliciousRep3Share<T>>;

int main(int argc, const char** argv)
{
    run<SpdzWiseRep3Share>(argc, argv);
}
