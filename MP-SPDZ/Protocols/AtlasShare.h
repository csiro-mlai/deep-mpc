/*
 * AtlasShare.h
 *
 */

#ifndef PROTOCOLS_ATLASSHARE_H_
#define PROTOCOLS_ATLASSHARE_H_

#include "ShamirShare.h"

template<class T> class Atlas;
template<class T> class AtlasPrep;

namespace GC
{
class AtlasSecret;
}

template<class T>
class AtlasShare : public ShamirShare<T>
{
    typedef AtlasShare This;
    typedef ShamirShare<T> super;

public:
    typedef Atlas<This> Protocol;
    typedef ::Input<This> Input;
    typedef IndirectShamirMC<This> MAC_Check;
    typedef ShamirMC<This> Direct_MC;
    typedef ::PrivateOutput<This> PrivateOutput;
    typedef AtlasPrep<This> LivePrep;
    typedef LivePrep TriplePrep;

#ifndef NO_MIXED_CIRCUITS
    typedef GC::AtlasSecret bit_type;
#endif

    AtlasShare()
    {
    }

    template<class U>
    AtlasShare(const U& other) :
            super(other)
    {
    }
};

#endif /* PROTOCOLS_ATLASSHARE_H_ */
