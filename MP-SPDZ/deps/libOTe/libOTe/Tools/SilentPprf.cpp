#include "SilentPprf.h"
#ifdef ENABLE_SILENTOT

#include <cryptoTools/Common/Log.h>
#include <cryptoTools/Crypto/RandomOracle.h>
#include <libOTe/Tools/Tools.h>
//#define DEBUG_PRINT_PPRF

namespace osuCrypto
{

    void SilentMultiPprfSender::setBase(span<const std::array<block, 2>> baseMessages)
    {
        if (baseOtCount() != static_cast<u64>(baseMessages.size()))
            throw RTE_LOC;

        mBaseOTs.resize(mPntCount, mDepth);
        for (u64 i = 0; i < static_cast<u64>(mBaseOTs.size()); ++i)
            mBaseOTs(i) = baseMessages[i];
    }

    void SilentMultiPprfReceiver::setBase(span<const block> baseMessages)
    {
        if (baseOtCount() != static_cast<u64>(baseMessages.size()))
            throw RTE_LOC;

        // The OTs are used in blocks of 8, so make sure that there is a whole
        // number of blocks.
        mBaseOTs.resize(roundUpTo(mPntCount, 8), mDepth);
        memcpy(mBaseOTs.data(), baseMessages.data(), baseMessages.size() * sizeof(block));
    }

    // This function copies the leaf values of the GGM tree
    // to the output location. There are two modes for this
    // funcation. If interleaved == false, then each tree is
    // copied to a different contiguous regious of the output.
    // If interleaved == true, then trees are interleaved such that ....
    // @lvl         - the GGM tree leafs.
    // @output      - the location that the GGM leafs should be written to.
    // @numTrees    - How many trees there are in total.
    // @tIdx        - the index of the first tree.
    // @oFormat     - do we interleave the output?
    // @mal         - ...
    void copyOut(
        span<std::array<block, 8>> lvl,
        MatrixView<block> output,
        u64 totalTrees,
        u64 tIdx,
        PprfOutputFormat oFormat)
    {

        if (oFormat == PprfOutputFormat::InterleavedTransposed)
        {
            // not having an even (8) number of trees is not supported.
            if (totalTrees % 8)
                throw RTE_LOC;
            if (lvl.size() % 16)
                throw RTE_LOC;

            //
            //auto rowsPer = 16;
            //auto step = lvl.size()

            //auto sectionSize =

            if (lvl.size() < 16)
                throw RTE_LOC;


            auto setIdx = tIdx / 8;
            auto blocksPerSet = lvl.size() * 8 / 128;



            auto numSets = totalTrees / 8;
            auto begin = setIdx;
            auto step = numSets;

            if (oFormat == PprfOutputFormat::InterleavedTransposed)
            {
                auto end = std::min<u64>(begin + step * blocksPerSet, output.cols());

                for (u64 i = begin, k = 0; i < end; i += step, ++k)
                {
                    auto& io = *(std::array<block, 128>*)(&lvl[k * 16]);
                    transpose128(io.data());
                    for (u64 j = 0; j < 128; ++j)
                        output(j, i) = io[j];
                }
            }
            else
            {
                // no op
            }


        }
        else if (oFormat == PprfOutputFormat::Plain)
        {

            auto curSize = std::min<u64>(totalTrees - tIdx, 8);
            if (curSize == 8)
            {

                for (u64 i = 0; i < output.rows(); ++i)
                {
                    auto oi = output[i].subspan(tIdx, 8);
                    auto& ii = lvl[i];
                    oi[0] = ii[0];
                    oi[1] = ii[1];
                    oi[2] = ii[2];
                    oi[3] = ii[3];
                    oi[4] = ii[4];
                    oi[5] = ii[5];
                    oi[6] = ii[6];
                    oi[7] = ii[7];
                }
            }
            else
            {
                for (u64 i = 0; i < output.rows(); ++i)
                {
                    auto oi = output[i].subspan(tIdx, curSize);
                    auto& ii = lvl[i];
                    for (u64 j = 0; j < curSize; ++j)
                        oi[j] = ii[j];
                }
            }

        }
        else if (oFormat == PprfOutputFormat::BlockTransposed)
        {

            auto curSize = std::min<u64>(totalTrees - tIdx, 8);
            if (curSize == 8)
            {
                for (u64 i = 0; i < output.cols(); ++i)
                {
                    auto& ii = lvl[i];
                    output(tIdx + 0, i) = ii[0];
                    output(tIdx + 1, i) = ii[1];
                    output(tIdx + 2, i) = ii[2];
                    output(tIdx + 3, i) = ii[3];
                    output(tIdx + 4, i) = ii[4];
                    output(tIdx + 5, i) = ii[5];
                    output(tIdx + 6, i) = ii[6];
                    output(tIdx + 7, i) = ii[7];
                }
            }
            else
            {
                for (u64 i = 0; i < output.cols(); ++i)
                {
                    auto& ii = lvl[i];
                    for (u64 j = 0; j < curSize; ++j)
                        output(tIdx + j, i) = ii[j];
                }
            }

        }
        else if (oFormat == PprfOutputFormat::Interleaved)
        {
            // no op
        }
        else
            throw RTE_LOC;
    }

    u64 interleavedPoint(u64 point, u64 treeIdx, u64 totalTrees, u64 domain, PprfOutputFormat format)
    {


        switch (format)
        {
        case osuCrypto::PprfOutputFormat::Interleaved:
        {

            if (domain <= point)
                return ~u64(0);

            auto subTree = treeIdx % 8;
            auto forest = treeIdx / 8;

            return (forest * domain + point) * 8 + subTree;
        }
        break;
        case osuCrypto::PprfOutputFormat::InterleavedTransposed:
        {
            auto numSets = totalTrees / 8;

            auto setIdx = treeIdx / 8;
            auto subIdx = treeIdx % 8;

            auto sectionIdx = point / 16;
            auto posIdx = point % 16;


            auto setOffset = setIdx * 128;
            auto subOffset = subIdx + 8 * posIdx;
            auto secOffset = sectionIdx * numSets * 128;

            return setOffset + subOffset + secOffset;
        }
        default:
            throw RTE_LOC;
            break;
        }
        //auto totalTrees = points.size();

    }

    void interleavedPoints(span<u64> points, u64 domain, PprfOutputFormat format)
    {

        for (i64 i = 0; i < points.size(); ++i)
        {
            points[i] = interleavedPoint(points[i], i, points.size(), domain, format);
        }
    }

    u64 getActivePath(const span<u8>& choiceBits)
    {
        u64 point = 0;
        for (i64 i = 0; i < choiceBits.size(); ++i)
        {
            auto shift = choiceBits.size() - i - 1;

            point |= u64(1 ^ choiceBits[i]) << shift;
        }
        return point;
    }

    void SilentMultiPprfReceiver::getPoints(span<u64> points, PprfOutputFormat format)
    {

        switch (format)
        {
        case PprfOutputFormat::Plain:
        case PprfOutputFormat::BlockTransposed:

            memset(points.data(), 0, points.size() * sizeof(u64));
            for (u64 j = 0; j < mPntCount; ++j)
            {
                points[j] = getActivePath(mBaseChoices[j]);
            }

            break;
        case PprfOutputFormat::InterleavedTransposed:
        case PprfOutputFormat::Interleaved:

            if ((u64)points.size() != mPntCount)
                throw RTE_LOC;
            if (points.size() % 8)
                throw RTE_LOC;

            getPoints(points, PprfOutputFormat::Plain);
            interleavedPoints(points, mDomain, format);

            break;
        default:
            throw RTE_LOC;
            break;
        }
    }


    BitVector SilentMultiPprfReceiver::sampleChoiceBits(u64 modulus, PprfOutputFormat format, PRNG& prng)
    {
        BitVector choices(mPntCount * mDepth);

        // The points are read in blocks of 8, so make sure that there is a
        // whole number of blocks.
        mBaseChoices.resize(roundUpTo(mPntCount, 8), mDepth);
        for (u64 i = 0; i < mPntCount; ++i)
        {
            u64 idx;
            switch (format)
            {
            case osuCrypto::PprfOutputFormat::Plain:
            case osuCrypto::PprfOutputFormat::BlockTransposed:
                do {
                    for (u64 j = 0; j < mDepth; ++j)
                        mBaseChoices(i, j) = prng.getBit();
                    idx = getActivePath(mBaseChoices[i]);
                } while (idx >= modulus);

                break;
            case osuCrypto::PprfOutputFormat::Interleaved:
            case osuCrypto::PprfOutputFormat::InterleavedTransposed:

                // make sure that atleast the first element of this tree
                // is within the modulus.
                idx = interleavedPoint(0, i, mPntCount, mDomain, format);
                if (idx >= modulus)
                    throw RTE_LOC;


                do {
                    for (u64 j = 0; j < mDepth; ++j)
                        mBaseChoices(i, j) = prng.getBit();
                    idx = getActivePath(mBaseChoices[i]);

                    idx = interleavedPoint(idx, i, mPntCount, mDomain, format);
                } while (idx >= modulus);


                break;
            default:
                throw RTE_LOC;
                break;
            }

        }

        for (u64 i = 0; i < mBaseChoices.size(); ++i)
        {
            choices[i] = mBaseChoices(i);
        }

        return choices;
    }

    void SilentMultiPprfReceiver::setChoiceBits(PprfOutputFormat format, BitVector choices)
    {
        // Make sure we're given the right number of OTs.
        if (choices.size() != baseOtCount())
            throw RTE_LOC;

        mBaseChoices.resize(roundUpTo(mPntCount, 8), mDepth);
        for (u64 i = 0; i < mPntCount; ++i)
        {
            switch (format)
            {
            case osuCrypto::PprfOutputFormat::Plain:
            case osuCrypto::PprfOutputFormat::BlockTransposed:
                for (u64 j = 0; j < mDepth; ++j)
                    mBaseChoices(i, j) = choices[mDepth * i + j];
                break;

                // Not sure what ordering would be good for Interleaved or
                // InterleavedTransposed.

            default:
                throw RTE_LOC;
                break;
            }

            if (getActivePath(mBaseChoices[i]) >= mDomain)
                throw RTE_LOC;
        }
    }

    //block SilentMultiPprfSender::expand(
    //    Channel& chl,
    //    block value,
    //    PRNG& prng,
    //    MatrixView<block> output,
    //    PprfOutputFormat oFormat, bool mal)
    //{
    //    return expand({ &chl, 1 }, value, prng, output, oFormat, mal);
    //}

    void SilentMultiPprfSender::expand(
        Channel& chls,
        block value,
        PRNG& prng,
        MatrixView<block> output,
        PprfOutputFormat oFormat,
        u64 numThreads)
    {
        std::vector<block> vv(mPntCount, value);
        expand(chls, vv, prng, output, oFormat, numThreads);
    }

    void SilentMultiPprfSender::expand(
        Channel& chl,
        span<const block> value,
        PRNG& prng,
        MatrixView<block> output,
        PprfOutputFormat oFormat,
        u64 numThreads)
    {
        setValue(value);
        setTimePoint("pprf.send.start");
        gTimer.setTimePoint("send.enter");



        if (oFormat == PprfOutputFormat::Plain)
        {
            if (output.rows() != mDomain)
                throw RTE_LOC;

            if (output.cols() != mPntCount)
                throw RTE_LOC;
        }
        else if (oFormat == PprfOutputFormat::BlockTransposed)
        {
            if (output.cols() != mDomain)
                throw RTE_LOC;

            if (output.rows() != mPntCount)
                throw RTE_LOC;
        }
        else if (oFormat == PprfOutputFormat::InterleavedTransposed)
        {
            if (output.rows() != 128)
                throw RTE_LOC;

            //if (output.cols() > (mDomain * mPntCount + 127) / 128)
            //    throw RTE_LOC;

            if (mPntCount & 7)
                throw RTE_LOC;
        }
        else if (oFormat == PprfOutputFormat::Interleaved)
        {
            if (output.cols() != 1)
                throw RTE_LOC;
            if (mDomain & 1)
                throw RTE_LOC;

            auto rows = output.rows();
            if (rows > (mDomain * mPntCount) ||
                rows / 128 != (mDomain * mPntCount) / 128)
                throw RTE_LOC;
            if (mPntCount & 7)
                throw RTE_LOC;
        }
        else
        {
            throw RTE_LOC;
        }

        // Should we program the active child to be its correct value XOR delta?
        // If not, it will just take a random value.
        const bool activeChildXorDelta = (mValue.size() != 0);


        // ss will hold the malicious check block. Will be
        // the ZeroBlock if semi-honest
        block seed = prng.get();

        // A public PRF/PRG that we will use for deriving the GGM tree.
        std::array<AES, 2> aes;
        aes[0].setKey(toBlock(3242342));
        aes[1].setKey(toBlock(8993849));

        struct TreeGrp
        {
            u64 g;
            std::array<std::vector<std::array<block, 8>>, 2> sums;
            std::vector<std::array<block, 4>> lastOts;
        };
        std::mutex sendMtx;

        //auto& chl = chls[0];
        auto sendOne = [&](TreeGrp& tg)
        {
            std::lock_guard<std::mutex> lock(sendMtx);
            chl.asyncSendCopy(tg.g);
            chl.asyncSend(std::move(tg.sums[0]));
            chl.asyncSend(std::move(tg.sums[1]));
            if (activeChildXorDelta)
                chl.asyncSend(std::move(tg.lastOts));
        };

        // The function that each thread will run. Each thread will
        // process 8 GGM trees in parallel.
        auto routine = [&](u64 threadIdx)
        {
            // A local PRNG for this thread.
            PRNG prng(seed ^ toBlock(threadIdx));

            // get our channel for this thread.
            //auto& chl = chls[threadIdx];

            TreeGrp treeGrp;

            // mySums will hold the left and right GGM tree sums
            // for each level. For example sums[0][i][5]  will
            // hold the sum of the left children for level i of
            // the 5th tree.
            std::array<std::vector<std::array<block, 8>>, 2>& sums = treeGrp.sums;

            auto dd = mDepth + (oFormat == PprfOutputFormat::Interleaved ? 0 : 1);
            // tree will hold the full GGM tree. Note that there are 8
            // indepenendent trees that are being processed together.
            // The trees are flattenned to that the children of j are
            // located at 2*j+1 and 2*j+2.
            AlignedBlockPtr uPtr = allocAlignedBlockArray(8 * (1ull << (dd)));
            span<std::array<block, 8>> tree((std::array<block, 8>*)uPtr.get(), 1ull << (dd));

#ifdef DEBUG_PRINT_PPRF
            chl.asyncSendCopy(mValue);
#endif

            // Returns the i'th level of the current 8 trees. The
            // children of node j on level i are located at 2*j and
            // 2*j+1  on level i+1.
            auto getLevel = [&](u64 i, u64 g)
            {

                if (oFormat == PprfOutputFormat::Interleaved && i == mDepth)
                {
                    auto b = (std::array<block, 8>*)output.data();
                    auto forest = g / 8;
                    assert(g % 8 == 0);
                    b += forest * mDomain;
                    return span<std::array<block, 8>>(b, mDomain);
                }

                auto size = (1ull << i);
                auto offset = (size - 1);
                auto b = tree.begin() + offset;
                auto e = b + size;
                return span<std::array<block, 8>>(b, e);
            };

            // prints out the contents of b
            //auto print = [](span<block> b)
            //{
            //    std::stringstream ss;
            //    if (b.size())
            //        ss << b[0];
            //    for (i64 i = 1; i < b.size(); ++i)
            //    {
            //        ss << ", " << b[i];
            //    }
            //    return ss.str();
            //};

            // This thread will process 8 trees at a time. It will interlace
            // thich sets of trees are processed with the other threads.
            for (u64 g = threadIdx * 8; g < mPntCount; g += 8 * numThreads)
            {
                treeGrp.g = g;

                // The number of real trees for this iteration.
                auto min = std::min<u64>(8, mPntCount - g);
                gTimer.setTimePoint("send.start" + std::to_string(g));

                // Populate the zero'th level of the GGM tree with random seeds.
                prng.get(getLevel(0, g));

                // Allocate space for our sums of each level.
                sums[0].resize(mDepth);
                sums[1].resize(mDepth);

                // For each level perform the following.
                for (u64 d = 0; d < mDepth; ++d)
                {
                    // The previous level of the GGM tree.
                    auto level0 = getLevel(d, g);

                    // The next level of theGGM tree that we are populating.
                    auto level1 = getLevel(d + 1, g);

                    // The total number of children in this level.
                    auto width = static_cast<u64>(level1.size());

                    // For each child, populate the child by expanding the parent.
                    for (u64 childIdx = 0; childIdx < width; )
                    {
                        // Index of the parent in the previous level.
                        auto parentIdx = childIdx >> 1;

                        // The value of the parent.
                        auto& parent = level0[parentIdx];

                        // The bit that indicates if we are on the left child (0)
                        // or on the right child (1).
                        for (u64 keep = 0; keep < 2; ++keep, ++childIdx)
                        {
                            // The child that we will write in this iteration.
                            auto& child = level1[childIdx];

                            // The sum that this child node belongs to.
                            auto& sum = sums[keep][d];

                            // Each parent is expanded into the left and right children
                            // using a different AES fixed-key. Therefore our OWF is:
                            //
                            //    H(x) = (AES(k0, x) + x) || (AES(k1, x) + x);
                            //
                            // where each half defines one of the children.
                            aes[keep].ecbEnc8Blocks(parent.data(), child.data());
                            child[0] = child[0] ^ parent[0];
                            child[1] = child[1] ^ parent[1];
                            child[2] = child[2] ^ parent[2];
                            child[3] = child[3] ^ parent[3];
                            child[4] = child[4] ^ parent[4];
                            child[5] = child[5] ^ parent[5];
                            child[6] = child[6] ^ parent[6];
                            child[7] = child[7] ^ parent[7];

                            // Update the running sums for this level. We keep
                            // a left and right totals for each level.
                            sum[0] = sum[0] ^ child[0];
                            sum[1] = sum[1] ^ child[1];
                            sum[2] = sum[2] ^ child[2];
                            sum[3] = sum[3] ^ child[3];
                            sum[4] = sum[4] ^ child[4];
                            sum[5] = sum[5] ^ child[5];
                            sum[6] = sum[6] ^ child[6];
                            sum[7] = sum[7] ^ child[7];
                        }
                    }
                }


#ifdef DEBUG_PRINT_PPRF
                // If we are debugging, then send over the full tree
                // to make sure its correct on the other side.
                chl.asyncSendCopy(tree);
#endif

                // For all but the last level, mask the sums with the
                // OT strings and send them over. Include the last level as
                // well if we don't need to program it to be the correct value
                // XOR delta.
                for (u64 d = 0; d < mDepth - activeChildXorDelta; ++d)
                {
                    for (u64 j = 0; j < min; ++j)
                    {
#ifdef DEBUG_PRINT_PPRF
                        if (mPrint)
                        {
                            std::cout << "c[" << g + j << "][" << d << "][0] " << sums[0][d][j] << " " << mBaseOTs[g + j][d][0] << std::endl;;
                            std::cout << "c[" << g + j << "][" << d << "][1] " << sums[1][d][j] << " " << mBaseOTs[g + j][d][1] << std::endl;;
                        }
#endif
                        sums[0][d][j] = sums[0][d][j] ^ mBaseOTs[g + j][d][0];
                        sums[1][d][j] = sums[1][d][j] ^ mBaseOTs[g + j][d][1];
                    }
                }

                if (activeChildXorDelta)
                {
                    // For the last level, we are going to do something special.
                    // The other party is currently missing both leaf children of
                    // the active parent. Since this is the last level, we want
                    // the inactive child to just be the normal value but the
                    // active child should be the correct value XOR the delta.
                    // This will be done by sending the sums and the sums plus
                    // delta and ensure that they can only decrypt the correct ones.
                    auto d = mDepth - 1;
                    std::vector<std::array<block, 4>>& lastOts = treeGrp.lastOts;
                    lastOts.resize(min);
                    for (u64 j = 0; j < min; ++j)
                    {
                        // Construct the sums where we will allow the delta (mValue)
                        // to either be on the left child or right child depending
                        // on which has the active path.
                        lastOts[j][0] = sums[0][d][j];
                        lastOts[j][1] = sums[1][d][j] ^ mValue[g + j];
                        lastOts[j][2] = sums[1][d][j];
                        lastOts[j][3] = sums[0][d][j] ^ mValue[g + j];

                        // We are going to expand the 128 bit OT string
                        // into a 256 bit OT string using AES.
                        std::array<block, 4> masks, maskIn;
                        maskIn[0] = mBaseOTs[g + j][d][0];
                        maskIn[1] = mBaseOTs[g + j][d][0] ^ AllOneBlock;
                        maskIn[2] = mBaseOTs[g + j][d][1];
                        maskIn[3] = mBaseOTs[g + j][d][1] ^ AllOneBlock;
                        mAesFixedKey.ecbEncFourBlocks(maskIn.data(), masks.data());
                        masks[0] = masks[0] ^ maskIn[0];
                        masks[1] = masks[1] ^ maskIn[1];
                        masks[2] = masks[2] ^ maskIn[2];
                        masks[3] = masks[3] ^ maskIn[3];

#ifdef DEBUG_PRINT_PPRF
                        if (mPrint) {
                            std::cout << "c[" << g + j << "][" << d << "][0] " << sums[0][d][j] << " " << mBaseOTs[g + j][d][0] << std::endl;;
                            std::cout << "c[" << g + j << "][" << d << "][1] " << sums[1][d][j] << " " << mBaseOTs[g + j][d][1] << std::endl;;
                        }
#endif

                        // Add the OT masks to the sums and send them over.
                        lastOts[j][0] = lastOts[j][0] ^ masks[0];
                        lastOts[j][1] = lastOts[j][1] ^ masks[1];
                        lastOts[j][2] = lastOts[j][2] ^ masks[2];
                        lastOts[j][3] = lastOts[j][3] ^ masks[3];
                    }

                    // Resize the sums to that they dont include
                    // the unmasked sums on the last level!
                    sums[0].resize(mDepth - 1);
                    sums[1].resize(mDepth - 1);
                }

                // Send the sums to the other party.
                sendOne(treeGrp);
                //chl.asyncSend(std::move(sums[0]));
                //chl.asyncSend(std::move(sums[1]));

                //// send the special OT messages for the last level.
                //chl.asyncSend(std::move(lastOts));
                gTimer.setTimePoint("send.expand_send");

                // copy the last level to the output. If desired, this is
                // where the tranpose is performed.
                auto lvl = getLevel(mDepth, g);

                // s is a checksum that is used for malicous security.
                copyOut(lvl, output, mPntCount, g, oFormat);
            }
        };

        std::vector<std::thread> thrds(numThreads-1);
        for (u64 i = 0; i < thrds.size(); ++i)
            thrds[i] = std::thread(routine, i);

        routine(thrds.size());

        for (u64 i = 0; i < thrds.size(); ++i)
            thrds[i].join();




        mBaseOTs = {};
    }

    void SilentMultiPprfSender::setValue(span<const block> value)
    {
        if (value.size() != 0 && (u64)value.size() != mPntCount)
            throw RTE_LOC;

        mValue.resize(value.size());
        std::copy(value.begin(), value.end(), mValue.begin());
    }

    void SilentMultiPprfSender::clear()
    {
        mBaseOTs.resize(0, 0);
        mDomain = 0;
        mDepth = 0;
        mPntCount = 0;
    }

    void SilentMultiPprfReceiver::expand(Channel& chl, PRNG& prng, MatrixView<block> output,
        PprfOutputFormat oFormat, bool activeChildXorDelta,
        u64 numThreads)
    {

        setTimePoint("pprf.recv.start");

        //lout << " d " << mDomain << " p " << mPntCount << " do " << mDepth << std::endl;

        if (oFormat == PprfOutputFormat::Plain)
        {
            if (output.rows() != mDomain)
                throw RTE_LOC;

            if (output.cols() != mPntCount)
                throw RTE_LOC;
        }
        else if (oFormat == PprfOutputFormat::BlockTransposed)
        {
            if (output.cols() != mDomain)
                throw RTE_LOC;

            if (output.rows() != mPntCount)
                throw RTE_LOC;
        }
        else if (oFormat == PprfOutputFormat::InterleavedTransposed)
        {
            if (output.rows() != 128)
                throw RTE_LOC;

            //if (output.cols() > (mDomain * mPntCount + 127) / 128)
            //    throw RTE_LOC;

            if (mPntCount & 7)
                throw RTE_LOC;
        }
        else if (oFormat == PprfOutputFormat::Interleaved)
        {
            if (output.cols() != 1)
                throw RTE_LOC;
            if (mDomain & 1)
                throw RTE_LOC;
            auto rows = output.rows();
            if (rows > (mDomain * mPntCount) ||
                rows / 128 != (mDomain * mPntCount) / 128)
                throw RTE_LOC;
            if (mPntCount & 7)
                throw RTE_LOC;
        }
        else
        {
            throw RTE_LOC;
        }

        gTimer.setTimePoint("recv.enter");

        // The vector holding the indices of the active
        // leaves. Each index is in [0,mDomain).
        std::vector<u64> points(roundUpTo(mPntCount, 8));
        getPoints(points, PprfOutputFormat::Plain);

        // A public PRF/PRG that we will use for deriving the GGM tree.
        std::array<AES, 2> aes;
        aes[0].setKey(toBlock(3242342));
        aes[1].setKey(toBlock(8993849));
        Timer& timer = gTimer;
        //block X = prng.get();


        std::mutex recvMtx;

        // The function that each thread will run. Each thread will
        // process 8 GGM trees in parallel.
        auto routine = [&](u64 threadIdx)
        {
            // get our channel for this thread.
            //auto& chl = chls[threadIdx];
            gTimer.setTimePoint("recv.routine");

            // mySums will hold the left and right GGM tree sums
            // for each level. For example mySums[5][0]  will
            // hold the sum of the left children for the 5th tree. This
            // sum will be "missing" the children of the active parent.
            // The sender will give of one of the full somes so we can
            // compute the missing inactive child.
            std::array<std::array<block, 8>, 2> mySums;

            // A buffer for receiving the sums from the other party.
            // These will be masked by the OT strings.
            std::array<std::vector<std::array<block, 8>>, 2> theirSums;
            theirSums[0].resize(mDepth - activeChildXorDelta);
            theirSums[1].resize(mDepth - activeChildXorDelta);

            auto dd = mDepth + (oFormat == PprfOutputFormat::Interleaved ? 0 : 1);
            // tree will hold the full GGM tree. Not that there are 8
            // indepenendent trees that are being processed together.
            // The trees are flattenned to that the children of j are
            // located at 2*j  and 2*j+1.
            AlignedBlockPtr uPtr = allocAlignedBlockArray(8 * (1ull << (dd)));
            span<std::array<block, 8>> tree((std::array<block, 8>*)uPtr.get(), 1ull << (dd));

            gTimer.setTimePoint("recv.alloc");

            //std::vector<std::array<block, 8>> stack(mDepth);

#ifdef DEBUG_PRINT_PPRF
            // This will be the full tree and is sent by the reciever to help debug.
            std::vector<std::array<block, 8>> ftree(1ull << (mDepth + 1));

            // The delta value on the active path.
            //block deltaValue;
            chl.recv(mDebugValue);
#endif

            // Returns the i'th level of the current 8 trees. The
            // children of node j on level i are located at 2*j and
            // 2*j+1  on level i+1.
            auto getLevel = [&](u64 i, u64 g, bool f = false)
            {
                auto size = (1ull << i), offset = (size - 1);
#ifdef DEBUG_PRINT_PPRF
                auto b = (f ? ftree.begin() : tree.begin()) + offset;
#else
                if (oFormat == PprfOutputFormat::Interleaved && i == mDepth)
                {
                    auto b = (std::array<block, 8>*)output.data();
                    auto forest = g / 8;
                    assert(g % 8 == 0);
                    b += forest * mDomain;
                    auto zone = span<std::array<block, 8>>(b, mDomain);
                    return zone;
                }

                auto b = tree.begin() + offset;
#endif
                return span<std::array<block, 8>>(b, b + size);
            };

#ifdef DEBUG_PRINT_PPRF
            // prints out the contents of the d'th level.
            auto printLevel = [&](u64 d)
            {

                auto level0 = getLevel(d);
                auto flevel0 = getLevel(d, true);

                std::cout
                    << "---------------------\nlevel " << d
                    << "\n---------------------" << std::endl;

                std::array<block, 2> sums{ ZeroBlock ,ZeroBlock };
                for (i64 i = 0; i < level0.size(); ++i)
                {
                    for (u64 j = 0; j < 8; ++j)
                    {

                        if (neq(level0[i][j], flevel0[i][j]))
                            std::cout << Color::Red;

                        std::cout << "p[" << i << "][" << j << "] "
                            << level0[i][j] << " " << flevel0[i][j] << std::endl << Color::Default;

                        if (i == 0 && j == 0)
                            sums[i & 1] = sums[i & 1] ^ flevel0[i][j];
                    }
                }

                std::cout << "sums[0] = " << sums[0] << " " << sums[1] << std::endl;
            };
#endif


            // The number of real trees for this iteration.
            std::vector<std::array<block, 4>> lastOts(8);
            // This thread will process 8 trees at a time. It will interlace
            // thich sets of trees are processed with the other threads.
            for (u64 gg = threadIdx * 8; gg < mPntCount; gg += 8 * numThreads)
            {
#ifdef DEBUG_PRINT_PPRF
                chl.recv(ftree);
                auto l1f = getLevel(1, true);
#endif
                //timer.setTimePoint("recv.start" + std::to_string(g));

                // Receive their full set of sums for these 8 trees.
                u64 g;
                {
                    std::lock_guard<std::mutex> lock(recvMtx);
                    chl.recv(g);
                    chl.recv(theirSums[0].data(), theirSums[0].size());
                    chl.recv(theirSums[1].data(), theirSums[1].size());
                    if (activeChildXorDelta)
                        chl.recv(lastOts.data(), lastOts.size());
                }
                //TODO("Optimize this recv so that if we have fewer than 8 trees then less data is sent..");


                timer.setTimePoint("recv.recv");


                auto l1 = getLevel(1, g);

                for (u64 i = 0; i < 8; ++i)
                {
                    // For the non-active path, set the child of the root node
                    // as the OT message XOR'ed with the correction sum.
                    int notAi = mBaseChoices[i + g][0];
                    l1[notAi][i] = mBaseOTs[i + g][0] ^ theirSums[notAi][0][i];
                    l1[notAi ^ 1][i] = ZeroBlock;
                    //auto idxn = i + (notAi^1) * mPntCount8;
                    //l1[idxn] = mBaseOTs[i] ^ sums[notAi^1](i);

#ifdef DEBUG_PRINT_PPRF
                    if (neq(l1[notAi][i], l1f[notAi][i])) {
                        std::cout << "l1[" << notAi << "][" << i << "] " << l1[notAi][i] << " = "
                            << (mBaseOTs[i + g][0]) << " ^ "
                            << theirSums[notAi][0][i] << " vs " << l1f[notAi][i] << std::endl;
                    }
#endif
                }

#ifdef DEBUG_PRINT_PPRF
                if (mPrint)
                    printLevel(1);
#endif

                // For all other levels, expand the GGM tree and add in
                // the correction along the active path.
                for (u64 d = 1; d < mDepth; ++d)
                {
                    // The already constructed level. Only missing the
                    // GGM tree node value along the active path.
                    auto level0 = getLevel(d, g);

                    // The next level that we want to construct.
                    auto level1 = getLevel(d + 1, g);

                    // Zero out the previous sums.
                    memset(mySums[0].data(), 0, mySums[0].size() * sizeof(block));
                    memset(mySums[1].data(), 0, mySums[1].size() * sizeof(block));

                    // We will iterate over each node on this level and
                    // expand it into it's two children. Note that the
                    // active node will also be expanded. Later we will just
                    // overwrite whatever the value was. This is an optimization.
                    auto width = static_cast<u64>(level1.size());
                    for (u64 childIdx = 0; childIdx < width; )
                    {

                        // Index of the parent in the previous level.
                        auto parentIdx = childIdx >> 1;

                        // The value of the parent.
                        auto parent = level0[parentIdx];

                        for (u64 keep = 0; keep < 2; ++keep, ++childIdx)
                        {

                            //// The bit that indicates if we are on the left child (0)
                            //// or on the right child (1).
                            //u8 keep = childIdx & 1;


                            // The child that we will write in this iteration.
                            auto& child = level1[childIdx];

                            // Each parent is expanded into the left and right children
                            // using a different AES fixed-key. Therefore our OWF is:
                            //
                            //    H(x) = (AES(k0, x) + x) || (AES(k1, x) + x);
                            //
                            // where each half defines one of the children.
                            aes[keep].ecbEnc8Blocks(parent.data(), child.data());
                            child[0] = child[0] ^ parent[0];
                            child[1] = child[1] ^ parent[1];
                            child[2] = child[2] ^ parent[2];
                            child[3] = child[3] ^ parent[3];
                            child[4] = child[4] ^ parent[4];
                            child[5] = child[5] ^ parent[5];
                            child[6] = child[6] ^ parent[6];
                            child[7] = child[7] ^ parent[7];



#ifdef DEBUG_PRINT_PPRF
                            // For debugging, set the active path to zero.
                            for (u64 i = 0; i < 8; ++i)
                                if (eq(parent[i], ZeroBlock))
                                    child[i] = ZeroBlock;
#endif
                            // Update the running sums for this level. We keep
                            // a left and right totals for each level. Note that
                            // we are actually XOR in the incorrect value of the
                            // children of the active parent (assuming !DEBUG_PRINT_PPRF).
                            // This is ok since we will later XOR off these incorrect values.
                            auto& sum = mySums[keep];
                            sum[0] = sum[0] ^ child[0];
                            sum[1] = sum[1] ^ child[1];
                            sum[2] = sum[2] ^ child[2];
                            sum[3] = sum[3] ^ child[3];
                            sum[4] = sum[4] ^ child[4];
                            sum[5] = sum[5] ^ child[5];
                            sum[6] = sum[6] ^ child[6];
                            sum[7] = sum[7] ^ child[7];
                        }
                    }

                    // For everything but the last level we have to
                    // 1) fix our sums so they dont include the incorrect
                    //    values that are the children of the active parent
                    // 2) Update the non-active child of the active parent.
                    //
                    // When we aren't trying to get the correct value XOR delta
                    // on the active child, this works for the last level as
                    // well.
                    if (!activeChildXorDelta || d != mDepth - 1)
                    {

                        // TODO: Fix cache timing leak of active path.
                        for (u64 i = 0; i < 8; ++i)
                        {
                            // the index of the leaf node that is active.
                            auto leafIdx = points[i + g];

                            // The index of the active child node.
                            auto activeChildIdx = leafIdx >> (mDepth - 1 - d);

                            // The index of the active child node sibling.
                            auto inactiveChildIdx = activeChildIdx ^ 1;

                            // The indicator as to the left or right child is inactive
                            auto notAi = inactiveChildIdx & 1;
#ifdef DEBUG_PRINT_PPRF
                            auto prev = level1[inactiveChildIdx][i];
#endif

                            auto& inactiveChild = level1[inactiveChildIdx][i];

                            // correct the sum value by XORing off the incorrect
                            auto correctSum =
                                inactiveChild ^
                                theirSums[notAi][d][i];

                            inactiveChild =
                                correctSum ^
                                mySums[notAi][i] ^
                                mBaseOTs[i + g][d];

#ifdef DEBUG_PRINT_PPRF
                            if (mPrint)
                                std::cout << "up[" << i << "] = level1[" << inactiveChildIdx << "][" << i << "] "
                                << prev << " -> " << level1[inactiveChildIdx][i] << " " << activeChildIdx << " " << inactiveChildIdx << " ~~ "
                                << mBaseOTs[i + g][d] << " " << theirSums[notAi][d][i] << " @ " << (i + g) << " " << d << std::endl;

                            auto fLevel1 = getLevel(d + 1, true);
                            if (neq(fLevel1[inactiveChildIdx][i], inactiveChild))
                                throw RTE_LOC;
#endif
                        }
                    }
#ifdef DEBUG_PRINT_PPRF
                    if (mPrint)
                        printLevel(d + 1);
#endif

                }

                timer.setTimePoint("recv.expanded");

                if (activeChildXorDelta)
                {
                    // Now processes the last level. This one is special
                    // because we we must XOR in the correction value as
                    // before but we must also fixed the child value for
                    // the active child. To do this, we will receive 4
                    // values. Two for each case (left active or right active).
                    timer.setTimePoint("recv.recvLast");

                    auto level = getLevel(mDepth, g);
                    auto d = mDepth - 1;
                    for (u64 j = 0; j < 8; ++j)
                    {
                        // The index of the child on the active path.
                        auto activeChildIdx = points[j + g];

                        // The index of the other (inactive) child.
                        auto inactiveChildIdx = activeChildIdx ^ 1;

                        // The indicator as to the left or right child is inactive
                        auto notAi = inactiveChildIdx & 1;

                        std::array<block, 2> masks, maskIn;

                        // We are going to expand the 128 bit OT string
                        // into a 256 bit OT string using AES.
                        maskIn[0] = mBaseOTs[j + g][d];
                        maskIn[1] = mBaseOTs[j + g][d] ^ AllOneBlock;
                        mAesFixedKey.ecbEncTwoBlocks(maskIn.data(), masks.data());
                        masks[0] = masks[0] ^ maskIn[0];
                        masks[1] = masks[1] ^ maskIn[1];

                        // now get the chosen message OT strings by XORing
                        // the expended (random) OT strings with the lastOts values.
                        auto& ot0 = lastOts[j][2 * notAi + 0];
                        auto& ot1 = lastOts[j][2 * notAi + 1];
                        ot0 = ot0 ^ masks[0];
                        ot1 = ot1 ^ masks[1];

#ifdef DEBUG_PRINT_PPRF
                        auto prev = level[inactiveChildIdx][j];
#endif

                        auto& inactiveChild = level[inactiveChildIdx][j];
                        auto& activeChild = level[activeChildIdx][j];

                        // Fix the sums we computed previously to not include the
                        // incorrect child values.
                        auto inactiveSum = mySums[notAi][j] ^ inactiveChild;
                        auto activeSum = mySums[notAi ^ 1][j] ^ activeChild;

                        // Update the inactive and active child to have to correct
                        // value by XORing their full sum with out partial sum, which
                        // gives us exactly the value we are missing.
                        inactiveChild = ot0 ^ inactiveSum;
                        activeChild = ot1 ^ activeSum;

#ifdef DEBUG_PRINT_PPRF
                        auto fLevel1 = getLevel(d + 1, true);
                        if (neq(fLevel1[inactiveChildIdx][j], inactiveChild))
                            throw RTE_LOC;
                        if (neq(fLevel1[activeChildIdx][j], activeChild ^ mDebugValue))
                            throw RTE_LOC;

                        if (mPrint)
                            std::cout << "up[" << d << "] = level1[" << (inactiveChildIdx / mPntCount) << "][" << (inactiveChildIdx % mPntCount) << " "
                            << prev << " -> " << level[inactiveChildIdx][j] << " ~~ "
                            << mBaseOTs[j + g][d] << " " << ot0 << " @ " << (j + g) << " " << d << std::endl;
#endif
                    }

                    timer.setTimePoint("recv.expandLast");
                }

                // copy the last level to the output. If desired, this is
                // where the tranpose is performed.
                auto lvl = getLevel(mDepth, g);

                // s is a checksum that is used for malicous security.
                copyOut(lvl, output, mPntCount, g, oFormat);
            }
        };

        std::vector<std::thread> thrds(numThreads -1);
        for (u64 i = 0; i < thrds.size(); ++i)
            thrds[i] = std::thread(routine, i);

        routine(thrds.size());

        for (u64 i = 0; i < thrds.size(); ++i)
            thrds[i].join();

        mBaseOTs = {};
        mBaseChoices = {};
    }

}

#endif
