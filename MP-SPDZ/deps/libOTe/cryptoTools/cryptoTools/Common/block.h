#pragma once
#include "cryptoTools/Common/config.h"
#include <cstdint>
#include <array>
#include <iostream>
#include <memory>
#include <new>
#include <string.h>

#include <boost/align/aligned_allocator.hpp>

namespace osuCrypto
{
    struct alignas(16) block
    {
#ifdef OC_ENABLE_SSE2
        __m128i mData;
#else
        std::uint64_t mData[2];
#endif

        block() = default;
        block(uint64_t x1, uint64_t x0)
#ifdef OC_ENABLE_SSE2
        {
            mData = _mm_set_epi64x(x1, x0);
        }
#else
            : block(std::array<std::uint64_t, 2> {x0, x1}) {}
#endif

        block(char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
#ifdef OC_ENABLE_SSE2
        {
            mData = _mm_set_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0);
        }
#else
            : block(std::array<char, 16> {
                e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
            }) {}
#endif

        explicit block(uint64_t x)
        {
            *this = block(0, x);
        }

        template<typename T,
            typename Enable = typename std::enable_if<
                std::is_trivial<T>::value &&
                (sizeof(T) <= 16) &&
                (16 % sizeof(T) == 0)
            >::type>
        block(const std::array<T, 16 / sizeof(T)>& arr)
        {
            memcpy(data(), arr.data(), 16);
        }

#ifdef OC_ENABLE_SSE2
        block(const __m128i& x)
        {
            mData = x;
        }

        operator const __m128i& () const
        {
            return mData;
        }
        operator __m128i& ()
        {
            return mData;
        }

        __m128i& m128i()
        {
            return mData;
        }
        const __m128i& m128i() const
        {
            return mData;
        }
#endif

        unsigned char* data()
        {
            return (unsigned char*) &mData;
        }

        const unsigned char* data() const
        {
            return (const unsigned char*) &mData;
        }

        template<typename T>
        typename std::enable_if<
            std::is_trivial<T>::value &&
            (sizeof(T) <= 16) &&
            (16 % sizeof(T) == 0)
            ,
            std::array<T, 16 / sizeof(T)>
        >::type as() const
        {
            std::array<T, 16 / sizeof(T)> output;
            memcpy(output.data(), data(), 16);
            return output;
        }

        // For integer types, this will be specialized with SSE futher down.
        template<typename T>
        static typename std::enable_if<
            std::is_trivial<T>::value &&
            (sizeof(T) <= 16) &&
            (16 % sizeof(T) == 0),
        block>::type allSame(T val)
        {
            std::array<T, 16 / sizeof(T)> arr;
            for (T& x: arr)
                x = val;
            return arr;
        }

        inline osuCrypto::block operator^(const osuCrypto::block& rhs) const
        {
#ifdef OC_ENABLE_SSE2
            return mm_xor_si128(rhs);
#else
            return cc_xor_si128(rhs);
#endif
        }
#ifdef OC_ENABLE_SSE2
        inline osuCrypto::block mm_xor_si128(const osuCrypto::block& rhs) const
        {
            return _mm_xor_si128(*this, rhs);
        }
#endif
        inline osuCrypto::block cc_xor_si128(const osuCrypto::block& rhs) const
        {
            auto ret = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            ret[0] ^= rhsa[0];
            ret[1] ^= rhsa[1];
            return ret;
        }

        inline osuCrypto::block& operator^=(const osuCrypto::block& rhs)
        {
            *this = *this ^ rhs;
            return *this;
        }

        inline block operator~() const
        {
            return *this ^ block(-1, -1);
        }


        inline osuCrypto::block operator&(const osuCrypto::block& rhs)const
        {
#ifdef OC_ENABLE_SSE2
            return mm_and_si128(rhs);
#else
            return cc_and_si128(rhs);
#endif
        }

        inline osuCrypto::block& operator&=(const osuCrypto::block& rhs)
        {
            *this = *this & rhs;
            return *this;
        }

#ifdef OC_ENABLE_SSE2
        inline osuCrypto::block mm_and_si128(const osuCrypto::block& rhs)const
        {
            return _mm_and_si128(*this, rhs);
        }
#endif
        inline osuCrypto::block cc_and_si128(const osuCrypto::block& rhs)const
        {
            auto ret = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            ret[0] &= rhsa[0];
            ret[1] &= rhsa[1];
            return ret;
        }


        inline osuCrypto::block operator|(const osuCrypto::block& rhs)const
        {
#ifdef OC_ENABLE_SSE2
            return mm_or_si128(rhs);
#else
            return cc_or_si128(rhs);
#endif
        }
#ifdef OC_ENABLE_SSE2
        inline osuCrypto::block mm_or_si128(const osuCrypto::block& rhs)const
        {
            return _mm_or_si128(*this, rhs);
        }
#endif
        inline osuCrypto::block cc_or_si128(const osuCrypto::block& rhs)const
        {
            auto ret = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            ret[0] |= rhsa[0];
            ret[1] |= rhsa[1];
            return ret;
        }

        inline osuCrypto::block& operator|=(const osuCrypto::block& rhs)
        {
            *this = *this | rhs;
            return *this;
        }


        inline osuCrypto::block operator<<(const std::uint8_t& rhs)const
	{
#if defined(OC_ENABLE_SSE2) and not defined(SIMDE_X86_SSE2_ENABLE_NATIVE_ALIASES)
            return mm_slli_epi64(rhs);
#else
            return cc_slli_epi64(rhs);
#endif
        }

        inline osuCrypto::block& operator<<=(const std::uint8_t& rhs)
        {
            *this = *this << rhs;
            return *this;
        }

#if defined(OC_ENABLE_SSE2) and not defined(SIMDE_X86_SSE2_ENABLE_NATIVE_ALIASES)
        inline osuCrypto::block mm_slli_epi64(const std::uint8_t& rhs)const
        {
            return _mm_slli_epi64(mData, rhs);
        }
#endif
        inline osuCrypto::block cc_slli_epi64(const std::uint8_t& rhs)const
        {
            if (rhs >= 64)
                return {0, 0};

            auto ret = as<std::uint64_t>();
            ret[0] <<= rhs;
            ret[1] <<= rhs;
            return ret;
        }

        inline block operator>>(const std::uint8_t& rhs)const
        {
#if defined(OC_ENABLE_SSE2) and not defined(SIMDE_X86_SSE2_ENABLE_NATIVE_ALIASES)
            return mm_srli_epi64(rhs);
#else
            return cc_srli_epi64(rhs);
#endif
        }

        inline osuCrypto::block& operator>>=(const std::uint8_t& rhs)
        {
            *this = *this >> rhs;
            return *this;
        }

#if defined(OC_ENABLE_SSE2) and not defined(SIMDE_X86_SSE2_ENABLE_NATIVE_ALIASES)
        inline block mm_srli_epi64(const std::uint8_t& rhs) const
        {
            return _mm_srli_epi64(mData, rhs);
        }
#endif
        inline block cc_srli_epi64(const std::uint8_t& rhs) const
        {
            if (rhs >= 64)
                return {0, 0};

            auto ret = as<std::uint64_t>();
            ret[0] >>= rhs;
            ret[1] >>= rhs;
            return ret;;
        }


        inline osuCrypto::block operator+(const osuCrypto::block& rhs)const
        {
#ifdef OC_ENABLE_SSE2
            return mm_add_epi64(rhs);
#else
            return cc_add_epi64(rhs);
#endif
        }

        inline osuCrypto::block& operator+=(const osuCrypto::block& rhs)
        {
            *this = *this + rhs;
            return *this;
        }

#ifdef OC_ENABLE_SSE2
        inline block mm_add_epi64(const osuCrypto::block& rhs) const
        {
            return _mm_add_epi64(*this, rhs);

        }
#endif
        inline block cc_add_epi64(const osuCrypto::block& rhs) const
        {
            auto ret = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            ret[0] += rhsa[0];
            ret[1] += rhsa[1];
            return ret;
        }


        inline osuCrypto::block operator-(const osuCrypto::block& rhs)const
        {
#ifdef OC_ENABLE_SSE2
            return mm_sub_epi64(rhs);
#else
            return cc_sub_epi64(rhs);
#endif
        }

        inline osuCrypto::block& operator-=(const osuCrypto::block& rhs)
        {
            *this = *this - rhs;
            return *this;
        }

#ifdef OC_ENABLE_SSE2
        inline block mm_sub_epi64(const osuCrypto::block& rhs) const
        {
            return _mm_sub_epi64(*this, rhs);

        }
#endif
        inline block cc_sub_epi64(const osuCrypto::block& rhs) const
        {
            auto ret = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            ret[0] -= rhsa[0];
            ret[1] -= rhsa[1];
            return ret;
        }

        inline block& cmov(const osuCrypto::block& rhs, bool cond);

        // Same, but expects cond to be either 0x00 or 0xff.
        inline block& cmovBytes(const osuCrypto::block& rhs, uint8_t cond);

        inline bool operator==(const osuCrypto::block& rhs) const
        {
#ifdef OC_ENABLE_AVX
            auto neq = _mm_xor_si128(*this, rhs);
            return _mm_test_all_zeros(neq, neq) != 0;
#else
            return as<std::uint64_t>() == rhs.as<std::uint64_t>();
#endif
        }

        inline bool operator!=(const osuCrypto::block& rhs)const
        {
            return !(*this == rhs);
        }


        inline bool operator<(const osuCrypto::block& rhs)const
        {
            auto lhsa = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            return lhsa[1] < rhsa[1] || (lhsa[1] == rhsa[1] && lhsa[0] < rhsa[0]);
        }

        inline bool operator>(const block& rhs) const
        {
            return rhs < *this;
        }

        inline bool operator<=(const block& rhs) const
        {
            return !(*this > rhs);
        }

        inline bool operator>=(const block& rhs) const
        {
            return !(*this < rhs);
        }



        inline block srai_epi16(int imm8) const
        {
#if defined(OC_ENABLE_SSE2) and not defined(SIMDE_X86_SSE2_ENABLE_NATIVE_ALIASES)
            return mm_srai_epi16(imm8);
#else
            return cc_srai_epi16(imm8);
#endif
        }

#if defined(OC_ENABLE_SSE2) and not defined(SIMDE_X86_SSE2_ENABLE_NATIVE_ALIASES)
        inline block mm_srai_epi16(char imm8) const
        {
            return _mm_srai_epi16(*this, imm8);
        }
#endif
        inline block cc_srai_epi16(char imm8) const
        {
            auto v = as<std::int16_t>();
            std::array<std::int16_t, 8> r;
            if (imm8 <= 15)
            {
                r[0] = v[0] >> imm8;
                r[1] = v[1] >> imm8;
                r[2] = v[2] >> imm8;
                r[3] = v[3] >> imm8;
                r[4] = v[4] >> imm8;
                r[5] = v[5] >> imm8;
                r[6] = v[6] >> imm8;
                r[7] = v[7] >> imm8;
            }
            else
            {
                for (int i = 0; i < 8; i++)
                    r[i] = (v[i] & 0x8000) ? 0xFFFF : 0;
            }
            return r;
        }


        inline int movemask_epi8() const
        {
#ifdef OC_ENABLE_SSE2
            return mm_movemask_epi8();
#else
            return cc_movemask_epi8();
#endif
        }

#ifdef OC_ENABLE_SSE2
        inline int mm_movemask_epi8() const
        {
            return _mm_movemask_epi8(*this);
        }
#endif

        inline int cc_movemask_epi8() const
        {
            int ret{ 0 };
            auto v = as<unsigned char>();
            int j = 0;
            for (int i = 7; i >= 0; --i)
                ret |= std::uint16_t(v[j++] & 128) >> i;

            for (size_t i = 1; i <= 8; i++)
                ret |= std::uint16_t(v[j++] & 128) << i;

            return ret;
        }

        inline int testc(const block& b) const
        {
#ifdef OC_ENABLE_AVX
            return mm_testc_si128(b);
#else
            return cc_testc_si128(b);
#endif
        }

        inline int cc_testc_si128(const block& rhs) const
        {
            auto lhsa = as<std::uint64_t>();
            auto rhsa = rhs.as<std::uint64_t>();
            auto v0 = ~lhsa[0] & rhsa[0];
            auto v1 = ~lhsa[1] & rhsa[1];
            return (v0 || v1) ? 0 : 1;
        }

#ifdef OC_ENABLE_SSE2
        inline int mm_testc_si128(const block& b) const
        {
            return _mm_testc_si128(*this, b);
        }
#endif

        inline void gf128Mul(const block& y, block& xy1, block& xy2) const
        {
#ifdef OC_ENABLE_PCLMUL
            mm_gf128Mul(y, xy1, xy2);
#else
            cc_gf128Mul(y, xy1, xy2);
#endif // !OC_ENABLE_PCLMUL
        }

        inline block gf128Mul(const block& y) const
        {
            block xy1, xy2;
#ifdef OC_ENABLE_PCLMUL
            mm_gf128Mul(y, xy1, xy2);
#else
            cc_gf128Mul(y, xy1, xy2);
#endif // !OC_ENABLE_PCLMUL

            return xy1.gf128Reduce(xy2);
        }

        inline block gf128Pow(std::uint64_t i) const
        {
            if (*this == block(0,0))
                return block(0, 0);

            block pow2 = *this;
            block s = block(0, 1);
            while (i)
            {
                if (i & 1)
                {
                    //s = 1 * i_0 * x^{2^{1}} * ... * i_j x^{2^{j+1}}
                    s = s.gf128Mul(pow2);
                }

                // pow2 = x^{2^{j+1}}
                pow2 = pow2.gf128Mul(pow2);
                i >>= 1;
            }

            return s;
        }


#ifdef OC_ENABLE_PCLMUL
        inline void mm_gf128Mul(const block& y, block& xy1, block& xy2) const
        {
            auto& x = *this;

            block t1 = _mm_clmulepi64_si128(x, y, (int)0x00);
            block t2 = _mm_clmulepi64_si128(x, y, 0x10);
            block t3 = _mm_clmulepi64_si128(x, y, 0x01);
            block t4 = _mm_clmulepi64_si128(x, y, 0x11);
            t2 = (t2 ^ t3);
            t3 = _mm_slli_si128(t2, 8);
            t2 = _mm_srli_si128(t2, 8);
            t1 = (t1 ^ t3);
            t4 = (t4 ^ t2);

            xy1 = t1;
            xy2 = t4;
        }
#endif
        inline void cc_gf128Mul(const block& y, block& xy1, block& xy2) const
        {
            static const constexpr std::uint64_t mod = 0b10000111;
            auto shifted = as<uint64_t>();
            auto ya = y.as<uint64_t>();
            std::array<uint64_t, 2> result0, result1;

            result0[0] = 0;
            result0[1] = 0;
            result1[0] = 0;
            result1[1] = 0;

            for (int64_t i = 0; i < 2; ++i) {
                for (int64_t j = 0; j < 64; ++j) {
                    if (ya[i] & (1ull << j)) {
                        result0[0] ^= shifted[0];
                        result0[1] ^= shifted[1];
                    }

                    if (shifted[1] & (1ull << 63)) {
                        shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                        shifted[0] = (shifted[0] << 1) ^ mod;
                    }
                    else {
                        shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                        shifted[0] = shifted[0] << 1;
                    }
                }
            }

            xy1 = result0;
            xy2 = result1;
        }


        block gf128Reduce(const block& x1) const
        {
#ifdef OC_ENABLE_PCLMUL
            return mm_gf128Reduce(x1);
#else
            return cc_gf128Reduce(x1);
#endif
        }


        block cc_gf128Reduce(const block& x1) const;

#ifdef OC_ENABLE_PCLMUL
        block mm_gf128Reduce(const block& x1) const
        {
            auto mul256_low = *this;
            auto mul256_high = x1;
            static const constexpr std::uint64_t mod = 0b10000111;

            /* reduce w.r.t. high half of mul256_high */
            const __m128i modulus = _mm_loadl_epi64((const __m128i*) & (mod));
            __m128i tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x01);
            mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(tmp, 8));
            mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(tmp, 8));

            /* reduce w.r.t. low half of mul256_high */
            tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x00);
            mul256_low = _mm_xor_si128(mul256_low, tmp);

            //std::cout << "redu " << bits(x, 128) << std::endl;
            //std::cout << "     " << bits(mul256_low, 128) << std::endl;

            return mul256_low;
        }
#endif

#ifdef OC_ENABLE_PCLMUL
        template<int imm8>
        block mm_clmulepi64_si128(const block b) const
        {
            return _mm_clmulepi64_si128(*this, b, imm8);
        }
#endif

        template<int imm8>
        block cc_clmulepi64_si128(const block b) const
        {

            std::array<uint64_t,2> shifted, result0;

            auto x = extract_epi64<imm8 & 1>();
            auto y = b.extract_epi64<(imm8 >> 4) & 1>();
            result0[0] = x * (y & 1);
            result0[1] = 0;

            for (int64_t j = 1; j < 64; ++j) {
                auto bit = (y >> j) & 1ull;

                shifted[0] = x << j;
                shifted[1] = x >> (64 - j);

                result0[0] ^= shifted[0] * bit;
                result0[1] ^= shifted[1] * bit;
            }

            return result0;
        }

        template<int imm8>
        block clmulepi64_si128(block b) const
        {
#ifdef OC_ENABLE_PCLMUL
            return mm_clmulepi64_si128<imm8>(b);
#else
            return cc_clmulepi64_si128<imm8>(b);
#endif
        }

#ifdef OC_ENABLE_PCLMUL
        template<int imm8>
        uint64_t extract_epi64() const
        {
            return _mm_extract_epi64(mData, imm8);
        }
#else
        template<int imm8>
        uint64_t extract_epi64() const
        {
            uint64_t tmp[2];
            _mm_storeu_si128((__m128i*) &tmp, mData);
            return tmp[imm8];
        }
#endif
    };

    // Block arrays of static and dynamic sizes aligned to the maximal useful alignment, meaning 32
    // bytes for avx or 16 bytes for SSE. Also includes an allocator for aligning std::vector.
#ifdef ENABLE_AVX
    template<size_t N, typename T = block>
    struct alignas(32) AlignedBlockArray : public std::array<T, N>
    {
    private:
        using Base = std::array<T, N>;

        // Use std::array's constructors, etc.
    public:
        AlignedBlockArray() = default;
        using Base::Base;
        using Base::operator=;
    };

    namespace detail
    {
        template<typename T = block>
        struct AlignedBlockDeleter
        {
            void operator()(T* ptr) const
            {
                auto alignment = std::align_val_t(std::max((size_t) 32, alignof(T)));
                operator delete[](ptr, alignment);
            }
        };
    }

    template<typename T = block>
    using AlignedBlockPtrT = std::unique_ptr<T[], detail::AlignedBlockDeleter<T>>;

    template<typename T = block>
    inline AlignedBlockPtrT<T> allocAlignedBlockArray(size_t n)
    {
        auto alignment = std::align_val_t(std::max((size_t) 32, alignof(T)));
        return AlignedBlockPtrT<T>(new(alignment) T[n]);
    }

    template<typename T = block>
    using AlignedBlockAllocatorT = boost::alignment::aligned_allocator<T, 32>;

#else
    template<size_t N, typename T = block>
    using AlignedBlockArray = std::array<T, N>;
    template<typename T = block>
    using AlignedBlockPtrT = std::unique_ptr<T[]>;

    template<typename T = block>
    inline AlignedBlockPtrT<T> allocAlignedBlockArray(size_t n)
    {
        return AlignedBlockPtrT<T>(new T[n]);
    }

    template<typename T = block>
    using AlignedBlockAllocatorT = std::allocator<T>;

#endif
    using AlignedBlockPtr = AlignedBlockPtrT<>;
    using AlignedBlockAllocator = AlignedBlockAllocatorT<>;
    using AlignedBlockAllocator2 = AlignedBlockAllocatorT<std::array<block, 2>>;

#ifdef OC_ENABLE_SSE2
    template<>
    inline block block::allSame<uint8_t>(uint8_t val)
    {
        return _mm_set1_epi8(val);
    }

    template<>
    inline block block::allSame<int8_t>(int8_t val)
    {
        return _mm_set1_epi8(val);
    }

    template<>
    inline block block::allSame<uint16_t>(uint16_t val)
    {
        return _mm_set1_epi16(val);
    }

    template<>
    inline block block::allSame<int16_t>(int16_t val)
    {
        return _mm_set1_epi16(val);
    }

    template<>
    inline block block::allSame<uint32_t>(uint32_t val)
    {
        return _mm_set1_epi32(val);
    }

    template<>
    inline block block::allSame<int32_t>(int32_t val)
    {
        return _mm_set1_epi32(val);
    }

    template<>
    inline block block::allSame<uint64_t>(uint64_t val)
    {
        return _mm_set1_epi64x(val);
    }

    template<>
    inline block block::allSame<int64_t>(int64_t val)
    {
        return _mm_set1_epi64x(val);
    }
#endif

    // Specialize to send bool to all bits.
    template<>
    inline block block::allSame<bool>(bool val)
    {
        return block::allSame<uint64_t>(-(int64_t) val);
    }

    static_assert(sizeof(block) == 16, "expected block size");
    static_assert(std::alignment_of<block>::value == 16, "expected block alignment");
    static_assert(std::is_trivial<block>::value, "expected block trivial");
    static_assert(std::is_standard_layout<block>::value, "expected block pod");
    //#define _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS
        //static_assert(std::is_trivial<block>::value, "expected block pod");

    inline block toBlock(std::uint64_t high_u64, std::uint64_t low_u64)
    {
        return block(high_u64, low_u64);
    }
    inline block toBlock(std::uint64_t low_u64) { return toBlock(0, low_u64); }
    inline block toBlock(const std::uint8_t* data) { return toBlock(((std::uint64_t*)data)[1], ((std::uint64_t*)data)[0]); }

    inline block& block::cmov(const osuCrypto::block& rhs, bool cond)
    {
        return *this ^= allSame(cond) & (*this ^ rhs);
    }

    inline block& block::cmovBytes(const osuCrypto::block& rhs, uint8_t cond)
    {
        return *this ^= allSame(cond) & (*this ^ rhs);
    }

    inline void cswap(block& x, block& y, bool cond)
    {
        block diff = block::allSame(cond) & (x ^ y);
        x ^= diff;
        y ^= diff;
    }

    inline void cswapBytes(block& x, block& y, uint8_t cond)
    {
        block diff = block::allSame(cond) & (x ^ y);
        x ^= diff;
        y ^= diff;
    }

    extern const block ZeroBlock;
    extern const block OneBlock;
    extern const block AllOneBlock;
    extern const block CCBlock;
    extern const std::array<block, 2> zeroAndAllOne;
}

std::ostream& operator<<(std::ostream& out, const osuCrypto::block& block);
namespace osuCrypto
{
    using ::operator<<;
}

inline bool eq(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
    return lhs == rhs;
}

inline bool neq(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
    return lhs != rhs;
}



namespace std {

    template <>
    struct hash<osuCrypto::block>
    {
        std::size_t operator()(const osuCrypto::block& k) const;
    };

}
