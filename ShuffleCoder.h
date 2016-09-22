#pragma once

#include "Tile.h"
#include "Indices.h"
#include "RemoveZeros.h"
#include "Pack.h"

namespace codec {

// codec for sparse bit arrays.
// compressed size is within 4-12% of entropy for densities between 0.33-19%.
// compression / decompression throughput exceeds memcpy (!) and
// reaches 99% of the peak memory bandwidth (using Stream ensures we are
// writing to memory rather than cache).
//
// the name "ShuffleCoder" honors the crucial PSHUFB (universal shuffle) instruction.
//
// t_Packet is an N-bit integer type, such that each packet (on average)
// contains one 1-bit from the given input bit distribution.
//
// each t_Packet is encoded by storing indices of any 1-bits plus
// one bit indicating whether another index is required.
//
// for most t_Packet, the minimum unit of work is a "bundle"
// corresponding to one (output) value per VecU8 lane. for example,
// Uint32 packets need four VecU32 for the requisite 16 lanes (= SSE4 kVectorSize).
//
// because packets are often (20-30%) entirely zero, we increase compression
// by providing a single nonZeroBit without a preceding index.
// to avoid individual bit operations, all bits for a bundle are
// stored together. actually, SSE4 kVectorSize (16) is too small for
// writing the array efficiently, so we deal with "pairs" of bundles.
//
// specialized implementations exist for 4, 8, 128 and 256 bit packets.
//
// index bytes are appended to a temporary buffer, which is
// subsequently packed into a bit stream (see Pack.h).
typedef Uint32 NonZeroBits;

// primary templates for use by EncodeTileShuffle/DecodeTileShuffle.
// these process a pair of VecU8 lanes of packets;
// only nonzero indices are stored.
template<typename t_Packet> struct Bundle
{
    //static INLINE size_t Encode(const PtrR p_Values, PtrW& p_PosIndices);
    //static INLINE void Decode(size_t p_NonZeroBits, PtrR& p_PosIndices, const PtrW p_Decoded);
};

//-----------------------------------------------------------------------------
// BitsPerOffset = 4 (U16)

// (a 2-bit selector with 011+ is only 0.4% better - not worth the trouble;
// 012+ is worse, and S3 is also uncompetitive.)
template<> struct Bundle<Uint16>
{
    // 4 + 24N ops
    static INLINE size_t Encode(const PtrR p_Values, PtrW& p_PosIndices)
    {
        const VecI16 negOne((VecU16(SetMax())));

        auto valuesL = LoadA<VecI16>(p_Values + 0 * kVectorSize);
        auto valuesH = LoadA<VecI16>(p_Values + 1 * kVectorSize);

        VecI16 nonZeroL = NonZero(valuesL);
        VecI16 nonZeroH = NonZero(valuesH);
        VecI8 nonZero = Pack(nonZeroL, nonZeroH);

        const size_t nonZeroBits = _mm_movemask_epi8(nonZero);
        size_t remainingBits = nonZeroBits;
        while (remainingBits != 0)
        {
            VecU8 indices = IndicesOfLowest1(valuesL, valuesH);

            valuesL &= valuesL + negOne;
            valuesH &= valuesH + negOne;
            nonZeroL = NonZero(valuesL);
            nonZeroH = NonZero(valuesH);
            nonZero = Pack(nonZeroL, nonZeroH);
            // set "further index follows" bit
            indices += VecU8(VecU16(nonZero) >> 3);

            p_PosIndices = RemoveAndStore(indices, remainingBits, p_PosIndices);
            remainingBits = _mm_movemask_epi8(nonZero);
        }

        return nonZeroBits;
    }

    // 4 + 15N ops
    static INLINE PtrR Decode(size_t p_NonZeroBits, const PtrR p_PosIndices, const PtrW p_Decoded)
    {
        PtrR posIndices = p_PosIndices;

        // see U32 comment below.
        const VecU8 k80(0x80);
        VecU16 values0(k80 - k80);
        VecU16 values1 = values0;

        while (p_NonZeroBits != 0)
        {
            VecU8 indices;
            posIndices = LoadAndRestore80(k80, posIndices, p_NonZeroBits, indices);

            {
                // upper bit is set if another bit index is needed to reconstruct the value.
                const VecU8 furtherBits(VecU16(indices) << 3);
                p_NonZeroBits &= _mm_movemask_epi8(furtherBits);
            }

            // (values remain unchanged if indices == 0x80)
            ValuesFromIndices16(indices, values0, values1);
        }

        // (about the same as StoreA)
        (void)Stream(values0, p_Decoded + 0 * kVectorSize);
        (void)Stream(values1, p_Decoded + 1 * kVectorSize);

        return posIndices;
    }
};

//-----------------------------------------------------------------------------
// BitsPerOffset = 5 (U32)

template<> struct Bundle<Uint32>
{
    static INLINE size_t Encode(const PtrR p_Values, PtrW& p_PosIndices)
    {
        const VecU32 negOne(SetMax());

        auto values0 = LoadA<VecU32>(p_Values + 0 * kVectorSize);
        auto values1 = LoadA<VecU32>(p_Values + 1 * kVectorSize);
        auto values2 = LoadA<VecU32>(p_Values + 2 * kVectorSize);
        auto values3 = LoadA<VecU32>(p_Values + 3 * kVectorSize);

        VecI8 nonZero = NonZero32(values0, values1, values2, values3);
        const size_t nonZeroBits = _mm_movemask_epi8(nonZero);
        size_t remainingBits = nonZeroBits;

        while (remainingBits != 0)  // values still have bit(s) to encode
        {
            VecU8 indices = IndicesOfLowest1(values0, values1, values2, values3);

            // clear lowest1
            values0 &= values0 + negOne;
            values1 &= values1 + negOne;
            values2 &= values2 + negOne;
            values3 &= values3 + negOne;

            nonZero = NonZero32(values0, values1, values2, values3);
            // set "further index follows" bit
            indices += VecU8(VecU16(nonZero) >> 2);

            p_PosIndices = RemoveAndStore(indices, remainingBits, p_PosIndices);
            remainingBits = _mm_movemask_epi8(nonZero);
        }

        return nonZeroBits;
    }

    static INLINE PtrR Decode(size_t p_NonZeroBits, const PtrR p_PosIndices, const PtrW p_Decoded)
    {
        PtrR posIndices = p_PosIndices;

        // SetZero() generates spurious byte stores!
        // _mm_setzero_si128() generates VXORPS and assignments (=> extra register).
        // uninitialized __m128i or VecU8 lead to spills.
        // subtracting existing constant avoids a separate zero register.
        const VecU8 k80(0x80);
        VecU32 values0(k80 - k80);
        VecU32 values1 = values0, values2 = values0, values3 = values0;
        while (p_NonZeroBits != 0)
        {
            VecU8 indices;
            posIndices = LoadAndRestore80(k80, posIndices, p_NonZeroBits, indices);

            {
                // upper bit is set if another bit index is needed to reconstruct the value.
                const VecU8 furtherBits(VecU16(indices) << 2);
                p_NonZeroBits &= _mm_movemask_epi8(furtherBits);
            }

            // (values remain unchanged if indices == 0x80)
            ValuesFromIndices32(indices, values0, values1, values2, values3);
        }

        // (about 1% faster than StoreA)
        (void)Stream(values0, p_Decoded + 0 * kVectorSize);
        (void)Stream(values1, p_Decoded + 1 * kVectorSize);
        (void)Stream(values2, p_Decoded + 2 * kVectorSize);
        (void)Stream(values3, p_Decoded + 3 * kVectorSize);

        return posIndices;
    }
};

//-----------------------------------------------------------------------------
// BitsPerOffset = 6 (U64)

template<> struct Bundle<Uint64>
{
    static INLINE size_t Encode(const PtrR p_Values, PtrW& p_PosIndices)
    {
        const VecU64 negOne(SetMax());

        auto values0 = LoadA<VecU64>(p_Values + 0 * kVectorSize);
        auto values1 = LoadA<VecU64>(p_Values + 1 * kVectorSize);
        auto values2 = LoadA<VecU64>(p_Values + 2 * kVectorSize);
        auto values3 = LoadA<VecU64>(p_Values + 3 * kVectorSize);
        auto values4 = LoadA<VecU64>(p_Values + 4 * kVectorSize);
        auto values5 = LoadA<VecU64>(p_Values + 5 * kVectorSize);
        auto values6 = LoadA<VecU64>(p_Values + 6 * kVectorSize);
        auto values7 = LoadA<VecU64>(p_Values + 7 * kVectorSize);

        VecI8 nonZero = NonZero64(values0, values1, values2, values3, values4, values5, values6, values7);
        const size_t nonZeroBits = _mm_movemask_epi8(nonZero);
        size_t remainingBits = nonZeroBits;

        while (remainingBits != 0)  // values still have bit(s) to encode
        {
            VecU8 indices = IndicesOfLowest1(values0, values1, values2, values3,
                values4, values5, values6, values7);

            // clear lowest1
            values0 &= values0 + negOne;
            values1 &= values1 + negOne;
            values2 &= values2 + negOne;
            values3 &= values3 + negOne;
            values4 &= values4 + negOne;
            values5 &= values5 + negOne;
            values6 &= values6 + negOne;
            values7 &= values7 + negOne;

            nonZero = NonZero64(values0, values1, values2, values3, values4, values5, values6, values7);
            // set "further index follows" bit
            indices += VecU8(VecU16(nonZero) >> 1);

            p_PosIndices = RemoveAndStore(indices, remainingBits, p_PosIndices);
            remainingBits = _mm_movemask_epi8(nonZero);
        }

        return nonZeroBits;
    }

    static INLINE PtrR Decode(size_t p_NonZeroBits, const PtrR p_PosIndices, const PtrW p_Decoded)
    {
        PtrR posIndices = p_PosIndices;

        // see U32 comment above.
        const VecU8 k80(0x80);
        VecU64 values0(k80 - k80);
        VecU64 values1 = values0, values2 = values0, values3 = values0;
        VecU64 values4 = values0, values5 = values0, values6 = values0, values7 = values0;

        while (p_NonZeroBits != 0)
        {
            VecU8 indices;
            posIndices = LoadAndRestore80(k80, posIndices, p_NonZeroBits, indices);

            {
                // upper bit is set if another bit index is needed to reconstruct the value.
                const VecU8 furtherBits = indices + indices;
                p_NonZeroBits &= _mm_movemask_epi8(furtherBits);
            }

            // (values remain unchanged if indices == 0x80)
            ValuesFromIndices64(indices, values0, values1, values2, values3,
                values4, values5, values6, values7);
        }

        // (2% faster than StoreA)
        (void)Stream(values0, p_Decoded + 0 * kVectorSize);
        (void)Stream(values1, p_Decoded + 1 * kVectorSize);
        (void)Stream(values2, p_Decoded + 2 * kVectorSize);
        (void)Stream(values3, p_Decoded + 3 * kVectorSize);
        (void)Stream(values4, p_Decoded + 4 * kVectorSize);
        (void)Stream(values5, p_Decoded + 5 * kVectorSize);
        (void)Stream(values6, p_Decoded + 6 * kVectorSize);
        (void)Stream(values7, p_Decoded + 7 * kVectorSize);

        return posIndices;
    }
};

//-----------------------------------------------------------------------------
// Pair

template<typename t_Packet>
struct Pair
{
    static const size_t bundleSize = kVectorSize * sizeof(t_Packet);

    static INLINE void Encode(const PtrR p_Values, PtrW& p_PosIndices, const PtrW p_NonZeroPos)
    {
        const size_t nonZeroBitsL = Bundle<t_Packet>::Encode(p_Values, p_PosIndices);
        const size_t nonZeroBitsH = Bundle<t_Packet>::Encode(p_Values + bundleSize, p_PosIndices);
        const size_t nonZeroBits = (nonZeroBitsH << 16) + nonZeroBitsL;
        (void)Store(static_cast<NonZeroBits>(nonZeroBits), p_NonZeroPos);
    }

    // @return posIndices (the others advance by known amounts)
    static INLINE PtrR Decode(const PtrR p_PosIndices, const PtrR p_NonZeroPos, const PtrW p_Decoded)
    {
        PtrR posIndices = p_PosIndices;

        const size_t nonZeroBits = Load<NonZeroBits>(p_NonZeroPos);
        const size_t nonZeroBitsL = nonZeroBits & 0xFFFF;
        const size_t nonZeroBitsH = nonZeroBits >> 16;

        posIndices = Bundle<t_Packet>::Decode(nonZeroBitsL, posIndices, p_Decoded);
        posIndices = Bundle<t_Packet>::Decode(nonZeroBitsH, posIndices, p_Decoded + bundleSize);
        return posIndices;
    }
};

//-----------------------------------------------------------------------------
// primary

template<typename t_Packet>
static INLINE void EncodeTileShuffle(const PtrR p_Values,
    const PtrW p_Buffers, PtrW& p_PosAligned, PtrW& p_PosAny)
{
    const size_t bitsPerOffset = CEIL_LOG2(sizeof(t_Packet) * CHAR_BIT);
    const size_t sizeNonZero = kTileSize / sizeof(t_Packet) / CHAR_BIT;

    const PtrW buf1 = p_Buffers + 1 * kBufferSize;
    PtrW posIndices = buf1;
    for (size_t nonZeroPos = 0; nonZeroPos < sizeNonZero; nonZeroPos += sizeof(NonZeroBits))
    {
        Pair<t_Packet>::Encode(p_Values + nonZeroPos * sizeof(t_Packet) * CHAR_BIT,
            posIndices, p_Buffers + nonZeroPos);
    }
    const size_t numIndices = posIndices - buf1;
    ASSERT(numIndices != 0);

    const size_t numBits = 1 + bitsPerOffset;
    const size_t numTruncated = Blocks<numBits>::Truncate(numIndices);
    const size_t numRemainders = numIndices - numTruncated;
    const size_t sizeAligned = Blocks<numBits>::PackedSize(numTruncated);
    const size_t sizeAny = sizeNonZero + 2 + Remainders<numBits>::PackedSize(numRemainders);
    ASSERT(sizeAligned + sizeAny < kTileSize);

    // (kTileDim <= 64 => sizeNonZero < kVectorSize => store in any)
    p_PosAny = StoreFrom(RFromW(p_Buffers), sizeNonZero, p_PosAny);
    p_PosAny = StoreFrom(&numIndices, 2, p_PosAny);
    p_PosAny = Remainders<numBits>::Pack(RFromW(posIndices) - numRemainders, numRemainders, p_PosAny);

    p_PosAligned = Blocks<numBits>::Pack(RFromW(posIndices) - numIndices, numTruncated, p_PosAligned);
}

template<typename t_Packet>
static INLINE void DecodeTileShuffle(
    PtrR& p_PosAligned, PtrR& p_PosAny, const PtrW p_Buffers, const PtrW p_Out)
{
    const size_t sizeNonZero = kTileSize / sizeof(t_Packet) / CHAR_BIT;
    const size_t bitsPerOffset = CEIL_LOG2(sizeof(t_Packet) * CHAR_BIT);

    const PtrR nonZero = p_PosAny; p_PosAny += sizeNonZero;

    const size_t numIndices = Load<Uint32>(p_PosAny) & 0xFFFF; p_PosAny += 2;
    const size_t numBits = 1 + bitsPerOffset;
    const size_t numTruncated = Blocks<numBits>::Truncate(numIndices);
    const size_t numRemainders = numIndices - numTruncated;

    p_PosAligned = Blocks<numBits>::Unpack(p_PosAligned, numTruncated, p_Buffers);
    p_PosAny = Remainders<numBits>::Unpack(p_PosAny, numRemainders, p_Buffers + numTruncated);

    PtrR posIndices = RFromW(p_Buffers);
    for (size_t posNonZero = 0; posNonZero < sizeNonZero; posNonZero += sizeof(NonZeroBits))
    {
        posIndices = Pair<t_Packet>::Decode(posIndices,
            nonZero + posNonZero, p_Out + posNonZero * sizeof(t_Packet) * CHAR_BIT);
    }
}

void TestShuffleCoder();

}  // namespace codec

#include "ShuffleCoder8.h"
