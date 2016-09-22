#pragma once

#include "Bits.h"  // ROUND_UP
#include "Ptr.h"
#include "BitBuffer.h"

namespace codec {

// remove the upper 0-bits ("null suppression") from numbers of a
// known and smaller bit depth stored within larger `cell' types.
// example: 4-bit within 8-bit cells, or 9-bit within 16-bit cells.
// at first, only 8-bit cells were needed, but LSBPP now supports
// 256-bit packets with 1+8 bits. cells larger than 16-bit are
// unlikely to be useful and are not yet supported.
//
// LSBPP packs directly to the output stream (which need not stay cache-resident),
// so we use streaming stores. note that unpacked cells WILL be accessed again,
// so they use cached writes.
//
// streaming stores must be aligned (and ideally fill cache lines),
// so the output stream must be aligned (=> every tile is padded).
// we therefore pack in `blocks'. their packed size is typically the
// smallest possible number of whole vectors.

template<bool t_Large> struct CellType { typedef Uint8 T; };
template<> struct CellType<true> { typedef Uint16 T; };

template<size_t t_NumBits>
struct Cell
{
    typedef typename CellType<(t_NumBits > CHAR_BIT)>::T T;
};

//-----------------------------------------------------------------------------
// remainder

// because blocks are fairly large, it is important to pack any remaining
// cells efficiently; we use a bit buffer.

template<size_t t_NumBits>
struct Remainders
{
    typedef typename Cell<t_NumBits>::T Cell;

    // predict how many bytes will be required to store that many cells.
    // (required for reserving space beforehand)
    static INLINE size_t PackedSize(size_t p_NumCells)
    {
        const size_t numBits = p_NumCells * t_NumBits;
        const size_t numBytes = Align<CHAR_BIT>(numBits) / CHAR_BIT;
        return numBytes;
    }

    // @return packedPos
    static INLINE PtrW Pack(const PtrR p_Cells, size_t p_NumCells, const PtrW p_Packed)
    {
        if (p_NumCells == 0) return p_Packed;

        PtrW packedPos = p_Packed;

        BitSink buf;
        for (size_t i = 0; i < p_NumCells; ++i)
        {
            const size_t cell = Load<Cell>(p_Cells + i * sizeof(Cell));
            buf.Insert(cell, t_NumBits);
            packedPos = buf.FlushLowerBits(packedPos);
        }
        packedPos = buf.Flush(packedPos);

        ASSERT(packedPos == p_Packed + PackedSize(p_NumCells));
        return packedPos;
    }

    // @return packedPos
    static INLINE PtrR Unpack(const PtrR p_Packed, size_t p_NumCells, const PtrW p_Cells)
    {
        if (p_NumCells == 0) return p_Packed;

        BitSource source;
        PtrR packedPos = source.FillBuffer(p_Packed);
        for (size_t i = 0; i < p_NumCells; ++i)
        {
            const size_t cell = source.Extract(t_NumBits);
            Store(static_cast<Cell>(cell), p_Cells + i * sizeof(Cell));
            if (source.NeedsRefill())
            {
                packedPos = source.Refill(packedPos);
            }
        }
        packedPos = source.Rewind(packedPos);
        return packedPos;
    }
};

// specialize this for every t_NumBits (= 1 + bitsPerPosition).
template<size_t t_NumBits> struct BlockTraits
{
    //static const size_t size, packedSize;
    //static INLINE PtrW Pack(const PtrR p_Bytes, const PtrW p_Packed);
    //static INLINE PtrR Unpack(const PtrR p_Packed, const PtrW p_Bytes);
};

//-----------------------------------------------------------------------------
// 1

// 4 ops for 32 bytes
static INLINE void Pack32BytesTo32Bits(const PtrR p_Bytes, const PtrW p_Packed)
{
    const VecU8 bytes0 = LoadA<VecU8>(p_Bytes + 0 * kVectorSize);
    const VecU8 bytes1 = LoadA<VecU8>(p_Bytes + 1 * kVectorSize);
    const VecU8 zero = bytes0 - bytes0;
    // 0 or 1 => can use signed 8-bit comparison
    const size_t bits0 = _mm_movemask_epi8(_mm_cmpgt_epi8(bytes0, zero));
    const size_t bits1 = _mm_movemask_epi8(_mm_cmpgt_epi8(bytes1, zero));
    const size_t bits = (bits1 << 16) + bits0;
    Store(static_cast<Uint32>(bits), p_Packed);
}

// 9 ops for 32 bytes
static INLINE void Unpack32BytesFrom32Bits(const PtrR p_Packed, const PtrW p_Bytes)
{
    const VecU8 bytes3210 = Load64<VecU8>(p_Packed);
    const VecU8 bytes32(_mm_srli_si128(bytes3210, 2));

    // broadcast bytes to quads (8 different bit values)
    const VecU8 control(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    const VecU8 quads0 = Shuffle(bytes3210, control);
    const VecU8 quads1 = Shuffle(bytes32, control);

    const VecU8 zero = control - control;
    const VecU8 slices(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01,
        0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
    const VecU8 isZero0 = (quads0 & slices) == zero;
    const VecU8 isZero1 = (quads1 & slices) == zero;

    // convert from (isZero ? 0xFF : 0x00) to (isZero? 0 : 1)
    const VecU8 bytes0 = Shuffle(slices, isZero0);
    const VecU8 bytes1 = Shuffle(slices, isZero1);

    StoreA(bytes0, p_Bytes + 0 * kVectorSize);
    StoreA(bytes1, p_Bytes + 1 * kVectorSize);
}

// specialization of Blocks for smaller amounts of nonZeroBits -
// faster than Remainders, but not vector-aligned.
static INLINE PtrW Pack1(const PtrR p_Cells, size_t p_NumCells, const PtrW p_Packed)
{
    ASSERT((p_NumCells % 32) == 0);

    PtrW packedPos = p_Packed;
    for (size_t i = 0; i < p_NumCells; i += 32)
    {
        const PtrR in = p_Cells + i;
        Pack32BytesTo32Bits(in, packedPos);
        packedPos += 4;
    }

    return packedPos;
}

static INLINE PtrR Unpack1(const PtrR p_Packed, size_t p_NumCells, const PtrW p_Cells)
{
    ASSERT((p_NumCells % 32) == 0);

    PtrR packedPos = p_Packed;
    for (size_t i = 0; i < p_NumCells; i += 32)
    {
        const PtrW out = p_Cells + i;
        Unpack32BytesFrom32Bits(packedPos, out);
        packedPos += 4;
    }

    return packedPos;
}

template<> struct BlockTraits<1>
{
    static const size_t numCells = 8 * kVectorSize;
    static const size_t packedSize = numCells / CHAR_BIT;

    // 14 ops for 128 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        // (need U16 for shifting)
        const VecU16 bytes0 = LoadA<VecU16>(p_Bytes + 0 * kVectorSize);
        const VecU16 bytes1 = LoadA<VecU16>(p_Bytes + 1 * kVectorSize);
        const VecU16 bytes2 = LoadA<VecU16>(p_Bytes + 2 * kVectorSize);
        const VecU16 bytes3 = LoadA<VecU16>(p_Bytes + 3 * kVectorSize);
        const VecU16 bytes4 = LoadA<VecU16>(p_Bytes + 4 * kVectorSize);
        const VecU16 bytes5 = LoadA<VecU16>(p_Bytes + 5 * kVectorSize);
        const VecU16 bytes6 = LoadA<VecU16>(p_Bytes + 6 * kVectorSize);
        const VecU16 bytes7 = LoadA<VecU16>(p_Bytes + 7 * kVectorSize);

        // (digit = 1 bit)
        // 0000000B0000000A bytes0
        // 000000b0000000a  bytes1 << 1
        const VecU16 packed = (bytes7 << 7) + (bytes6 << 6) + (bytes5 << 5) + (bytes4 << 4) +
            (bytes3 << 3) + (bytes2 << 2) + (bytes1 << 1) + bytes0;

        StoreA(packed, p_Packed);
    }

    // 14 ops for 128 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask1(0x0101);  // for individual bytes (need U16 for shifting)

        VecU16 packed = LoadA<VecU16>(p_Packed);

        const VecU16 bytes0 = packed & mask1;
        packed >>= 1;
        StoreA(bytes0, p_Bytes + 0 * kVectorSize);

        const VecU16 bytes1 = packed & mask1;
        packed >>= 1;
        StoreA(bytes1, p_Bytes + 1 * kVectorSize);

        const VecU16 bytes2 = packed & mask1;
        packed >>= 1;
        StoreA(bytes2, p_Bytes + 2 * kVectorSize);

        const VecU16 bytes3 = packed & mask1;
        packed >>= 1;
        StoreA(bytes3, p_Bytes + 3 * kVectorSize);

        const VecU16 bytes4 = packed & mask1;
        packed >>= 1;
        StoreA(bytes4, p_Bytes + 4 * kVectorSize);

        const VecU16 bytes5 = packed & mask1;
        packed >>= 1;
        StoreA(bytes5, p_Bytes + 5 * kVectorSize);

        const VecU16 bytes6 = packed & mask1;
        packed >>= 1;
        StoreA(bytes6, p_Bytes + 6 * kVectorSize);

        const VecU16 bytes7 = packed & mask1;
        StoreA(bytes7, p_Bytes + 7 * kVectorSize);
    }
};

//-----------------------------------------------------------------------------
// 2

template<> struct BlockTraits<2>
{
    static const size_t packedSize = kVectorSize;
    static const size_t numCells = 4 * packedSize;

    // 6 ops for 64 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        // (need U16 for shifting)
        const VecU16 bytes0 = LoadA<VecU16>(p_Bytes + 0 * kVectorSize);
        const VecU16 bytes1 = LoadA<VecU16>(p_Bytes + 1 * kVectorSize);
        const VecU16 bytes2 = LoadA<VecU16>(p_Bytes + 2 * kVectorSize);
        const VecU16 bytes3 = LoadA<VecU16>(p_Bytes + 3 * kVectorSize);

        // (digit = 2 bits)
        // 000D000C000B000A  bytes0
        // 00d000c000b000a0  bytes1..
        const VecU16 packed = (bytes3 << 6) + (bytes2 << 4) + (bytes1 << 2) + bytes0;

        Stream(packed, p_Packed);
    }

    // 7 ops for 64 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask2(0x0303);  // for individual bytes (need U16 for shifting)

        VecU16 packed = LoadA<VecU16>(p_Packed);

        const VecU16 bytes0 = packed & mask2;
        packed >>= 2;
        StoreA(bytes0, p_Bytes + 0 * kVectorSize);

        const VecU16 bytes1 = packed & mask2;
        packed >>= 2;
        StoreA(bytes1, p_Bytes + 1 * kVectorSize);

        const VecU16 bytes2 = packed & mask2;
        packed >>= 2;
        StoreA(bytes2, p_Bytes + 2 * kVectorSize);

        const VecU16 bytes3 = packed & mask2;
        StoreA(bytes3, p_Bytes + 3 * kVectorSize);
    }
};

//-----------------------------------------------------------------------------
// 3

template<> struct BlockTraits<3>
{
    static const size_t numCells = 8 * kVectorSize;
    static const size_t packedSize = 3 * kVectorSize;

    // @return VecU16 with lower 8 bits valid
    // 23 ops for 64 bytes
    // stores 1 * kVectorSize.
    static INLINE VecU16 Pack64(const PtrR p_Bytes, const PtrW p_Packed)
    {
        const VecU8 zero(_mm_setzero_si128());

        // load 64 values, each stored in a byte => 192 useful bits
        const VecU8 bytes10 = LoadA<VecU8>(p_Bytes + 0 * kVectorSize);
        const VecU8 bytes32 = LoadA<VecU8>(p_Bytes + 1 * kVectorSize);
        const VecU8 bytes54 = LoadA<VecU8>(p_Bytes + 2 * kVectorSize);
        const VecU8 bytes76 = LoadA<VecU8>(p_Bytes + 3 * kVectorSize);

        VecU16 upperTwoBits7;
        {
            const VecU16 bytes0 = U16FromU8<0>(bytes10);
            const VecU16 bytes2 = U16FromU8<0>(bytes32);
            const VecU16 bytes4 = U16FromU8<0>(bytes54);
            const VecU16 bytes6 = U16FromU8<0>(bytes76);
            const VecU16 bytes5(UnpackHigh(bytes54, zero));
            const VecU16 bytes7(UnpackHigh(bytes76, zero));
            // each of the eight Uint16 contains five values plus an additional bit.
            const VecU16 packed = (bytes7 << 15) + (bytes6 << 12) +
                (bytes5 << 9) + (bytes4 << 6) + (bytes2 << 3) + bytes0;
            Stream(VecU8(packed), p_Packed + 0 * kVectorSize);
            upperTwoBits7 = bytes7 >> 1;
        }

        const VecU16 bytes1(UnpackHigh(bytes10, zero));
        const VecU16 bytes3(UnpackHigh(bytes32, zero));
        const VecU16 eightBits = (upperTwoBits7 << 6) + (bytes3 << 3) + bytes1;
        return eightBits;
    }

    // 20 ops for 64 bytes
    static INLINE PtrR Unpack64(const VecU16& p_EightBits, const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask3(0x7);
        VecU16 packedL = LoadA<VecU16>(p_Packed);
        VecU16 packedH = p_EightBits;

        {
            const VecU8 bytes10 = ::Pack(packedL & mask3, packedH & mask3);
            StoreA(bytes10, p_Bytes + 0 * kVectorSize);
            packedL >>= 3; packedH >>= 3;
        }

        {
            const VecU8 bytes32 = ::Pack(packedL & mask3, packedH & mask3);
            StoreA(bytes32, p_Bytes + 1 * kVectorSize);
            packedL >>= 3; packedH >>= 3;
        }

        {
            const VecU16 bytes4 = packedL & mask3;
            packedL >>= 3;
            const VecU16 bytes5 = packedL & mask3;
            packedL >>= 3;
            const VecU8 bytes54 = ::Pack(bytes4, bytes5);
            StoreA(bytes54, p_Bytes + 2 * kVectorSize);
        }

        {
            const VecU16 bytes6 = packedL & mask3;
            packedL >>= 3;
            const VecU16 bytes7 = (packedH + packedH) + packedL;
            const VecU8 bytes76 = ::Pack(bytes6, bytes7);
            StoreA(bytes76, p_Bytes + 3 * kVectorSize);
        }

        return p_Packed + kVectorSize;
    }

    // 48 ops for 128 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        const VecU16 eightBitsL = Pack64(p_Bytes +  0, p_Packed + 0 * kVectorSize);
        const VecU16 eightBitsH = Pack64(p_Bytes + 64, p_Packed + 1 * kVectorSize);
        const VecU16 eightBits = (eightBitsH << 8) + eightBitsL;
        Stream(eightBits, p_Packed + 2 * kVectorSize);
    }

    // 42 ops for 128 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask8(0x00FF);

        const VecU16 eightBits = LoadA<VecU16>(p_Packed + 2 * kVectorSize);

        const VecU16 eightBitsL = eightBits & mask8;
        Unpack64(eightBitsL, p_Packed + 0 * kVectorSize, p_Bytes + 0);

        const VecU16 eightBitsH = eightBits >> 8;
        Unpack64(eightBitsH, p_Packed + 1 * kVectorSize, p_Bytes + 64);
    }
};

//-----------------------------------------------------------------------------
// 4

template<> struct BlockTraits<4>
{
    static const size_t packedSize = kVectorSize;
    static const size_t numCells = 2 * packedSize;

    // @param p_Bytes .. 0H0G0F0E0D0C0B0A
    // 2 ops for 32 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        // (need U16 for shifting)
        const VecU16 bytes0 = LoadA<VecU16>(p_Bytes + 0 * kVectorSize);
        const VecU16 bytes1 = LoadA<VecU16>(p_Bytes + 1 * kVectorSize);

        const VecU16 packed = (bytes1 << 4) + bytes0;
        Stream(packed, p_Packed);
    }

    // @param p_Packed .. HGFEDCBA
    // 3 ops for 32 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask4(0x0F0F);  // for individual bytes (need U16 for shifting)

        VecU16 packed = LoadA<VecU16>(p_Packed);

        const VecU16 bytes0 = packed & mask4;
        packed >>= 4;
        StoreA(bytes0, p_Bytes + 0 * kVectorSize);

        const VecU16 bytes1 = packed & mask4;
        StoreA(bytes1, p_Bytes + 1 * kVectorSize);
    }
};

//-----------------------------------------------------------------------------
// 5

template<> struct BlockTraits<5>
{
    static const size_t numCells = 8 * kVectorSize;
    static const size_t packedSize = 5 * kVectorSize;

    // @return remaining upper four bits of 16 values.
    // 23 ops for 64 bytes
    // stores 32 bytes.
    static INLINE VecU8 Pack64(const PtrR p_Bytes, const PtrW p_Packed)
    {
        // load 64 values, each stored in a byte => 320 useful bits
        const VecU8 bytes10 = LoadA<VecU8>(p_Bytes + 0 * kVectorSize);
        const VecU8 bytes32 = LoadA<VecU8>(p_Bytes + 1 * kVectorSize);
        const VecU8 bytes54 = LoadA<VecU8>(p_Bytes + 2 * kVectorSize);
        const VecU8 bytes76 = LoadA<VecU8>(p_Bytes + 3 * kVectorSize);

        VecU16 upperFourBitsL;
        {
            const VecU16 bytes0 = U16FromU8<0>(bytes10);
            const VecU16 bytes2 = U16FromU8<0>(bytes32);
            const VecU16 bytes4 = U16FromU8<0>(bytes54);
            const VecU16 bytes6 = U16FromU8<0>(bytes76);
            // each of the eight Uint16 contains three values plus an additional bit.
            const VecU16 packedL = (bytes6 << 15) + (bytes4 << 10) + (bytes2 << 5) + bytes0;
            Stream(VecU8(packedL), p_Packed + 0 * kVectorSize);
            upperFourBitsL = bytes6 >> 1;
        }

        VecU16 upperFourBitsH;
        {
            const VecU8 zero(_mm_setzero_si128());
            const VecU16 bytes1(UnpackHigh(bytes10, zero));
            const VecU16 bytes3(UnpackHigh(bytes32, zero));
            const VecU16 bytes5(UnpackHigh(bytes54, zero));
            const VecU16 bytes7(UnpackHigh(bytes76, zero));
            const VecU16 packedH = (bytes7 << 15) + (bytes5 << 10) + (bytes3 << 5) + bytes1;
            Stream(VecU8(packedH), p_Packed + 1 * kVectorSize);
            upperFourBitsH = bytes7 >> 1;
        }

        const VecU8 upperFourBitsHL(_mm_packus_epi16(upperFourBitsL, upperFourBitsH));
        return upperFourBitsHL;
    }

    // 16 ops for 64 bytes
    static INLINE PtrR Unpack64(const VecU8& p_UpperFourBits, const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask5(0x1F);
        VecU16 packedL = LoadA<VecU16>(p_Packed + 0 * kVectorSize);
        VecU16 packedH = LoadA<VecU16>(p_Packed + 1 * kVectorSize);

        {
            const VecU8 bytes10 = ::Pack(packedL & mask5, packedH & mask5);
            StoreA(bytes10, p_Bytes + 0 * kVectorSize);
            packedL >>= 5; packedH >>= 5;
        }

        {
            const VecU8 bytes32 = ::Pack(packedL & mask5, packedH & mask5);
            StoreA(bytes32, p_Bytes + 1 * kVectorSize);
            packedL >>= 5; packedH >>= 5;
        }

        {
            const VecU8 bytes54 = ::Pack(packedL & mask5, packedH & mask5);
            StoreA(bytes54, p_Bytes + 2 * kVectorSize);
            packedL >>= 5; packedH >>= 5;
        }

        {
            const VecU8 singleBits = ::Pack(packedL, packedH);
            const VecU8 bytes76 = (p_UpperFourBits + p_UpperFourBits) + singleBits;
            StoreA(bytes76, p_Bytes + 3 * kVectorSize);
        }

        return p_Packed + 2 * kVectorSize;
    }

    // 48 ops for 128 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        const VecU8 upperFourBitsL = Pack64(p_Bytes +  0, p_Packed + 0 * kVectorSize);
        const VecU8 upperFourBitsH = Pack64(p_Bytes + 64, p_Packed + 2 * kVectorSize);
        const VecU8 upperFourBitsHL(VecU8(VecU16(upperFourBitsH) << 4) + upperFourBitsL);

        Stream(upperFourBitsHL, p_Packed + 4 * kVectorSize);
    }

    // 41 ops for 128 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU8 mask4(0x0F);

        const VecU8 upperFourBitsHL = LoadA<VecU8>(p_Packed + 4 * kVectorSize);

        const VecU8 upperFourBitsL = upperFourBitsHL & mask4;
        Unpack64(upperFourBitsL, p_Packed + 0 * kVectorSize, p_Bytes + 0);

        const VecU8 upperFourBitsH = VecU8(VecU16(upperFourBitsHL) >> 4) & mask4;
        Unpack64(upperFourBitsH, p_Packed + 2 * kVectorSize, p_Bytes + 64);
    }
};

//-----------------------------------------------------------------------------
// 6

template<> struct BlockTraits<6>
{
    static const size_t numCells = 4 * kVectorSize;
    static const size_t packedSize = 3 * kVectorSize;

    // 24 ops for 64 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        const VecU8 zero(_mm_setzero_si128());

        const VecU8 bytes10 = LoadA<VecU8>(p_Bytes + 0 * kVectorSize);
        const VecU8 bytes32 = LoadA<VecU8>(p_Bytes + 1 * kVectorSize);
        const VecU8 bytes54 = LoadA<VecU8>(p_Bytes + 2 * kVectorSize);
        const VecU8 bytes76 = LoadA<VecU8>(p_Bytes + 3 * kVectorSize);

        VecU16 upperTwoBits4;
        {
            // treat as Uint16 to allow shifting
            const VecU16 bytes0 = U16FromU8<0>(bytes10);
            const VecU16 bytes2 = U16FromU8<0>(bytes32);
            const VecU16 bytes4 = U16FromU8<0>(bytes54);
            const VecU16 packed0 = (bytes4 << 12) + (bytes2 << 6) + bytes0;
            Stream(VecU8(packed0), p_Packed + 0 * kVectorSize);
            upperTwoBits4 = bytes4 >> 4;
        }

        VecU16 upperTwoBits5;
        {
            const VecU16 bytes1(UnpackHigh(bytes10, zero));
            const VecU16 bytes3(UnpackHigh(bytes32, zero));
            const VecU16 bytes5(UnpackHigh(bytes54, zero));
            const VecU16 packed1 = (bytes5 << 12) + (bytes3 << 6) + bytes1;
            Stream(VecU8(packed1), p_Packed + 1 * kVectorSize);
            upperTwoBits5 = bytes5 >> 4;
        }

        {
            const VecU16 bytes6 = U16FromU8<0>(bytes76);
            const VecU16 bytes7(UnpackHigh(bytes76, zero));
            const VecU16 packed2 = (upperTwoBits5 << 14) + (upperTwoBits4 << 12) + (bytes7 << 6) + bytes6;
            Stream(VecU8(packed2), p_Packed + 2 * kVectorSize);
        }
    }

    // 20 ops for 64 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask6(0x3F);

        PtrR packedPos = p_Packed;

        VecU16 packed0 = LoadA<VecU16>(packedPos + 0 * kVectorSize);
        VecU16 packed1 = LoadA<VecU16>(packedPos + 1 * kVectorSize);
        VecU16 packed2 = LoadA<VecU16>(packedPos + 2 * kVectorSize);

        {
            const VecU8 bytes10 = ::Pack(packed0 & mask6, packed1 & mask6);
            StoreA(bytes10, p_Bytes + 0 * kVectorSize);
            packed0 >>= 6; packed1 >>= 6;
        }

        {
            const VecU8 bytes32 = ::Pack(packed0 & mask6, packed1 & mask6);
            StoreA(bytes32, p_Bytes + 1 * kVectorSize);
            packed0 >>= 6; packed1 >>= 6;
        }

        VecU8 bytes76;
        {
            const VecU16 bytes6 = packed2 & mask6;
            packed2 >>= 6;
            const VecU16 bytes7 = packed2 & mask6;
            packed2 >>= 6;
            bytes76 = ::Pack(bytes6, bytes7);
        }

        {
            const VecU16 bytes4 = packed0 + ((packed2 << 4) & mask6);  // 1% faster than 2x shift
            const VecU16 bytes5 = packed1 + ((packed2 >> 2) << 4);
            const VecU8 bytes54 = ::Pack(bytes4, bytes5);
            StoreA(bytes54, p_Bytes + 2 * kVectorSize);
            StoreA(bytes76, p_Bytes + 3 * kVectorSize);
        }
    }
};

//-----------------------------------------------------------------------------
// 7

template<> struct BlockTraits<7>
{
    static const size_t numCells = 8 * kVectorSize;
    static const size_t packedSize = 7 * kVectorSize;

    // @return upperFive5 | upperThree4
    // 25 ops for 64 bytes
    // stores 48 bytes.
    static INLINE VecU16 Pack64(const PtrR p_Bytes, const PtrW p_Packed)
    {
        const VecU8 zero(_mm_setzero_si128());

        // load 64 values, each stored in a byte => 448 useful bits
        const VecU8 bytes10 = LoadA<VecU8>(p_Bytes + 0 * kVectorSize);
        const VecU8 bytes32 = LoadA<VecU8>(p_Bytes + 1 * kVectorSize);
        const VecU8 bytes54 = LoadA<VecU8>(p_Bytes + 2 * kVectorSize);
        const VecU8 bytes76 = LoadA<VecU8>(p_Bytes + 3 * kVectorSize);

        VecU16 upperFive4;
        {
            const VecU16 bytes0 = U16FromU8<0>(bytes10);
            const VecU16 bytes2 = U16FromU8<0>(bytes32);
            const VecU16 bytes4 = U16FromU8<0>(bytes54);
            // each of the eight Uint16 contains two values plus two bits.
            const VecU16 packed0 = (bytes4 << 14) + (bytes2 << 7) + bytes0;
            Stream(VecU8(packed0), p_Packed + 0 * kVectorSize);
            upperFive4 = bytes4 >> 2;
        }

        VecU16 upperFive5;
        {
            const VecU16 bytes1(UnpackHigh(bytes10, zero));
            const VecU16 bytes3(UnpackHigh(bytes32, zero));
            const VecU16 bytes5(UnpackHigh(bytes54, zero));
            const VecU16 packed1 = (bytes5 << 14) + (bytes3 << 7) + bytes1;
            Stream(VecU8(packed1), p_Packed + 1 * kVectorSize);
            upperFive5 = bytes5 >> 2;
        }

        VecU16 upperThree4;
        {
            const VecU16 bytes6 = U16FromU8<0>(bytes76);
            const VecU16 bytes7(UnpackHigh(bytes76, zero));
            // two values plus 2 of the 5 extra bits
            const VecU16 packed2 = (upperFive4 << 14) + (bytes7 << 7) + bytes6;
            Stream(VecU8(packed2), p_Packed + 2 * kVectorSize);
            upperThree4 = upperFive4 >> 2;
        }

        return (upperFive5 << 3) + upperThree4;
    }

    // 26 ops for 64 bytes
    static INLINE PtrR Unpack64(const VecU16& p_Five5Three4, const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 mask3(0x07);
        const VecU16 mask7(0x7F);
        const VecU16 kFFFC(0xFFFC);
        VecU16 packed0 = LoadA<VecU16>(p_Packed + 0 * kVectorSize);
        VecU16 packed1 = LoadA<VecU16>(p_Packed + 1 * kVectorSize);
        VecU16 packed2 = LoadA<VecU16>(p_Packed + 2 * kVectorSize);

        {
            const VecU8 bytes10 = ::Pack(packed0 & mask7, packed1 & mask7);
            StoreA(bytes10, p_Bytes + 0 * kVectorSize);
            packed0 >>= 7; packed1 >>= 7;
        }

        {
            const VecU8 bytes32 = ::Pack(packed0 & mask7, packed1 & mask7);
            StoreA(bytes32, p_Bytes + 1 * kVectorSize);
            packed0 >>= 7; packed1 >>= 7;
        }

        {
            const VecU16 bytes6 = packed2 & mask7;
            packed2 >>= 7;
            const VecU16 bytes7 = packed2 & mask7;
            packed2 >>= 7;
            const VecU8 bytes76 = ::Pack(bytes6, bytes7);
            StoreA(bytes76, p_Bytes + 3 * kVectorSize);
        }

        {
            const VecU16 bytes4 = ((p_Five5Three4 & mask3) << 4) + (packed2 << 2) + packed0;
            const VecU16 bytes5 = ((p_Five5Three4 >> 1) & kFFFC) + packed1;
            const VecU8 bytes54 = ::Pack(bytes4, bytes5);
            StoreA(bytes54, p_Bytes + 2 * kVectorSize);
            packed0 >>= 7; packed1 >>= 7;
        }

        return p_Packed + 3 * kVectorSize;
    }

    // 52 ops for 128 bytes
    static INLINE void Pack(const PtrR p_Bytes, const PtrW p_Packed)
    {
        const VecU16 upperFive5Three4L = Pack64(p_Bytes +  0, p_Packed + 0 * kVectorSize);
        const VecU16 upperFive5Three4H = Pack64(p_Bytes + 64, p_Packed + 3 * kVectorSize);
        const VecU8 upperFive5Three4HL = ::Pack(upperFive5Three4L, upperFive5Three4H);
        Stream(upperFive5Three4HL, p_Packed + 6 * kVectorSize);
    }

    // 41 ops for 128 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU8 upperFive5Three4HL = LoadA<VecU8>(p_Packed + 6 * kVectorSize);

        const VecU16 upperFive5Three4L = U16FromU8<0>(upperFive5Three4HL);
        Unpack64(upperFive5Three4L, p_Packed + 0 * kVectorSize, p_Bytes + 0);

        const VecU16 upperFive5Three4H(UnpackHigh(upperFive5Three4HL, VecU8(_mm_setzero_si128())));
        Unpack64(upperFive5Three4H, p_Packed + 3 * kVectorSize, p_Bytes + 64);
    }
};

//-----------------------------------------------------------------------------
// 9

template<> struct BlockTraits<9>
{
    static const size_t numCells = 16 * kVectorSize / sizeof(Uint16);
    static const size_t packedSize = 9 * kVectorSize;

    // 8 ops for 64 bytes
    static INLINE VecU16 Pack64(const PtrR p_Words, const PtrW p_Packed)
    {
        // load 32 values, each stored in a Uint16 => 288 useful bits =>
        // 256 bits = 2 vectors + 32 remainder bits (4 in 8 lanes)
        const VecU16 words0 = LoadA<VecU16>(p_Words + 0 * kVectorSize);
        const VecU16 words1 = LoadA<VecU16>(p_Words + 1 * kVectorSize);
        const VecU16 words2 = LoadA<VecU16>(p_Words + 2 * kVectorSize);
        const VecU16 words3 = LoadA<VecU16>(p_Words + 3 * kVectorSize);

        VecU16 upperTwoBits2;
        {
            const VecU16 packedL = (words2 << 9) + words0;
            Stream(packedL, p_Packed + 0 * kVectorSize);
            upperTwoBits2 = words2 >> 7;
        }

        VecU16 upperTwoBits3;
        {
            const VecU16 packedH = (words3 << 9) + words1;
            Stream(packedH, p_Packed + 1 * kVectorSize);
            upperTwoBits3 = words3 >> 7;
        }

        const VecU16 upperTwoBits32 = (upperTwoBits3 << 2) + upperTwoBits2;
        return upperTwoBits32;
    }

    // 9 ops for 64 bytes
    static INLINE PtrR Unpack64(const VecU16& p_UpperTwoBits32, const PtrR p_Packed, const PtrW p_Words)
    {
        const VecU16 mask9((1 << 9) - 1);
        const VecU16 maskUpper2(3 << 7);
        VecU16 packedL = LoadA<VecU16>(p_Packed + 0 * kVectorSize);
        VecU16 packedH = LoadA<VecU16>(p_Packed + 1 * kVectorSize);

        {
            const VecU16 words0 = packedL & mask9;
            const VecU16 words1 = packedH & mask9;
            StoreA(words0, p_Words + 0 * kVectorSize);
            StoreA(words1, p_Words + 1 * kVectorSize);
            packedL >>= 9; packedH >>= 9;
        }

        {
            // (argument is already shifted)
            const VecU16 upperTwoBits2 = p_UpperTwoBits32 & maskUpper2;
            const VecU16 upperTwoBits3 = (p_UpperTwoBits32 >> 2) & maskUpper2;
            const VecU16 words2 = packedL + upperTwoBits2;
            const VecU16 words3 = packedH + upperTwoBits3;
            StoreA(words2, p_Words + 2 * kVectorSize);
            StoreA(words3, p_Words + 3 * kVectorSize);
        }

        return p_Packed + 2 * kVectorSize;
    }

    // 38 ops for 256 bytes
    static INLINE void Pack(const PtrR p_Words, const PtrW p_Packed)
    {
        const VecU16 upperTwoBits0 = Pack64(p_Words + 0 * 64, p_Packed + 0 * kVectorSize);
        const VecU16 upperTwoBits1 = Pack64(p_Words + 1 * 64, p_Packed + 2 * kVectorSize);
        const VecU16 upperTwoBits2 = Pack64(p_Words + 2 * 64, p_Packed + 4 * kVectorSize);
        const VecU16 upperTwoBits3 = Pack64(p_Words + 3 * 64, p_Packed + 6 * kVectorSize);
        const VecU16 upperTwoBits = (upperTwoBits3 << 12) + (upperTwoBits2 << 8) +
            (upperTwoBits1 << 4) + upperTwoBits0;

        Stream(upperTwoBits, p_Packed + 8 * kVectorSize);
    }

    // 40 ops for 256 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 upperTwoBits = LoadA<VecU16>(p_Packed + 8 * kVectorSize);

        Unpack64(upperTwoBits << 7, p_Packed + 0 * kVectorSize, p_Bytes + 0 * 64);
        Unpack64(upperTwoBits << 3, p_Packed + 2 * kVectorSize, p_Bytes + 1 * 64);
        Unpack64(upperTwoBits >> 1, p_Packed + 4 * kVectorSize, p_Bytes + 2 * 64);
        Unpack64(upperTwoBits >> 5, p_Packed + 6 * kVectorSize, p_Bytes + 3 * 64);
    }
};

//-----------------------------------------------------------------------------
// 10

template<> struct BlockTraits<10>
{
    static const size_t numCells = 8 * kVectorSize / sizeof(Uint16);
    static const size_t packedSize = 5 * kVectorSize;

    // 3 ops for 32 bytes
    static INLINE VecU16 Pack32(const PtrR p_Words, const PtrW p_Packed)
    {
        // load 16 U16 values => 160 useful bits =>
        // 1 vector + 32 remainder bits (4 in 8 lanes)
        const VecU16 words0 = LoadA<VecU16>(p_Words + 0 * kVectorSize);
        const VecU16 words1 = LoadA<VecU16>(p_Words + 1 * kVectorSize);

        const VecU16 packedL = (words1 << 10) + words0;
        Stream(packedL, p_Packed);
        const VecU16 upperFourBits1 = words1 >> 6;
        return upperFourBits1;
    }

    // 4 ops for 32 bytes
    static INLINE PtrR Unpack32(const VecU16& p_UpperFourBits1, const PtrR p_Packed, const PtrW p_Words)
    {
        const VecU16 mask10((1 << 10) - 1);
        const VecU16 maskUpperFour(0xF << 6);
        const VecU16 packed = LoadA<VecU16>(p_Packed + 0 * kVectorSize);

        const VecU16 words0 = packed & mask10;
        // (p_UpperFourBits1 is already shifted)
        const VecU16 words1 = (p_UpperFourBits1 & maskUpperFour) + (packed >> 10);
        StoreA(words0, p_Words + 0 * kVectorSize);
        StoreA(words1, p_Words + 1 * kVectorSize);

        return p_Packed + kVectorSize;
    }

    // 18 ops for 128 bytes
    static INLINE void Pack(const PtrR p_Words, const PtrW p_Packed)
    {
        const VecU16 upperFourBits1 = Pack32(p_Words + 0 * 32, p_Packed + 0 * kVectorSize);
        const VecU16 upperFourBits3 = Pack32(p_Words + 1 * 32, p_Packed + 1 * kVectorSize);
        const VecU16 upperFourBits5 = Pack32(p_Words + 2 * 32, p_Packed + 2 * kVectorSize);
        const VecU16 upperFourBits7 = Pack32(p_Words + 3 * 32, p_Packed + 3 * kVectorSize);
        const VecU16 upperFourBits = (upperFourBits7 << 12) + (upperFourBits5 << 8) +
            (upperFourBits3 << 4) + upperFourBits1;

        Stream(upperFourBits, p_Packed + 4 * kVectorSize);
    }

    // 20 ops for 128 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 upperFourBits = LoadA<VecU16>(p_Packed + 4 * kVectorSize);

        Unpack32(upperFourBits << 6, p_Packed + 0 * kVectorSize, p_Bytes + 0 * 32);
        Unpack32(upperFourBits << 2, p_Packed + 1 * kVectorSize, p_Bytes + 1 * 32);
        Unpack32(upperFourBits >> 2, p_Packed + 2 * kVectorSize, p_Bytes + 2 * 32);
        Unpack32(upperFourBits >> 6, p_Packed + 3 * kVectorSize, p_Bytes + 3 * 32);
    }
};

//-----------------------------------------------------------------------------
// 11

template<> struct BlockTraits<11>
{
    static const size_t numCells = 16 * kVectorSize / sizeof(Uint16);
    static const size_t packedSize = 11 * kVectorSize;

    // 8 ops for 64 bytes
    static INLINE VecU16 Pack64(const PtrR p_Words, const PtrW p_Packed)
    {
        // load 32 values, each stored in a Uint16 => 352 useful bits =>
        // 2 vectors + 96 remainder bits (12 in 8 lanes).
        const VecU16 words0 = LoadA<VecU16>(p_Words + 0 * kVectorSize);
        const VecU16 words1 = LoadA<VecU16>(p_Words + 1 * kVectorSize);
        const VecU16 words2 = LoadA<VecU16>(p_Words + 2 * kVectorSize);
        const VecU16 words3 = LoadA<VecU16>(p_Words + 3 * kVectorSize);

        VecU16 upperSixBits2;
        {
            const VecU16 packedL = (words2 << 11) + words0;
            Stream(packedL, p_Packed + 0 * kVectorSize);
            upperSixBits2 = words2 >> 5;
        }

        VecU16 upperSixBits3;
        {
            const VecU16 packedH = (words3 << 11) + words1;
            Stream(packedH, p_Packed + 1 * kVectorSize);
            upperSixBits3 = words3 >> 5;
        }

        const VecU16 upperSixBits32 = (upperSixBits3 << 6) + upperSixBits2;
        return upperSixBits32;
    }

    // 11 ops for 64 bytes
    static INLINE PtrR Unpack64(const VecU16& p_UpperSixBits32, const PtrR p_Packed, const PtrW p_Words)
    {
        const VecU16 mask6((1 << 6) - 1);
        const VecU16 mask11((1 << 11) - 1);
        VecU16 packedL = LoadA<VecU16>(p_Packed + 0 * kVectorSize);
        VecU16 packedH = LoadA<VecU16>(p_Packed + 1 * kVectorSize);

        {
            const VecU16 words0 = packedL & mask11;
            const VecU16 words1 = packedH & mask11;
            StoreA(words0, p_Words + 0 * kVectorSize);
            StoreA(words1, p_Words + 1 * kVectorSize);
            packedL >>= 11; packedH >>= 11;
        }

        {
            // (p_UpperSixBits32 is 12 bits, so we cannot shift by 5 beforehand.)
            const VecU16 upperSixBits2 = p_UpperSixBits32 & mask6;
            const VecU16 upperSixBits3 = (p_UpperSixBits32 >> 6) & mask6;
            const VecU16 words2 = (upperSixBits2 << 5) + packedL;
            const VecU16 words3 = (upperSixBits3 << 5) + packedH;
            StoreA(words2, p_Words + 2 * kVectorSize);
            StoreA(words3, p_Words + 3 * kVectorSize);
        }

        return p_Packed + 2 * kVectorSize;
    }

    // 47 ops for 256 bytes
    static INLINE void Pack(const PtrR p_Words, const PtrW p_Packed)
    {
        const VecU16 upperSixBits32 = Pack64(p_Words + 0 * 64, p_Packed + 0 * kVectorSize);
        const VecU16 upperSixBits76 = Pack64(p_Words + 1 * 64, p_Packed + 2 * kVectorSize);
        const VecU16 upperSixBitsBA = Pack64(p_Words + 2 * 64, p_Packed + 4 * kVectorSize);
        const VecU16 upperSixBitsFE = Pack64(p_Words + 3 * 64, p_Packed + 6 * kVectorSize);
        const VecU16 four6Six32 = (upperSixBits76 << 12) + upperSixBits32;
        const VecU16 fourESixBA = (upperSixBitsFE << 12) + upperSixBitsBA;
        const VecU16 eightF7(::Pack(upperSixBits76 >> 4, upperSixBitsFE >> 4));

        Stream(four6Six32, p_Packed + 8 * kVectorSize);
        Stream(fourESixBA, p_Packed + 9 * kVectorSize);
        Stream(eightF7, p_Packed + 10 * kVectorSize);
    }

    // 52 ops for 256 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 four6Six32 = LoadA<VecU16>(p_Packed + 8 * kVectorSize);
        const VecU16 fourESixBA = LoadA<VecU16>(p_Packed + 9 * kVectorSize);
        const VecU16 eightF7 = LoadA<VecU16>(p_Packed + 10 * kVectorSize);

        const VecU16 zero = four6Six32 - four6Six32;
        const VecU16 eight7 = U16FromU8<0>(VecU8(eightF7));
        const VecU16 eightF(_mm_unpackhi_epi8(eightF7, zero));
        const VecU16 upperSixBits32 = four6Six32;  // (upper four will be masked away)
        const VecU16 upperSixBits76 = (eight7 << 4) + (four6Six32 >> 12);
        const VecU16 upperSixBitsBA = fourESixBA;
        const VecU16 upperSixBitsFE = (eightF << 4) + (fourESixBA >> 12);
        Unpack64(upperSixBits32, p_Packed + 0 * kVectorSize, p_Bytes + 0 * 64);
        Unpack64(upperSixBits76, p_Packed + 2 * kVectorSize, p_Bytes + 1 * 64);
        Unpack64(upperSixBitsBA, p_Packed + 4 * kVectorSize, p_Bytes + 2 * 64);
        Unpack64(upperSixBitsFE, p_Packed + 6 * kVectorSize, p_Bytes + 3 * 64);
    }
};

//-----------------------------------------------------------------------------
// 12

template<> struct BlockTraits<12>
{
    static const size_t numCells = 4 * kVectorSize / sizeof(Uint16);
    static const size_t packedSize = 3 * kVectorSize;

    // 3 ops for 32 bytes
    static INLINE VecU16 Pack32(const PtrR p_Words, const PtrW p_Packed)
    {
        // load 16 U16 values => 192 useful bits =>
        // 1 vector + 64 remainder bits (8 in 8 lanes)
        const VecU16 words0 = LoadA<VecU16>(p_Words + 0 * kVectorSize);
        const VecU16 words1 = LoadA<VecU16>(p_Words + 1 * kVectorSize);

        const VecU16 packedL = (words1 << 12) + words0;
        Stream(packedL, p_Packed);
        const VecU16 upperEightBits1 = words1 >> 4;
        return upperEightBits1;
    }

    // 4 ops for 32 bytes
    static INLINE PtrR Unpack32(const VecU16& p_UpperEightBits1, const PtrR p_Packed, const PtrW p_Words)
    {
        const VecU16 mask12((1 << 12) - 1);
        const VecU16 packed = LoadA<VecU16>(p_Packed);

        const VecU16 words0 = packed & mask12;
        const VecU16 words1 = (p_UpperEightBits1 << 4) + (packed >> 12);
        StoreA(words0, p_Words + 0 * kVectorSize);
        StoreA(words1, p_Words + 1 * kVectorSize);

        return p_Packed + kVectorSize;
    }

    // 7 ops for 64 bytes
    static INLINE void Pack(const PtrR p_Words, const PtrW p_Packed)
    {
        const VecU16 upperEightBits1 = Pack32(p_Words + 0 * 32, p_Packed + 0 * kVectorSize);
        const VecU16 upperEightBits3 = Pack32(p_Words + 1 * 32, p_Packed + 1 * kVectorSize);
        const VecU8 upperEightBits = ::Pack(upperEightBits1, upperEightBits3);

        Stream(upperEightBits, p_Packed + 2 * kVectorSize);
    }

    // 10 ops for 64 bytes
    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const VecU16 upperEightBits = LoadA<VecU16>(p_Packed + 2 * kVectorSize);
        const VecU16 zero = upperEightBits - upperEightBits;
        const VecU16 upperEightBits1 = U16FromU8<0>(VecU8(upperEightBits));
        const VecU16 upperEightBits3(_mm_unpackhi_epi8(upperEightBits, zero));

        Unpack32(upperEightBits1, p_Packed + 0 * kVectorSize, p_Bytes + 0 * 32);
        Unpack32(upperEightBits3, p_Packed + 1 * kVectorSize, p_Bytes + 1 * 32);
    }
};

//-----------------------------------------------------------------------------
// 13-15 (tiles are so sparse that full blocks are unlikely)

struct BlockTraitsDisabled
{
    // power of two for fast division/modulo, but so large that it
    // is never exceeded => Truncate will return 0.
    static const size_t numCells = 1ull << 32;
    static const size_t packedSize = kVectorSize;

    static INLINE void Pack(const PtrR p_Words, const PtrW p_Packed)
    {
        ASSERT(false);
    }

    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        ASSERT(false);
    }
};

template<> struct BlockTraits<13> : public BlockTraitsDisabled {};
template<> struct BlockTraits<14> : public BlockTraitsDisabled {};
template<> struct BlockTraits<15> : public BlockTraitsDisabled {};

//-----------------------------------------------------------------------------
// 16

template<> struct BlockTraits<16>
{
    static const size_t numCells = 1 * kVectorSize / sizeof(Uint16);
    static const size_t packedSize = 1 * kVectorSize;

    static INLINE void Pack(const PtrR p_Words, const PtrW p_Packed)
    {
        const auto words = LoadA<VecU16>(p_Words);
        Stream(words, p_Packed);
    }

    static INLINE void Unpack(const PtrR p_Packed, const PtrW p_Bytes)
    {
        const auto packed = LoadA<VecU16>(p_Packed);
        Stream(packed, p_Bytes);
    }
};

//-----------------------------------------------------------------------------

template<size_t t_NumBits> struct Blocks
{
    static const size_t cellSize = ROUND_UP(t_NumBits, CHAR_BIT) / CHAR_BIT;
    static const size_t cellsPerBlock = BlockTraits<t_NumBits>::numCells;
    static const size_t packedBlockSize = BlockTraits<t_NumBits>::packedSize;

    // @return how many cells can be processed as whole blocks.
    // (the rest are `remainders').
    static INLINE size_t Truncate(size_t p_NumCells)
    {
        return ROUND_DOWN(p_NumCells, cellsPerBlock);
    }

    // @return how many BYTES will be required.
    // (to avoid extra copying, parallel encoders reserve storage
    // once their encodedSize is known, and write there directly)
    static INLINE size_t PackedSize(size_t p_NumCells)
    {
        ASSERT(p_NumCells % cellsPerBlock == 0);
        return p_NumCells / cellsPerBlock * packedBlockSize;
    }

    // @param p_Packed (multiple of kVectorSize)
    // @return packedPos (multiple of kVectorSize)
    static INLINE PtrW Pack(const PtrR p_Cells, size_t p_NumCells, const PtrW p_Packed)
    {
        ASSERT((p_NumCells % cellsPerBlock) == 0);

        PtrW packedPos = p_Packed;
        for (size_t i = 0; i < p_NumCells; i += cellsPerBlock)
        {
            const PtrR in = p_Cells + i * cellSize;
            BlockTraits<t_NumBits>::Pack(in, packedPos);
            packedPos += packedBlockSize;
        }

        return packedPos;
    }

    static INLINE PtrR Unpack(const PtrR p_Packed, size_t p_NumCells, const PtrW p_Cells)
    {
        ASSERT((p_NumCells % cellsPerBlock) == 0);

        PtrR packedPos = p_Packed;
        for (size_t i = 0; i < p_NumCells; i += cellsPerBlock)
        {
            const PtrW out = p_Cells + i * cellSize;
            BlockTraits<t_NumBits>::Unpack(packedPos, out);
            packedPos += packedBlockSize;
        }

        return packedPos;
    }
};

void TestPack();

}  // namespace codec
