#pragma once

#include "Indices.h"
#include "RemoveZeros.h"
#include "Pack.h"

namespace codec {

// special case for two 1-bits out of 8. a straightforward representation
// involves two 3-bit positions, but there are only 28 valid combinations.
// enumerative coding associates each possible outcome with a 5-bit count.
// 4 values remain unused; we choose the interval [0,4) so that zero inputs
// (for non-enum5 cases) decode to zero without any extra masking.
//
// this same logic is also used in U4 for pairs of nibbles.

// `half' means either nibble is { 3,5,6,9,A,C } and the other nibble is zero.
struct TwoBitsHalf
{
    // 3 ops
    // @param p_ExclusiveL/H [0, 16), 0 if other nibble is nonzero
    // @return [4, 16) or 0
    static INLINE VecU8 Encode(const VecU8& p_ExclusiveL, const VecU8& p_ExclusiveH)
    {
        ASSERT(p_ExclusiveL[0] < 16);
        ASSERT(p_ExclusiveH[0] < 16);
        const VecU8 tableL(0, 0, 0,  9, 0,  8,  7, 0, 0,  6,  5, 0,  4, 0, 0, 0);
        const VecU8 tableH(0, 0, 0, 15, 0, 14, 13, 0, 0, 12, 11, 0, 10, 0, 0, 0);
        const VecU8 enum5L = Shuffle(tableL, p_ExclusiveL);
        const VecU8 enum5H = Shuffle(tableH, p_ExclusiveH);
        // (L/H are both zero, or exactly one is valid and nonzero => sum is valid or 0)
        return enum5L + enum5H;
    }

    // 1 op
    // @param p_Enum5L [4, 16) or 0 if zero, single or raw8
    // @return decoded 8-bit value with two 1-bits, or 0 if p_Enum5 == 0.
    static INLINE VecU8 Decode(const VecU8& p_Enum5L)
    {
        const VecU8 table(0xC0, 0xA0, 0x90, 0x60, 0x50, 0x30, 0x0C, 0x0A, 9, 6, 5, 3, 0, 0, 0, 0);
        return Shuffle(table, p_Enum5L);
    }

    static void Test()
    {
        const Uint8 values[12] = { 3, 5, 6, 9, 0xA, 0xC, 0x30, 0x50, 0x60, 0x90, 0xA0, 0xC0 };
        for (size_t val = 0; val < 256; ++val)
        {
            const size_t nibbleL = val & 0xF;
            const size_t nibbleH = val >> 4;
            const size_t exclusiveL = (nibbleH == 0) ? nibbleL : 0;
            const size_t exclusiveH = (nibbleL == 0) ? nibbleH : 0;
            const VecU8 encoded = Encode(VecU8(Uint8(exclusiveL)), VecU8(Uint8(exclusiveH)));
            const VecU8 decoded = Decode(encoded);

            const Uint8* pos = std::find(values, values + 12, val);
            if (pos == values + 12)  // not one of the values we can represent
            {
                ENSURE(encoded[0] == 0);
                ENSURE(decoded[0] == 0);
            }
            else  // encoded as index
            {
                ENSURE(encoded[0] == 4 + pos - values);
                ENSURE(decoded[0] == val);
            }
        }
    }
};

// `both' means exactly one bit in each nibble is set.
struct TwoBitsBoth
{
    // 5 ops
    // @param p_NibbleL/H [0, 16)
    // @return [16, 32) or p_EnumL
    static INLINE VecU8 Encode(const VecU8& p_NibbleL, const VecU8& p_NibbleH, const VecU8& p_EnumL)
    {
        ASSERT(p_NibbleL[0] < 16);
        ASSERT(p_NibbleH[0] < 16);

        // 1,2,4,8 in lower nibble -> bitIndex + 0x80
        const VecU8 tableL(0, 0, 0, 0, 0, 0, 0, 0x83, 0, 0, 0, 0x82, 0, 0x81, 0x80, 0);
        const VecU8 L = Shuffle(tableL, p_NibbleL);
        // 1,2,4,8 in upper nibble -> 4 * bitIndex + 0x90
        const VecU8 tableH(0, 0, 0, 0, 0, 0, 0, 0x9C, 0, 0, 0, 0x98, 0, 0x94, 0x90, 0);
        const VecU8 H4 = Shuffle(tableH, p_NibbleH);

        const VecU8 bothNonzero = L & H4;  // (only MSB valid)
        const VecU8 enum5H = H4 + L;  //  [0, 32) after overflow

        // return valid enum5, or p_EnumL (may also be zero).
        return Select(bothNonzero, enum5H, p_EnumL);
    }

    // 1 op
    // @param p_Enum5H [16, 32) or 0 for undefined result
    // @return decoded 8-bit value with two 1-bits
    static INLINE VecU8 Decode(const VecU8& p_Enum5H)
    {
        const VecU8 table(0x88, 0x84, 0x82, 0x81, 0x48, 0x44, 0x42, 0x41,
            0x28, 0x24, 0x22, 0x21, 0x18, 0x14, 0x12, 0x11);
        return Shuffle(table, p_Enum5H);
    }

    static void Test()
    {
        const Uint8 values[16] = { 0x11, 0x12, 0x14, 0x18, 0x21, 0x22, 0x24, 0x28,
            0x41, 0x42, 0x44, 0x48, 0x81, 0x82, 0x84, 0x88 };
        for (size_t val = 0; val < 256; ++val)
        {
            const Uint8 nibbleL = static_cast<Uint8>(val & 0xF);
            const Uint8 nibbleH = static_cast<Uint8>(val >> 4);
            const VecU8 encoded = Encode(VecU8(nibbleL), VecU8(nibbleH), VecU8(0xFE));
            const VecU8 decoded = Decode(encoded);

            const Uint8* pos = std::find(values, values + 16, val);
            if (pos == values + 16)  // not one of the values we can represent
            {
                ENSURE(encoded[0] == 0xFE);
                // decoded is undefined
            }
            else  // encoded as index
            {
                ENSURE(encoded[0] == 16 + pos - values);
                ENSURE(decoded[0] == val);
            }
        }
    }
};

//-----------------------------------------------------------------------------

struct U4 {};

// process pairs of nibbles so that we can reuse some of the U8 logic.
// encoding differs from U8 because all-zero is much less likely
// (only 10%, so a nonZero bit is wasteful). once again we take advantage
// of the relatively few "2 from 8" combinations and code them with
// 'fractional' bits by using 3 of the 16 possible selector values.
template<> struct Bundle<U4>
{
    // (different encoding than Bundle<U8>::SingleBit so that code4 can be computed as single >> 1)
    struct SingleBit
    {
        // @return 0 or [24, 32) if single bit
        // 3 ops
        static INLINE VecU8 Encode(const VecU8& p_ExclusiveL, const VecU8& p_ExclusiveH)
        {
            const VecU8 tableL(0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 26, 0, 25, 24, 0);
            const VecU8 tableH(0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 30, 0, 29, 28, 0);
            const VecU8 singleL = Shuffle(tableL, p_ExclusiveL);
            const VecU8 singleH = Shuffle(tableH, p_ExclusiveH);
            return singleL + singleH;
        }

        // 1 op
        // @param p_Index [24, 32) or 0
        // @return value of single bit or 0
        static INLINE VecU8 Decode(const VecU8& p_Index)
        {
            const VecU8 powerTable(0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
            return Shuffle(powerTable, p_Index);  // = 0 if zero, both, half, raw
        }

        static void Test()
        {
            const Uint8 values[8] = { 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80 };
            for (size_t val = 0; val < 256; ++val)
            {
                const size_t nibbleL = val & 0xF;
                const size_t nibbleH = val >> 4;
                const size_t exclusiveL = (nibbleH == 0) ? nibbleL : 0;
                const size_t exclusiveH = (nibbleL == 0) ? nibbleH : 0;
                const VecU8 encoded = Encode(VecU8(Uint8(exclusiveL)), VecU8(Uint8(exclusiveH)));
                const VecU8 decoded = Decode(encoded);

                const Uint8* pos = std::find(values, values + 8, val);
                if (pos == values + 8)  // not one of the values we can represent
                {
                    ENSURE(encoded[0] == 0);
                    ENSURE(decoded[0] == 0);
                }
                else  // encoded as index
                {
                    ENSURE(encoded[0] == 24 + pos - values);
                    ENSURE(decoded[0] == val);
                }
            }
        }
    };

    // 53 ops
    static INLINE void Encode(const PtrR p_Values,
        PtrW& p_Extra1Pos, PtrW& p_Extra2Pos, PtrW& p_Extra6Pos, const PtrW p_Code4Pos)
    {
        const VecU8 mask1(1);
        const VecU8 mask2(3);
        const VecU8 mask4(0x0F);
        const VecU8 mask6(0x3F);

        const auto bytes = LoadA<VecU8>(p_Values);

        // isolate nibbles (must mask before 16-bit shift!)
        const VecU8 maskedL = mask4 & bytes;
        const VecU8 maskedH = AndNot(mask4, bytes);
        const VecU8 isZeroH = (maskedL == bytes);
        const VecU8 isZeroL = (maskedH == bytes);
        const VecU8 nibbleH = VecU8(VecU16(maskedH) >> 4);

        const VecU8 exclusiveL = maskedL & isZeroH;
        const VecU8 exclusiveH = nibbleH & isZeroL;

        // 3 + 5 + 3 ops
        const VecU8 half = TwoBitsHalf::Encode(exclusiveL, exclusiveH);  // 4..15 or 0
        const VecU8 halfOrBoth = TwoBitsBoth::Encode(maskedL, nibbleH, half);  // both: 16..31 or 0
        const VecU8 single = SingleBit::Encode(exclusiveL, exclusiveH);  // 24..31 or 0

        const VecU8 isHalfOrBoth = NonZero(halfOrBoth);  // bit
        const VecU8 isSingle = NonZero(single);  // bit
        const VecU8 isRaw = AndNot((isHalfOrBoth | isSingle), NonZero(bytes));  // bit

        const VecU8 codeHalfOrBoth(VecU16(AndNot(mask2, halfOrBoth)) >> 2);  // 1..7 or 0
        const VecU8 codeSingle(VecU16(AndNot(mask1, single)) >> 1);  // 12..15 or 0
        const VecU8 codeHBS = codeHalfOrBoth + codeSingle;
        const VecU8 codeRaw = VecU8(8) + VecU8(VecU16(AndNot(mask6, bytes)) >> 6);  // 8..11
        const VecU8 code4 = Select(isRaw, codeRaw, codeHBS);
        StoreA(code4, p_Code4Pos);

        const VecU8 extra1 = single & mask1;
        const VecU8 extra2 = halfOrBoth & mask2;
        const VecU8 extra6 = bytes & mask6;

        p_Extra1Pos = RemoveAndStore(extra1, isSingle, p_Extra1Pos);
        p_Extra2Pos = RemoveAndStore(extra2, isHalfOrBoth, p_Extra2Pos);
        p_Extra6Pos = RemoveAndStore(extra6, isRaw, p_Extra6Pos);
    }

    // 39 ops
    static INLINE void Decode(PtrR& p_Extra1Pos, PtrR& p_Extra2Pos, PtrR& p_Extra6Pos,
        const PtrR p_Code4Pos, const PtrW p_Decoded)
    {
        const VecU8 k3(3);

        // zero: 0; half: 1..3; both: 4..7; raw: 8..11; single: 12..15
        const VecU8 code4 = LoadA<VecU8>(p_Code4Pos);

        // (only MSBs are valid! can only use with Select or LoadAndRestore.)
        const VecU8 isRawOrSingle(VecU16(code4) << 4);
        const VecU8 isHalfOrBoth = Select(isRawOrSingle, code4, NonZero(code4));  // code4 is 'zero' MSB.
        const VecU8 isBoth = isHalfOrBoth & VecU8(VecI8(code4) > VecI8(k3));
        const VecU8 isSingle = isRawOrSingle & (isRawOrSingle + isRawOrSingle);
        const VecU8 isRaw = AndNot(isSingle, isRawOrSingle);

        VecU8 extra1;
        p_Extra1Pos = LoadAndRestore00(p_Extra1Pos, isSingle, extra1);
        const VecU8 idxSingle = (code4 + code4) + extra1;  // single: 24..31; zero: 0

        VecU8 extra2;
        p_Extra2Pos = LoadAndRestore00(p_Extra2Pos, isHalfOrBoth, extra2);
        // half: 4..15; both: 16..31; zero: 0; single/raw: invalid
        const VecU8 idxHalfOrBoth = VecU8(VecU16(code4) << 2) + extra2;

        VecU8 extra6;
        p_Extra6Pos = LoadAndRestore00(p_Extra6Pos, isRaw, extra6);
        // zero: 0; half/both/single: invalid
        const VecU8 raw = VecU8(VecU16(code4 & k3) << 6) + extra6;

        // (both is invalid if code4 == 0, so only use it if isBoth.)
        const VecU8 half = TwoBitsHalf::Decode(idxHalfOrBoth);  // zero: 0
        const VecU8 both = TwoBitsBoth::Decode(idxHalfOrBoth);
        const VecU8 single = SingleBit::Decode(idxSingle);      // zero: 0
        const VecU8 rawOrSingle = Select(isRaw, raw, single);   // zero: 0
        const VecU8 halfOrBoth = Select(isBoth, both, half);    // zero: 0
        const VecU8 decoded = Select(isHalfOrBoth, halfOrBoth, rawOrSingle);

        (void)Stream(decoded, p_Decoded);
    }

    static void Test()
    {
        SingleBit::Test();
    }
};

// (don't need a Pair specialization because we don't load nonZeroBits from memory.
// code4, extra1, extra3 and extra6 are packed/unpacked into/from buffers.)

// @return false if size exceeds raw size
static INLINE bool EncodeTileShuffle4(const PtrR p_Values,
    const PtrW p_Buffers, PtrW& p_PosAligned, PtrW& p_PosAny)
{
    const PtrW buf1 = p_Buffers + 1 * kBufferSize;
    const PtrW buf2 = p_Buffers + 2 * kBufferSize;
    const PtrW buf3 = p_Buffers + 3 * kBufferSize;
    PtrW posExtra1 = buf1, posExtra2 = buf2, posExtra6 = buf3;

    // one code4 byte for pairs of 4-bit packets => same offset as input values
    for (size_t pos4 = 0; pos4 < kTileSize; pos4 += kVectorSize)
    {
        Bundle<U4>::Encode(p_Values + pos4, posExtra1, posExtra2, posExtra6, p_Buffers + pos4);
    }

    const size_t numExtra1 = posExtra1 - buf1;
    const size_t numExtra2 = posExtra2 - buf2;
    const size_t numExtra6 = posExtra6 - buf3;
    const size_t truncated4 = kTileSize;
    const size_t truncated1 = Blocks<1>::Truncate(numExtra1);
    const size_t truncated2 = Blocks<2>::Truncate(numExtra2);
    const size_t truncated6 = Blocks<6>::Truncate(numExtra6);
    // (remainders4 = 0)
    const size_t remainders1 = numExtra1 - truncated1;
    const size_t remainders2 = numExtra2 - truncated2;
    const size_t remainders6 = numExtra6 - truncated6;

    const size_t sizeAny = 6 +
        Remainders<1>::PackedSize(remainders1) +
        Remainders<2>::PackedSize(remainders2) +
        Remainders<6>::PackedSize(remainders6);
    const size_t sizeAligned =
        Blocks<4>::PackedSize(truncated4) +
        Blocks<1>::PackedSize(truncated1) +
        Blocks<2>::PackedSize(truncated2) +
        Blocks<6>::PackedSize(truncated6);
    ASSERT((numExtra1 | numExtra2 | numExtra6) != 0);
    if (sizeAligned + sizeAny >= kTileSize) return false;

    const Uint64 num621 = (numExtra6 << 32) + (numExtra2 << 16) + numExtra1;
    p_PosAny = StoreFrom(&num621, 6, p_PosAny);
    p_PosAny = Remainders<1>::Pack(RFromW(posExtra1) - remainders1, remainders1, p_PosAny);
    p_PosAny = Remainders<2>::Pack(RFromW(posExtra2) - remainders2, remainders2, p_PosAny);
    p_PosAny = Remainders<6>::Pack(RFromW(posExtra6) - remainders6, remainders6, p_PosAny);

    p_PosAligned = Blocks<4>::Pack(RFromW(p_Buffers), truncated4, p_PosAligned);
    p_PosAligned = Blocks<1>::Pack(RFromW(buf1), truncated1, p_PosAligned);
    p_PosAligned = Blocks<2>::Pack(RFromW(buf2), truncated2, p_PosAligned);
    p_PosAligned = Blocks<6>::Pack(RFromW(buf3), truncated6, p_PosAligned);
    return true;
}

static INLINE void DecodeTileShuffle4(
    PtrR& p_PosAligned, PtrR& p_PosAny, const PtrW p_Buffers, const PtrW p_Out)
{
    const Uint64 num621 = Load<Uint64>(p_PosAny); p_PosAny += 6;
    const size_t numExtra1 = num621 & 0xFFFF;
    const size_t numExtra2 = (num621 >> 16) & 0xFFFF;
    const size_t numExtra6 = (num621 >> 32) & 0xFFFF;
    const size_t truncated4 = kTileSize;
    const size_t truncated1 = Blocks<1>::Truncate(numExtra1);
    const size_t truncated2 = Blocks<2>::Truncate(numExtra2);
    const size_t truncated6 = Blocks<6>::Truncate(numExtra6);
    // (remainders4 = 0)
    const size_t remainders1 = numExtra1 - truncated1;
    const size_t remainders2 = numExtra2 - truncated2;
    const size_t remainders6 = numExtra6 - truncated6;

    const PtrW buf1 = p_Buffers + 1 * kBufferSize;
    const PtrW buf2 = p_Buffers + 2 * kBufferSize;
    const PtrW buf3 = p_Buffers + 3 * kBufferSize;
    p_PosAligned = Blocks<4>::Unpack(p_PosAligned, truncated4, p_Buffers);
    p_PosAligned = Blocks<1>::Unpack(p_PosAligned, truncated1, buf1);
    p_PosAligned = Blocks<2>::Unpack(p_PosAligned, truncated2, buf2);
    p_PosAligned = Blocks<6>::Unpack(p_PosAligned, truncated6, buf3);

    p_PosAny = Remainders<1>::Unpack(p_PosAny, remainders1, buf1 + truncated1);
    p_PosAny = Remainders<2>::Unpack(p_PosAny, remainders2, buf2 + truncated2);
    p_PosAny = Remainders<6>::Unpack(p_PosAny, remainders6, buf3 + truncated6);

    PtrR posExtra1 = RFromW(buf1), posExtra2 = RFromW(buf2), posExtra6 = RFromW(buf3);
    for (size_t pos4 = 0; pos4 < kTileSize; pos4 += kVectorSize)
    {
        Bundle<U4>::Decode(posExtra1, posExtra2, posExtra6, RFromW(p_Buffers) + pos4, p_Out + pos4);
    }
}

//-----------------------------------------------------------------------------

// for randomly generated bit planes, only 81% of T=U8 packets have zero or
// one bits set, and 9% have 3 or more. for reasons of both size and speed,
// we need to handle these dense cases by encoding the Uint8 value "raw".
//
// # code4 extra2 raw
// 0:  0     0     0
// 1: 8-F    0     0
// 2: 1-7   0-3    0
// R:  0     0    0-FF
template<> struct Bundle<Uint8>
{
    struct SingleBit
    {
        // @return [8,16) if single bit, otherwise 0
        // 3 ops
        static INLINE VecU8 Encode(const VecU8& p_ExclusiveL, const VecU8& p_ExclusiveH)
        {
            const VecU8 tableL(0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 10, 0, 9, 8, 0);
            const VecU8 tableH(0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 14, 0, 13, 12, 0);
            const VecU8 singleL = Shuffle(tableL, p_ExclusiveL);
            const VecU8 singleH = Shuffle(tableH, p_ExclusiveH);
            return singleL + singleH;
        }

        // 1 op
        // @param p_Index [0, 16)
        // @return if p_Index = [8,16), return value of single bit, otherwise 0
        static INLINE VecU8 Decode(const VecU8& p_Index)
        {
            const VecU8 powerTable(0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
            return Shuffle(powerTable, p_Index);  // = 0 if zero, enum5 or raw8
        }

        static void Test()
        {
            const Uint8 values[8] = { 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80 };
            for (size_t val = 0; val < 256; ++val)
            {
                const size_t nibbleL = val & 0xF;
                const size_t nibbleH = val >> 4;
                const size_t exclusiveL = (nibbleH == 0) ? nibbleL : 0;
                const size_t exclusiveH = (nibbleL == 0) ? nibbleH : 0;
                const VecU8 encoded = Encode(VecU8(Uint8(exclusiveL)), VecU8(Uint8(exclusiveH)));
                const VecU8 decoded = Decode(encoded);

                const Uint8* pos = std::find(values, values + 8, val);
                if (pos == values + 8)  // not one of the values we can represent
                {
                    ENSURE(encoded[0] == 0);
                    ENSURE(decoded[0] == 0);
                }
                else  // encoded as index
                {
                    ENSURE(encoded[0] == 8 + pos - values);
                    ENSURE(decoded[0] == val);
                }
            }
        }
    };

    // 21 ops
    static INLINE void ComputeEnum5AndCode4(const VecU8& p_Values, VecU8& p_Enum5, VecU8& p_Code4)
    {
        // isolate nibbles (must mask before 16-bit shift!)
        const VecU8 mask0F(0x0F);
        const VecU8 maskedL = mask0F & p_Values;
        const VecU8 maskedH = AndNot(mask0F, p_Values);
        const VecU8 isZeroH = (maskedL == p_Values);
        const VecU8 isZeroL = (maskedH == p_Values);
        const VecU8 nibbleH = VecU8(VecU16(maskedH) >> 4);

        const VecU8 exclusiveL = maskedL & isZeroH;
        const VecU8 exclusiveH = nibbleH & isZeroL;
        const VecU8 enum5L = TwoBitsHalf::Encode(exclusiveL, exclusiveH);
        p_Enum5 = TwoBitsBoth::Encode(maskedL, nibbleH, enum5L);
        const VecU8 single = SingleBit::Encode(exclusiveL, exclusiveH);

        p_Code4 = (VecU8(VecU16(p_Enum5) >> 2) & mask0F) + single;
    }

    // 6 ops
    static INLINE VecU8 ValueFromEnum5AndCode4(const VecU8& p_Enum5, const VecU8& p_Code4,
        const VecU8& p_IsTwoBit)
    {
        const VecU8 decodedHalf = TwoBitsHalf::Decode(p_Enum5);  // = 0 if zero or code4 or raw8
        const VecU8 decodedBoth = TwoBitsBoth::Decode(p_Enum5);
        const VecU8 isBoth(VecI8(p_Enum5) > VecI8(0x0F));
        const VecU8 twoBit = Select(isBoth, decodedBoth, decodedHalf);
        const VecU8 oneBit = SingleBit::Decode(p_Code4);  // = 0 if zero, enum5 or raw8
        return Select(p_IsTwoBit, twoBit, oneBit);
    }

    // 43 ops (previous 12R: 59)
    // @return nonZeroBits
    static INLINE size_t Encode(const PtrR p_Values,
        PtrW& p_Code4Pos, PtrW& p_Extra2Pos, PtrW& p_Raw8Pos)
    {
        const auto values = LoadA<VecU8>(p_Values);
        const size_t nonZeroBits = _mm_movemask_epi8(NonZero(values));

        VecU8 enum5, code4;
        ComputeEnum5AndCode4(values, enum5, code4);

        // 20 ops
        p_Code4Pos = RemoveAndStore(code4, nonZeroBits, p_Code4Pos);
        const size_t extra2Bits = _mm_movemask_epi8(NonZero(enum5));
        const VecU8 extra2 = enum5 & VecU8(3);
        p_Extra2Pos = RemoveAndStore(extra2, extra2Bits, p_Extra2Pos);
        const size_t raw8Bits = Raw8Bits(nonZeroBits, code4);
        p_Raw8Pos = RemoveAndStore(values, raw8Bits, p_Raw8Pos);

        return nonZeroBits;
    }

    // 2 ops
    static INLINE size_t Raw8Bits(size_t p_NonZeroBits, const VecU8& p_Code4)
    {
        const VecU8 zero(_mm_setzero_si128());
        return p_NonZeroBits & _mm_movemask_epi8(p_Code4 == zero);
    }

    // 29 ops (previous 12R: 40)
    static INLINE void Decode(size_t p_NonZeroBits,
        PtrR& p_Code4Pos, PtrR& p_Extra2Pos, PtrR& p_Raw8Pos, const PtrW p_Decoded)
    {
        VecU8 code4;  // = 0 if zero or raw8
        p_Code4Pos = LoadAndRestore00(p_Code4Pos, p_NonZeroBits, code4);

        VecU8 extra2;  // = 0 if zero or code4 or raw8
        const VecU8 extra2Table(0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0);
        const VecU8 extra2Mask = Shuffle(extra2Table, code4);
        p_Extra2Pos = LoadAndRestore00(p_Extra2Pos, extra2Mask, extra2);
        const VecU8 enum5 = VecU8(VecU16(code4 & extra2Mask) << 2) + extra2; // = 0 if zero, code4 or raw8

        VecU8 decoded;  // = 0 if zero or code4 or enum5
        const size_t raw8Bits = Raw8Bits(p_NonZeroBits, code4);
        p_Raw8Pos = LoadAndRestore00(p_Raw8Pos, raw8Bits, decoded);

        decoded += ValueFromEnum5AndCode4(enum5, code4, extra2Mask);

        (void)Stream(decoded, p_Decoded);
    }

    static void Test()
    {
        TwoBitsHalf::Test();
        TwoBitsBoth::Test();
        SingleBit::Test();

        {
            VecU8 enum5(_mm_setzero_si128()), code4(_mm_setzero_si128()), extra2Mask(_mm_setzero_si128());
            const VecU8 decoded = ValueFromEnum5AndCode4(enum5, code4, extra2Mask);
            ENSURE(decoded[0] == 0);
        }

        for (size_t val = 0; val < 256; ++val)
        {
            VecU8 enum5, code4;
            ComputeEnum5AndCode4(VecU8(Uint8(val)), enum5, code4);

            const size_t bits = PopulationCount64(val);
            if (bits != 1 && bits != 2)
            {
                ENSURE(enum5[0] == 0);
                ENSURE(code4[0] == 0);
            }
            else
            {
                const VecU8 extra2Table(0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0);
                const VecU8 extra2Mask = Shuffle(extra2Table, code4);
                const VecU8 decoded = ValueFromEnum5AndCode4(enum5, code4, extra2Mask);
                ENSURE(decoded[0] == val);
            }
        }
    }
};

// (still want to write 32 nonZeroBits at a time => use Pair interface)
template<> struct Pair<Uint8>
{
    static INLINE void Encode(const PtrR p_Values,
        PtrW& p_Pos4, PtrW& p_Pos2, PtrW& p_Pos8, const PtrW p_NonZeroPos)
    {
        const size_t nonZeroBitsL = Bundle<Uint8>::Encode(p_Values, p_Pos4, p_Pos2, p_Pos8);
        const size_t nonZeroBitsH = Bundle<Uint8>::Encode(p_Values + kVectorSize, p_Pos4, p_Pos2, p_Pos8);
        const size_t nonZeroBits = (nonZeroBitsH << 16) + nonZeroBitsL;
        (void)Store(static_cast<NonZeroBits>(nonZeroBits), p_NonZeroPos);
    }

    static INLINE void Decode(PtrR& p_Pos4, PtrR& p_Pos2, PtrR& p_Pos8, const PtrR p_NonZeroPos,
        const PtrW p_Decoded)
    {
        NonZeroBits nonZeroBits;
        (void)Load(p_NonZeroPos, nonZeroBits);
        const size_t nonZeroBitsL = nonZeroBits & 0xFFFF;
        const size_t nonZeroBitsH = nonZeroBits >> 16;

        Bundle<Uint8>::Decode(nonZeroBitsL, p_Pos4, p_Pos2, p_Pos8, p_Decoded);
        Bundle<Uint8>::Decode(nonZeroBitsH, p_Pos4, p_Pos2, p_Pos8, p_Decoded + kVectorSize);
    }
};

template<> INLINE void EncodeTileShuffle<Uint8>(const PtrR p_Values,
    const PtrW p_Buffers, PtrW& p_PosAligned, PtrW& p_PosAny)
{
    // (four 16-bit positions packed into 64 bits are compact but slow.
    // a reference to a mutable array of pointers is fairly fast,
    // but named parameters are easier to understand and slightly faster.)
    const PtrW buf1 = p_Buffers + 1 * kBufferSize;
    const PtrW buf2 = p_Buffers + 2 * kBufferSize;
    const PtrW buf3 = p_Buffers + 3 * kBufferSize;
    PtrW pos4 = buf1, pos2 = buf2;
    PtrW pos8 = buf3;

    const size_t sizeNonZero = kTileSize / CHAR_BIT;
    for (size_t ofs = 0; ofs < sizeNonZero; ofs += sizeof(NonZeroBits))
    {
        Pair<Uint8>::Encode(p_Values + ofs * CHAR_BIT, pos4, pos2, pos8, p_Buffers + ofs);
    }

    const size_t num4 = pos4 - buf1;
    const size_t num2 = pos2 - buf2;
    const size_t num8 = pos8 - buf3;

    const size_t truncated4 = Blocks<4>::Truncate(num4);
    const size_t truncated2 = Blocks<2>::Truncate(num2);
    const size_t remainders4 = num4 - truncated4;
    const size_t remainders2 = num2 - truncated2;

    const size_t sizeAny = 6 + num8 +
        Remainders<4>::PackedSize(remainders4) +
        Remainders<2>::PackedSize(remainders2);
    const size_t sizeAligned =
        Blocks<4>::PackedSize(truncated4) +
        Blocks<2>::PackedSize(truncated2);
    ASSERT((num4 + num2 + num8) != 0);
    ASSERT(sizeNonZero + sizeAny + sizeAligned < kTileSize);

    const Uint64 num248 = (num2 << 32) + (num4 << 16) + num8;
    p_PosAny = StoreFrom(&num248, 6, p_PosAny);
    // write contiguous raw8 to avoid copying in the decoder.
    // (unfortunately we cannot stream to p_PosAny due to misalignment)
    p_PosAny = StoreFrom(buf3, num8, p_PosAny);
    p_PosAny = Remainders<4>::Pack(RFromW(pos4) - remainders4, remainders4, p_PosAny);
    p_PosAny = Remainders<2>::Pack(RFromW(pos2) - remainders2, remainders2, p_PosAny);

    p_PosAligned = Stream64From(RFromW(p_Buffers), sizeNonZero, p_PosAligned);
    p_PosAligned = Blocks<4>::Pack(RFromW(buf1), truncated4, p_PosAligned);
    p_PosAligned = Blocks<2>::Pack(RFromW(buf2), truncated2, p_PosAligned);
}

template<> INLINE void DecodeTileShuffle<Uint8>(
    PtrR& p_PosAligned, PtrR& p_PosAny, const PtrW p_Buffers, const PtrW p_Out)
{
    const size_t sizeNonZero = kTileSize / CHAR_BIT;
    const Uint64 num248 = Load<Uint64>(p_PosAny); p_PosAny += 6;
    const size_t num8 = num248 & 0xFFFF;
    const size_t num4 = (num248 >> 16) & 0xFFFF;
    const size_t num2 = (num248 >> 32) & 0xFFFF;

    const size_t truncated4 = Blocks<4>::Truncate(num4);
    const size_t truncated2 = Blocks<2>::Truncate(num2);
    const size_t remainders4 = num4 - truncated4;
    const size_t remainders2 = num2 - truncated2;
    const size_t sizeAny = 6 + num8 +
        Remainders<4>::PackedSize(remainders4) +
        Remainders<2>::PackedSize(remainders2);

    const PtrR posNonZero = p_PosAligned; p_PosAligned += sizeNonZero;
    const PtrW buf1 = p_Buffers + 1 * kBufferSize;
    p_PosAligned = Blocks<4>::Unpack(p_PosAligned, truncated4, p_Buffers);
    p_PosAligned = Blocks<2>::Unpack(p_PosAligned, truncated2, buf1);

    PtrR pos8 = p_PosAny; p_PosAny += num8;
    p_PosAny = Remainders<4>::Unpack(p_PosAny, remainders4, p_Buffers + truncated4);
    p_PosAny = Remainders<2>::Unpack(p_PosAny, remainders2, buf1 + truncated2);

    PtrR pos4 = RFromW(p_Buffers);
    PtrR pos2 = RFromW(buf1);
    for (size_t ofs = 0; ofs < sizeNonZero; ofs += sizeof(NonZeroBits))
    {
        Pair<Uint8>::Decode(pos4, pos2, pos8, posNonZero + ofs, p_Out + ofs * CHAR_BIT);
    }
}

}  // namespace codec
