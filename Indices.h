#pragma once

#include "Bits.h"  // IndexOfLowest1
#include "Vector.h"

extern CACHE_ALIGNED(Uint64) g_Expand64[64 * VecU64::numLanes];

// @return (p_Values != 0) ? 0x80 : 0
// 1 op
static INLINE VecU8 NonZero(const VecU8& p_Values)
{
    // there is no integer != instruction, but the desired effect is obtained
    // via ssse3_sign_epi8(0x80, Uint8). if inputs are:
    // positive: 0x80 remains unchanged;
    // negative: -0x80 == 0x80 (undocumented, but the only non-destructive interpretation);
    // zero: 0.
    const VecU8 k80(0x80);
    const VecU8 notZero(ssse3_sign_epi8(k80, p_Values));

    // failed alternatives:
    // Abs(x) < 0 handles everything except 0x80 (see above).
    // string instructions have far too high latency.
    // saturating subtract 1 + comparison requires != or unsigned < (neither available).
    // Select + Shuffle also requires a constant.

    // viable alternative (2 ops, no loaded constant), neither faster nor slower:
    //const VecU8 zero(SetZero());
    //const VecU8 positive(VecI8(zero) < VecI8(p_Values));
    //const VecU8 notZero = positive ^ p_Values;

    return notZero;
}

// @return (p_Values != 0) ? 0x8000 : 0
// 1 op
static INLINE VecI16 NonZero(const VecI16& p_Values)
{
    const VecU16 k8000(0x8000);
    const VecI16 notZero(ssse3_sign_epi16(k8000, p_Values));
    return notZero;
}

// @return (p_Values != 0) ? 0x80000000 : 0
// 1 op
static INLINE VecI32 NonZero(const VecU32& p_Values)
{
    const VecU32 k80000000(0x80000000);
    const VecI32 notZero(ssse3_sign_epi32(k80000000, p_Values));
    return notZero;
}

// 7 ops
static INLINE VecI8 NonZero32(const VecU32& p_Values0, const VecU32& p_Values1,
    const VecU32& p_Values2, const VecU32& p_Values3)
{
    VecI32 notZero0 = NonZero(p_Values0);
    VecI32 notZero1 = NonZero(p_Values1);
    VecI16 notZero10 = Pack(notZero0, notZero1);
    VecI32 notZero2 = NonZero(p_Values2);
    VecI32 notZero3 = NonZero(p_Values3);
    VecI16 notZero32 = Pack(notZero2, notZero3);
    VecI8 notZero = Pack(notZero10, notZero32);
    return notZero;
}

// 3 ops for 4 values
static INLINE VecI32 IsZero64(const VecU64& p_Values10, const VecU64& p_Values32)
{
    const VecU64 zero(_mm_setzero_si128());
    const VecU64 isZero10 = (p_Values10 == zero);
    const VecU64 isZero32 = (p_Values32 == zero);
    const VecI32 isZero(Shuffle2<_MM_SHUFFLE(3, 1, 3, 1)>()(isZero10, isZero32));
    return isZero;
}

// 16 ops for 16 values
static INLINE VecI8 NonZero64(
    const VecU64& p_V0, const VecU64& p_V1, const VecU64& p_V2, const VecU64& p_V3,
    const VecU64& p_V4, const VecU64& p_V5, const VecU64& p_V6, const VecU64& p_V7)
{
    const VecI32 isZero0 = IsZero64(p_V0, p_V1);
    const VecI32 isZero1 = IsZero64(p_V2, p_V3);
    const VecI16 isZero01 = Pack(isZero0, isZero1);
    const VecI32 isZero2 = IsZero64(p_V4, p_V5);
    const VecI32 isZero3 = IsZero64(p_V6, p_V7);
    const VecI16 isZero23 = Pack(isZero2, isZero3);
    const VecI8 isZero = Pack(isZero01, isZero23);  // nonzero? 0x00 : 0xFF
    const VecI8 nonZero = AndNot(isZero, VecI8(-127 - 1));  // nonzero? 0x80 : 0x00
    return nonZero;
}

//-----------------------------------------------------------------------------
// Uint16

// 10 ops for 16 values
static INLINE VecU8 IndicesOfLowest1(const VecI16& p_ValuesL, const VecI16& p_ValuesH)
{
    // isolate lowest 1 (if present)
    const VecU16 max(SetMax());
    const VecI16 negL(ssse3_sign_epi16(p_ValuesL, max));
    const VecI16 negH(ssse3_sign_epi16(p_ValuesH, max));
    const VecU16 lsbL(p_ValuesL & negL);
    const VecU16 lsbH(p_ValuesH & negH);
    // minimal perfect hash function via de Bruijn sequence
    const VecU16 deBruijn(0x09AF);
    const VecU16 hashL = (lsbL * deBruijn) >> 12;  // 0..F
    const VecU16 hashH = (lsbH * deBruijn) >> 12;
    const VecU8 hash = Pack(hashL, hashH);
    // indexed by the hashes (0..F)
    const VecU8 permutedIndices(12, 13, 7, 14, 10, 8, 4, 15, 11, 6, 9, 3, 5, 2, 1, 0);
    const VecU8 indices = Shuffle(permutedIndices, hash);
    return indices;
}

// 6 ops
static INLINE void ValuesFromIndices16(const VecU8& p_Indices, VecU16& p_Values0, VecU16& p_Values1)
{
    // need two separate tables because the LUT entries are only 8 bits.
    const VecU8 tableL(_mm_cvtsi64_si128(0x8040201008040201ull));
    const VecU8 tableH(_mm_slli_si128(tableL, kVectorSize / 2));

    const VecU8 valuesL = Shuffle(tableL, p_Indices);
    const VecU8 valuesH = Shuffle(tableH, p_Indices);
    p_Values0 |= VecU16(UnpackLow(valuesL, valuesH));
    p_Values1 |= VecU16(UnpackHigh(valuesL, valuesH));
}

//-----------------------------------------------------------------------------
// Uint32

// 9 ops
static INLINE VecU16 BiasedExponentsOfLowest1(const VecU32& p_Values0, const VecU32& p_Values1,
    const VecU8& p_K81)
{
    const VecU32 lowestBitValue0(p_Values0 & VecU32(ssse3_sign_epi32(p_Values0, p_K81)));
    const VecU32 lowestBitValue1(p_Values1 & VecU32(ssse3_sign_epi32(p_Values1, p_K81)));
    const VecI32 floatBits0(_mm_castps_si128(_mm_cvtepi32_ps(lowestBitValue0)));
    const VecI32 floatBits1(_mm_castps_si128(_mm_cvtepi32_ps(lowestBitValue1)));
    const VecU16 biasedExponents10 = PackU(floatBits0 >> 23, floatBits1 >> 23);
    return biasedExponents10;
}

// 21 ops for 16 values
static INLINE VecU8 IndicesOfLowest1(const VecU32& p_Values0, const VecU32& p_Values1,
    const VecU32& p_Values2, const VecU32& p_Values3)
{
    const VecU8 k81(0x81);  // = -127; also for negating in BiasedExponentsOfLowest1
    VecU16 biasedExponents10 = BiasedExponentsOfLowest1(p_Values0, p_Values1, k81);
    VecU16 biasedExponents32 = BiasedExponentsOfLowest1(p_Values2, p_Values3, k81);
    const VecU8 exponents = Pack(biasedExponents10, biasedExponents32) + k81;
    // fix MSB (interpreted as a negative number)
    const VecU8 indices = Select(exponents, VecU8(31), exponents);
    return indices;
}

// 5 ops for 4 values
// @param p_ShiftCounts (0..3) << 3
static INLINE VecU32 ShiftBytesIntoU32(const VecU8& p_Values, const VecU8& p_ShiftCounts,
    VecU8& p_BroadcastX4)
{
    const VecU8 k18100800(VecU32(0x18100800));
    // send four bytes (from lower 3 index bits) to their four possible positions.
    const VecU8 valuesX4 = Shuffle(p_Values, p_BroadcastX4);
    // keep only the byte whose position matches the desired shift count.
    const VecU8 shiftCountsX4 = Shuffle(p_ShiftCounts, p_BroadcastX4);
    const VecU8 masks = (shiftCountsX4 == k18100800);
    p_BroadcastX4 += VecU8(4);  // much faster than shifting/aligning p_ShiftCounts and p_Values!
    const VecU8 shifted = valuesX4 & masks;
    return VecU32(shifted);
}

// 27 ops for 16 values
static INLINE void ValuesFromIndices32(const VecU8& p_Indices,
    VecU32& p_V0, VecU32& p_V1, VecU32& p_V2, VecU32& p_V3)
{
#if ENABLE_AVX2
    const VecU32 one(1);
    VecU8 indices = p_Indices & VecU8(0xDF);  // clear further bit
    const VecU32 indices0(sse41_cvtepu8_epi32(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU32::numLanes));
    p_V0 |= VecU32(_mm_sllv_epi32(one, indices0));
    const VecU32 indices1(sse41_cvtepu8_epi32(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU32::numLanes));
    p_V1 |= VecU32(_mm_sllv_epi32(one, indices1));
    const VecU32 indices2(sse41_cvtepu8_epi32(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU32::numLanes));
    p_V2 |= VecU32(_mm_sllv_epi32(one, indices2));
    const VecU32 indices3(sse41_cvtepu8_epi32(indices));
    p_V3 |= VecU32(_mm_sllv_epi32(one, indices3));
#else
    const VecU8 tableShift3(0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1, 0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1);
    const VecU8 values = Shuffle(tableShift3, p_Indices);
    // (clear 0x80, `further index' and lower 3 bits)
    const VecU8 shiftCounts = p_Indices & VecU8(0x18);

    VecU8 broadcastX4(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    p_V0 |= ShiftBytesIntoU32(values, shiftCounts, broadcastX4);
    p_V1 |= ShiftBytesIntoU32(values, shiftCounts, broadcastX4);
    p_V2 |= ShiftBytesIntoU32(values, shiftCounts, broadcastX4);
    p_V3 |= ShiftBytesIntoU32(values, shiftCounts, broadcastX4);
#endif
}

//-----------------------------------------------------------------------------
// Uint64

// 17 ops for 4 (mostly scalar)
// AVX512: vectorize with _mm512_lzcnt_epi64
// (packing U32 is slightly faster than interleaving bytes)
// (FPU normalization must combine separate u32 halves because SSE4 lacks U64->F64 conversion)
// (De Bruijn requires 64-bit multiply (not in SSE4) or 3 extra ops for folding;
// we'd still be left with a 6-bit table, i.e. 4 lookups + 3 select.)
// see https://chessprogramming.wikispaces.com/BitScan.
static INLINE VecI32 IndicesOfLowest1(const VecU64& p_Values0, const VecU64& p_Values1)
{
    // (sse41_extract_epi64 is slower)
    const Uint64 value0 = _mm_cvtsi128_si64(p_Values0);
    const Uint64 value1 = _mm_cvtsi128_si64(MoveHL(p_Values0));
    const Uint64 value2 = _mm_cvtsi128_si64(p_Values1);
    const Uint64 value3 = _mm_cvtsi128_si64(MoveHL(p_Values1));

    unsigned long tmp;  // shared variable avoids an unnecessary spill to memory
    (void)_BitScanForward64(&tmp, value0);
    const size_t index0 = tmp;
    (void)_BitScanForward64(&tmp, value1);
    const size_t index1 = tmp;
    (void)_BitScanForward64(&tmp, value2);
    const size_t index2 = tmp;
    (void)_BitScanForward64(&tmp, value3);
    const size_t index3 = tmp;

    // (faster than MOVQ + _mm_insert_epi16/32, and loading from memory)
    const VecI32 indexV0(_mm_cvtsi64_si128(index0));
    const VecI32 indexV1(_mm_cvtsi64_si128(index1));
    const VecI32 index10 = UnpackLow(indexV0, indexV1);
    const VecI32 indexV2(_mm_cvtsi64_si128(index2));
    const VecI32 indexV3(_mm_cvtsi64_si128(index3));
    const VecI32 index32 = UnpackLow(indexV2, indexV3);
    const VecI32 indices3210(_mm_unpacklo_epi64(index10, index32));
    return indices3210;
}

// 71 ops for 16
static INLINE VecU8 IndicesOfLowest1(
    const VecU64& p_V0, const VecU64& p_V1, const VecU64& p_V2, const VecU64& p_V3,
    const VecU64& p_V4, const VecU64& p_V5, const VecU64& p_V6, const VecU64& p_V7)
{
    {
        const VecI32 indices3210 = IndicesOfLowest1(p_V0, p_V1);
        const VecI32 indices7654 = IndicesOfLowest1(p_V2, p_V3);
        const VecI16 indices76543210 = Pack(indices3210, indices7654);

        const VecI32 indicesBA98 = IndicesOfLowest1(p_V4, p_V5);
        const VecI32 indicesFEDC = IndicesOfLowest1(p_V6, p_V7);
        const VecI16 indicesFEDCBA98 = Pack(indicesBA98, indicesFEDC);

        const VecU8 indices = PackU(indices76543210, indicesFEDCBA98);
        return indices;
    }
}

// given bytes, use upper 3 index bits to position them within Uint64 (via Shuffle).
// resulting decoder is 20% faster than when using scalar shifts.
// 3 ops
template<size_t t_Pos>
static INLINE VecU64 ValuesFromIndices64(const VecU8& p_Codes, VecU8& p_Lower)
{
    const size_t code = sse41_extract_epi8(p_Codes, t_Pos);
    ASSERT(code < 64);

    const VecU8 control = Load(crpcU8(g_Expand64) + code * kVectorSize);
    const VecU64 values(Shuffle(p_Lower, control));
    p_Lower = VecU8(_mm_srli_si128(p_Lower, VecU64::numLanes));
    return values;
}

static INLINE void ValuesFromIndices64(const VecU8& p_Indices,
    VecU64& p_V0, VecU64& p_V1, VecU64& p_V2, VecU64& p_V3,
    VecU64& p_V4, VecU64& p_V5, VecU64& p_V6, VecU64& p_V7)
{
#if ENABLE_AVX2
    const VecU64 one(1);
    VecU8 indices = p_Indices & VecU8(0xBF);  // clear further bit
    const VecU64 indices0(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V0 |= VecU64(_mm_sllv_epi64(one, indices0));
    const VecU64 indices1(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V1 |= VecU64(_mm_sllv_epi64(one, indices1));
    const VecU64 indices2(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V2 |= VecU64(_mm_sllv_epi64(one, indices2));
    const VecU64 indices3(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V3 |= VecU64(_mm_sllv_epi64(one, indices3));
    const VecU64 indices4(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V4 |= VecU64(_mm_sllv_epi64(one, indices4));
    const VecU64 indices5(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V5 |= VecU64(_mm_sllv_epi64(one, indices5));
    const VecU64 indices6(sse41_cvtepu8_epi64(indices));
    indices = VecU8(_mm_srli_si128(indices, VecU64::numLanes));
    p_V6 |= VecU64(_mm_sllv_epi64(one, indices6));
    const VecU64 indices7(sse41_cvtepu8_epi64(indices));
    p_V7 |= VecU64(_mm_sllv_epi64(one, indices7));
#else
    // lower three bits, or zero if p_Indices & 0x80
    const VecU8 table(0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1, 0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1);
    VecU8 lower = Shuffle(table, p_Indices);

    // remove 0x80 (handled by Shuffle), `further index' bit and lower index bits.
    const VecU8 indices = p_Indices & VecU8(0x38);  // 00HHH000 00LLL000
    const VecU8 indicesL(VecU16(indices) >> 3);  // 00000LLL
    // code = 00HHHLLL in even lanes.
    const VecU8 codes = indicesL + VecU8(_mm_srli_si128(indices, 1));

    p_V0 |= ValuesFromIndices64<0>(codes, lower);
    p_V1 |= ValuesFromIndices64<2>(codes, lower);
    p_V2 |= ValuesFromIndices64<4>(codes, lower);
    p_V3 |= ValuesFromIndices64<6>(codes, lower);
    p_V4 |= ValuesFromIndices64<8>(codes, lower);
    p_V5 |= ValuesFromIndices64<10>(codes, lower);
    p_V6 |= ValuesFromIndices64<12>(codes, lower);
    p_V7 |= ValuesFromIndices64<14>(codes, lower);
#endif
}

//-----------------------------------------------------------------------------
// 128 bit

// @return index in lower 64 bits, garbage in upper 64 bits
// (if p_Value128 == 0, return value is ignored)
// 8 ops
static INLINE VecU64 IndexOfLowest1(const VecU64& p_Value128,
    const VecU64& p_IsZeroHL, const VecU64& p_K64)
{
    unsigned long tmp;  // shared variable avoids an unnecessary spill to memory

    const Uint64 bitsL = _mm_cvtsi128_si64(p_Value128);
    (void)_BitScanForward64(&tmp, bitsL);
    const VecU64 indexL2(_mm_cvtsi32_si128(tmp));

    const Uint64 bitsH = sse41_extract_epi64(p_Value128, 1);
    (void)_BitScanForward64(&tmp, bitsH);
    const VecU64 indexH2(_mm_cvtsi32_si128(tmp));

    const VecU64 index = Select(p_IsZeroHL, indexH2 + p_K64, indexL2);
    return index;
}

// @param p_K1 = 1; hoists the constant load out of an inner loop.
static INLINE VecU64 ValueFromIndex128(size_t p_Index, const VecU64& p_K1)
{
    const VecU64 indexL(_mm_cvtsi64_si128(p_Index));       // 0 if p_Index >= 64
    const VecU64 indexH(_mm_cvtsi64_si128(p_Index - 64));  // 0 if p_Index < 64
    const VecU64 valueL = p_K1 << indexL;
    const VecU64 valueH = p_K1 << indexH;
    return UnpackLow(valueL, valueH);
}

void TestIndices();
