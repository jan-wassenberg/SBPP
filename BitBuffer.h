// Copyright (c) 2015 Jan Wassenberg

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "Vector.h"
#include "Ptr.h"

namespace codec {

// FIFO queues for single and multi-bit packets. load/store from/to memory in
// units of 4 or 8 bytes. inserting in the least-significant bits is convenient,
// and extracting from the most-significant bits is no loss because
// we use the sign bit to avoid masking.
//
// BitSource/Sink load/store 8 bytes the first time, otherwise 4 bytes.
// ensure the Load/Store64 match the usual behavior by byte-swapping the
// lower 32 bits. this also ensures the unused buf bits reside in the
// highest memory addresses and can therefore be skipped.

class BitSink
{
public:
    BitSink() : m_Buf(0), m_UpperBitsUsed(-32) {}

    // no effect if the lower half of the buffer isn't yet full.
    // postcondition: the buffer has space for at least 32 more bits.
    INLINE PtrW FlushLowerBits(const PtrW p_Out)
    {
        // lower bits are not yet full.
        if (m_UpperBitsUsed < 0) return p_Out;

        const VecU64 oldBits = m_Buf >> _mm_cvtsi64_si128(m_UpperBitsUsed);
        const Uint32 out = SDL_Swap32(_mm_cvtsi128_si32(oldBits));
        Store(out, p_Out);
        m_UpperBitsUsed -= 32;
        return p_Out + sizeof(Uint32);
    }

    // NOTE: no more than 32 bits may be inserted between calls to FlushLowerBits.
    INLINE void Insert0()
    {
        m_Buf <<= 1;
        m_UpperBitsUsed += 1;
    }

    // NOTE: no more than 32 bits may be inserted between calls to FlushLowerBits.
    INLINE void Insert(size_t p_Value, size_t p_NumBits)
    {
        m_Buf <<= _mm_cvtsi64_si128(p_NumBits);
        m_Buf |= VecU64(_mm_cvtsi64_si128(p_Value));
        m_UpperBitsUsed += p_NumBits;
    }

    // NOTE: no more than 32 bits may be inserted between calls to FlushLowerBits.
    INLINE void InsertV(const VecU64& p_Value, const VecU64& p_NumBits)
    {
        m_Buf <<= p_NumBits;
        m_Buf |= p_Value;
        m_UpperBitsUsed += _mm_cvtsi128_si64(p_NumBits);
    }

    // destroys internal state, must be called last.
    // @return end of output (rounded up to whole bytes)
    INLINE PtrW Flush(const PtrW p_Out)
    {
        // left-align oldest bit (inserting zeros at the bottom)
        m_Buf <<= _mm_cvtsi64_si128(32 - m_UpperBitsUsed);

        const Uint64 out = SDL_Swap64(_mm_cvtsi128_si64(m_Buf));
        const size_t bytesUsed = Align<CHAR_BIT>(m_UpperBitsUsed + 32) / CHAR_BIT;
        // must not write the whole Uint64 because that may overwrite
        // other threads' output!
        return StoreFrom(&out, bytesUsed, p_Out);
    }

private:
    VecU64 m_Buf;
    Sint64 m_UpperBitsUsed;
};

//-----------------------------------------------------------------------------

class BitSource
{
public:
    BitSource() : m_Buf(0), m_UpperBitsExtracted(-32) {}

    // must call after ctor (needs to be separate due to return value).
    // not the same as Refill - this fills the ENTIRE buffer and
    // is only safe if we know the buffer is empty.
    INLINE PtrR FillBuffer(const PtrR p_In)
    {
        m_Buf = _mm_cvtsi64_si128(SDL_Swap64(Load<Uint64>(p_In)));
        return p_In + sizeof(Uint64);
    }

    INLINE bool NeedsRefill() const
    {
        return (m_UpperBitsExtracted >= 0);
    }

    INLINE PtrR Refill(const PtrR p_In)
    {
        const __m128i shift = _mm_cvtsi64_si128(m_UpperBitsExtracted);
        m_Buf >>= shift;
        const Uint32 next = SDL_Swap32(Load<Uint32>(p_In));
        m_Buf = VecU64(sse41_insert_epi32(m_Buf, next, 0));
        m_Buf <<= shift;
        m_UpperBitsExtracted -= 32;
        return p_In + sizeof(Uint32);
    }

    // if we read past the end, rewind to the proper end.
    INLINE PtrR Rewind(const PtrR p_In)
    {
        if (m_UpperBitsExtracted >= 0) return p_In;

        const size_t excessBytes = (32 - m_UpperBitsExtracted) / CHAR_BIT;
        return p_In - excessBytes;
    }

    // @return the most-significant bit in the sign bit; all other bits are undefined.
    INLINE Sint64 ExtractBit()
    {
        const size_t ret = SizeFromM128(m_Buf);
        m_Buf += m_Buf;
        m_UpperBitsExtracted += 1;
        return static_cast<Sint64>(ret);
    }

    INLINE size_t Extract(size_t p_NumBits)
    {
        const VecU64 bits = m_Buf >> _mm_cvtsi64_si128(64 - p_NumBits);
        const size_t value = SizeFromM128(bits);
        m_Buf <<= _mm_cvtsi64_si128(p_NumBits);
        m_UpperBitsExtracted += p_NumBits;
        return value;
    }

private:
    VecU64 m_Buf;
    Sint64 m_UpperBitsExtracted;
};

}  // namespace codec
