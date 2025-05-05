#include <motioncam/RawData.hpp>
#include <vector>
#include <cstring>

#include <arm_neon.h>

#if defined(__GNUC__)
#define INLINE __attribute__((always_inline))
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define INLINE __forceinline
#define RESTRICT __restrict
#else
#define INLINE
#define RESTRICT
#endif

namespace motioncam
{
    namespace raw
    {

        static constexpr int ENCODING_BLOCK = 64;
        static constexpr int HEADER_LENGTH = 2;
        static constexpr int METADATA_OFFSET = 16;
        static constexpr int PREFETCH_DISTANCE = 256;

        static constexpr int ENCODING_BLOCK_LEN[17] = {
            0,   // bits = 0
            8,   // bits = 1
            16,  // bits = 2
            24,  // bits = 3
            32,  // bits = 4
            40,  // bits = 5
            48,  // bits = 6
            64,  // bits = 7
            64,  // bits = 8
            80,  // bits = 9
            80,  // bits = 10
            128, // bits = 11
            128, // bits = 12
            128, // bits = 13
            128, // bits = 14
            128, // bits = 15
            128  // bits = 16
        };

        // Handy NEON loaders/storers
        static inline uint16x8_t Load8(const uint8_t *ptr)
        {
            // vld1_u8 is an 8×8-bit load, then widen to 8×16
            return vmovl_u8(vld1_u8(ptr));
        }
        static inline void Store8(uint16_t *ptr, uint16x8_t v)
        {
            vst1q_u16(ptr, v);
        }

        // Prototype for each bit‐width decoder
        using DecodeFn = const uint8_t *(*)(uint16_t *RESTRICT, const uint8_t *RESTRICT);

        // bits = 0: all zeros
        static inline const uint8_t *Decode0(uint16_t *RESTRICT output, const uint8_t *input)
        {
            // produce sixty-four zeros
            std::memset(output, 0, ENCODING_BLOCK * sizeof(uint16_t));
            return input;
        }

        // bits = 1
        static inline const uint8_t *Decode1(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            uint16x8_t p = Load8(input);
            const uint16x8_t one = vdupq_n_u16(1);

            // extract lanes (p >> k) & 1
            uint16x8_t r0 = vandq_u16(p, one);
            uint16x8_t r1 = vandq_u16(vshrq_n_u16(p, 1), one);
            uint16x8_t r2 = vandq_u16(vshrq_n_u16(p, 2), one);
            uint16x8_t r3 = vandq_u16(vshrq_n_u16(p, 3), one);
            uint16x8_t r4 = vandq_u16(vshrq_n_u16(p, 4), one);
            uint16x8_t r5 = vandq_u16(vshrq_n_u16(p, 5), one);
            uint16x8_t r6 = vandq_u16(vshrq_n_u16(p, 6), one);
            uint16x8_t r7 = vandq_u16(vshrq_n_u16(p, 7), one);

            Store8(output + 0, r0);
            Store8(output + 8, r1);
            Store8(output + 16, r2);
            Store8(output + 24, r3);
            Store8(output + 32, r4);
            Store8(output + 40, r5);
            Store8(output + 48, r6);
            Store8(output + 56, r7);

            return input + ENCODING_BLOCK_LEN[1];
        }

        // bits = 2
        static inline const uint8_t *Decode2(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            // process 8 bytes → 4 lanes each of 2-bit words, then repeat
            auto sub = [&](uint16_t *out8, const uint8_t *in8)
            {
                uint16x8_t p = Load8(in8);
                const uint16x8_t m3 = vdupq_n_u16(0x03);
                uint16x8_t r0 = vandq_u16(p, m3);
                uint16x8_t r1 = vandq_u16(vshrq_n_u16(p, 2), m3);
                uint16x8_t r2 = vandq_u16(vshrq_n_u16(p, 4), m3);
                uint16x8_t r3 = vandq_u16(vshrq_n_u16(p, 6), m3);
                Store8(out8 + 0, r0);
                Store8(out8 + 8, r1);
                Store8(out8 + 16, r2);
                Store8(out8 + 24, r3);
            };
            sub(output + 0, input + 0);
            sub(output + 32, input + 8);
            return input + ENCODING_BLOCK_LEN[2];
        }

        // bits = 3
        static inline const uint8_t *Decode3(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            //  3 bits from p0,p1,p2; upper bits come from shifts of p2
            const uint16x8_t N = vdupq_n_u16(0x07);
            const uint16x8_t T = vdupq_n_u16(0x03);
            const uint16x8_t R = vdupq_n_u16(0x01);

            uint16x8_t p0 = Load8(input + 0);
            uint16x8_t p1 = Load8(input + 8);
            uint16x8_t p2 = Load8(input + 16);

            uint16x8_t r0 = vandq_u16(p0, N);
            uint16x8_t r1 = vandq_u16(vshrq_n_u16(p0, 3), N);
            uint16x8_t t2 = vandq_u16(vshrq_n_u16(p0, 6), T);
            uint16x8_t r3 = vandq_u16(p1, N);
            uint16x8_t r4 = vandq_u16(vshrq_n_u16(p1, 3), N);
            uint16x8_t t5 = vandq_u16(vshrq_n_u16(p1, 6), T);
            uint16x8_t r6 = vandq_u16(p2, N);
            uint16x8_t r7 = vandq_u16(vshrq_n_u16(p2, 3), N);

            // rebuild bits 2 & 5 from p2 high bits
            uint16x8_t b2 = vorrq_u16(t2,
                                      vshlq_n_u16(vandq_u16(vshrq_n_u16(p2, 6), R), 2));
            uint16x8_t b5 = vorrq_u16(t5,
                                      vshlq_n_u16(vandq_u16(vshrq_n_u16(p2, 7), R), 2));

            Store8(output + 0, r0);
            Store8(output + 8, r1);
            Store8(output + 16, b2);
            Store8(output + 24, r3);
            Store8(output + 32, r4);
            Store8(output + 40, b5);
            Store8(output + 48, r6);
            Store8(output + 56, r7);

            return input + ENCODING_BLOCK_LEN[3];
        }

        // bits = 4
        static inline const uint8_t *Decode4(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            auto sub = [&](uint16_t *out16, const uint8_t *in8)
            {
                uint16x8_t p = Load8(in8);
                const uint16x8_t M = vdupq_n_u16(0x0F);
                uint16x8_t r0 = vandq_u16(p, M);
                uint16x8_t r1 = vandq_u16(vshrq_n_u16(p, 4), M);
                Store8(out16 + 0, r0);
                Store8(out16 + 8, r1);
            };
            for (int i = 0; i < 4; i++)
            {
                sub(output + i * 16, input + i * 8);
            }
            return input + ENCODING_BLOCK_LEN[4];
        }

        // bits = 5
        static inline const uint8_t *Decode5(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            // see original for logic—just unrolled here
            const uint16x8_t N = vdupq_n_u16(0x1F);
            const uint16x8_t L = vdupq_n_u16(0x07);
            const uint16x8_t U = vdupq_n_u16(0x03);
            const uint16x8_t F = vdupq_n_u16(0x01);

            uint16x8_t p0 = Load8(input + 0);
            uint16x8_t p1 = Load8(input + 8);
            uint16x8_t p2 = Load8(input + 16);
            uint16x8_t p3 = Load8(input + 24);
            uint16x8_t p4 = Load8(input + 32);

            uint16x8_t r0 = vandq_u16(p0, N);
            uint16x8_t r1 = vandq_u16(p1, N);
            uint16x8_t r2 = vandq_u16(p2, N);
            uint16x8_t r3 = vandq_u16(p3, N);
            uint16x8_t r4 = vandq_u16(p4, N);

            uint16x8_t r5 = vorrq_u16(vandq_u16(vshrq_n_u16(p0, 5), L),
                                      vshlq_n_u16(vandq_u16(vshrq_n_u16(p3, 5), U), 3));
            uint16x8_t r6 = vorrq_u16(vandq_u16(vshrq_n_u16(p1, 5), L),
                                      vshlq_n_u16(vandq_u16(vshrq_n_u16(p4, 5), U), 3));

            uint16x8_t tmp = vandq_u16(vshrq_n_u16(p2, 5), L);
            uint16x8_t r7 = vorrq_u16(
                vorrq_u16(tmp,
                          vshlq_n_u16(vandq_u16(vshrq_n_u16(p3, 7), F), 3)),
                vshlq_n_u16(vandq_u16(vshrq_n_u16(p4, 7), F), 4));

            Store8(output + 0, r0);
            Store8(output + 8, r1);
            Store8(output + 16, r2);
            Store8(output + 24, r3);
            Store8(output + 32, r4);
            Store8(output + 40, r5);
            Store8(output + 48, r6);
            Store8(output + 56, r7);

            return input + ENCODING_BLOCK_LEN[5];
        }

        // bits = 6
        static inline const uint8_t *Decode6(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            const uint16x8_t N = vdupq_n_u16(0x3F);
            const uint16x8_t L = vdupq_n_u16(0x03);

            uint16x8_t p0 = Load8(input + 0);
            uint16x8_t p1 = Load8(input + 8);
            uint16x8_t p2 = Load8(input + 16);
            uint16x8_t p3 = Load8(input + 24);
            uint16x8_t p4 = Load8(input + 32);
            uint16x8_t p5 = Load8(input + 40);

            uint16x8_t r0 = vandq_u16(p0, N);
            uint16x8_t r1 = vandq_u16(p1, N);
            uint16x8_t r2 = vandq_u16(p2, N);
            uint16x8_t r3 = vandq_u16(p3, N);
            uint16x8_t r4 = vandq_u16(p4, N);
            uint16x8_t r5 = vandq_u16(p5, N);

            // r6: bits from p0..p2
            uint16x8_t r6 = vorrq_u16(
                vorrq_u16(vandq_u16(vshrq_n_u16(p0, 6), L),
                          vshlq_n_u16(vandq_u16(vshrq_n_u16(p1, 6), L), 2)),
                vshlq_n_u16(vandq_u16(vshrq_n_u16(p2, 6), L), 4));

            uint16x8_t r7 = vorrq_u16(
                vorrq_u16(vandq_u16(vshrq_n_u16(p3, 6), L),
                          vshlq_n_u16(vandq_u16(vshrq_n_u16(p4, 6), L), 2)),
                vshlq_n_u16(vandq_u16(vshrq_n_u16(p5, 6), L), 4));

            Store8(output + 0, r0);
            Store8(output + 8, r1);
            Store8(output + 16, r2);
            Store8(output + 24, r3);
            Store8(output + 32, r4);
            Store8(output + 40, r5);
            Store8(output + 48, r6);
            Store8(output + 56, r7);

            return input + ENCODING_BLOCK_LEN[6];
        }

        // bits = 7 or 8: raw 8-bit
        static inline const uint8_t *Decode8(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            for (int i = 0; i < 8; i++)
            {
                Store8(output + i * 8, Load8(input + i * 8));
            }
            return input + ENCODING_BLOCK_LEN[8];
        }

        // bits = 9 or 10: 10-bit split across 2 blocks of ten
        static inline const uint8_t *Decode10(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            // first 4 lanes in p0..p3, their hi2 bits in p4
            uint16x8_t p0 = Load8(input + 0);
            uint16x8_t p1 = Load8(input + 8);
            uint16x8_t p2 = Load8(input + 16);
            uint16x8_t p3 = Load8(input + 24);
            uint16x8_t p4 = Load8(input + 32);
            uint16x8_t p5 = Load8(input + 40);
            uint16x8_t p6 = Load8(input + 48);
            uint16x8_t p7 = Load8(input + 56);
            uint16x8_t p8 = Load8(input + 64);
            uint16x8_t p9 = Load8(input + 72);

            const uint16x8_t N = vdupq_n_u16(0xFF);
            const uint16x8_t L = vdupq_n_u16(0x03);

            // unpack lower 8 bits
            uint16x8_t _r0 = vandq_u16(p0, N);
            uint16x8_t _r1 = vandq_u16(p1, N);
            uint16x8_t _r2 = vandq_u16(p2, N);
            uint16x8_t _r3 = vandq_u16(p3, N);

            uint16x8_t r0 = vorrq_u16(_r0, vshlq_n_u16(vandq_u16(p4, L), 8));
            uint16x8_t r1 = vorrq_u16(_r1, vshlq_n_u16(vandq_u16(p4, L << 2), 6));
            uint16x8_t r2 = vorrq_u16(_r2, vshlq_n_u16(vandq_u16(p4, L << 4), 4));
            uint16x8_t r3 = vorrq_u16(_r3, vshlq_n_u16(vandq_u16(p4, L << 6), 2));

            uint16x8_t _r4 = vandq_u16(p5, N);
            uint16x8_t _r5 = vandq_u16(p6, N);
            uint16x8_t _r6 = vandq_u16(p7, N);
            uint16x8_t _r7 = vandq_u16(p8, N);

            uint16x8_t r4 = vorrq_u16(_r4, vshlq_n_u16(vandq_u16(p9, L), 8));
            uint16x8_t r5 = vorrq_u16(_r5, vshlq_n_u16(vandq_u16(p9, L << 2), 6));
            uint16x8_t r6 = vorrq_u16(_r6, vshlq_n_u16(vandq_u16(p9, L << 4), 4));
            uint16x8_t r7 = vorrq_u16(_r7, vshlq_n_u16(vandq_u16(p9, L << 6), 2));

            Store8(output + 0, r0);
            Store8(output + 8, r1);
            Store8(output + 16, r2);
            Store8(output + 24, r3);
            Store8(output + 32, r4);
            Store8(output + 40, r5);
            Store8(output + 48, r6);
            Store8(output + 56, r7);

            return input + ENCODING_BLOCK_LEN[10];
        }

        // bits = 16: raw 16-bit
        static inline const uint8_t *Decode16(uint16_t *RESTRICT output, const uint8_t *input)
        {
            __builtin_prefetch(input + PREFETCH_DISTANCE, 0, 1);
            // treat input as little-endian u16[]
            const uint16_t *in16 = reinterpret_cast<const uint16_t *>(input);
            for (int i = 0; i < 8; i++)
            {
                vst1q_u16(output + i * 8,
                          vld1q_u16(in16 + i * 8));
            }
            return input + ENCODING_BLOCK_LEN[16];
        }

        // Dispatch table, indexed by bit-width
        static DecodeFn const decodeTable[17] = {
            Decode0, Decode1, Decode2, Decode3,
            Decode4, Decode5, Decode6, Decode8,
            Decode8, Decode10, Decode10,
            Decode16, Decode16, Decode16, Decode16,
            Decode16, Decode16};

        // ——————————————————————————————————————————————————————————————————————————————
        //  Read a single block of ENCODING_BLOCK values
        // ——————————————————————————————————————————————————————————————————————————————
        static inline size_t DecodeBlock(
            uint16_t *RESTRICT output,
            uint16_t bits,
            const uint8_t *input,
            size_t offset,
            size_t len)
        {
            // guard
            if (offset + ENCODING_BLOCK_LEN[bits] > len)
                return len - offset;

            const uint8_t *ptr = input + offset;
            // call the right decoder
            ptr = decodeTable[bits](output, ptr);
            return ptr - (input + offset);
        }

        // ——————————————————————————————————————————————————————————————————————————————
        //  Decode metadata runs of blocks
        // ——————————————————————————————————————————————————————————————————————————————
        static inline size_t DecodeMetadata(
            const uint8_t *input,
            size_t offset,
            size_t len,
            std::vector<uint16_t> &out)
        {
            // first 4 bytes = number of blocks
            uint32_t num = uint32_t(input[offset + 0]) | uint32_t(input[offset + 1]) << 8 | uint32_t(input[offset + 2]) << 16 | uint32_t(input[offset + 3]) << 24;
            out.resize(num);
            offset += 4;

            uint16_t *data = out.data();
            for (uint32_t i = 0; i < num; i += ENCODING_BLOCK)
            {
                uint8_t bits;
                uint16_t ref;
                // header = 2 bytes
                bits = (input[offset + 0] >> 4) & 0x0F;
                ref = ((input[offset + 0] & 0x0F) << 8) | input[offset + 1];
                offset += HEADER_LENGTH;

                size_t consumed = DecodeBlock(data, bits, input, offset, len);
                offset += consumed;

                // add reference
                for (int j = 0; j < ENCODING_BLOCK; j++)
                    data[j] += ref;
                data += ENCODING_BLOCK;
            }
            return offset;
        }

        // ——————————————————————————————————————————————————————————————————————————————
        //  Top‐level decode entry point
        // ——————————————————————————————————————————————————————————————————————————————
        void ReadMetadataHeader(
            const uint8_t *in,
            uint32_t &w,
            uint32_t &h,
            uint32_t &bitsOff,
            uint32_t &refsOff)
        {
            w = uint32_t(in[0]) | uint32_t(in[1]) << 8 | uint32_t(in[2]) << 16 | uint32_t(in[3]) << 24;
            h = uint32_t(in[4]) | uint32_t(in[5]) << 8 | uint32_t(in[6]) << 16 | uint32_t(in[7]) << 24;
            bitsOff = uint32_t(in[8]) | uint32_t(in[9]) << 8 | uint32_t(in[10]) << 16 | uint32_t(in[11]) << 24;
            refsOff = uint32_t(in[12]) | uint32_t(in[13]) << 8 | uint32_t(in[14]) << 16 | uint32_t(in[15]) << 24;
        }

        size_t Decode(
            uint16_t *output,
            int width,
            int height,
            const uint8_t *input,
            size_t len)
        {
            const uint16_t *outStart = output;

            // read header
            uint32_t encW, encH, bitsOff, refsOff;
            ReadMetadataHeader(input, encW, encH, bitsOff, refsOff);
            if (bitsOff > len || refsOff > len || encW % ENCODING_BLOCK != 0 || encW < (uint32_t)width)
                return 0;

            // decode bit‐runs & ref‐runs
            std::vector<uint16_t> bits, refs;
            DecodeMetadata(input, bitsOff, len, bits);
            DecodeMetadata(input, refsOff, len, refs);

            // row buffers
            std::vector<uint16_t> row0(encW), row1(encW), row2(encW), row3(encW);

            size_t offset = METADATA_OFFSET;
            int metaIdx = 0;

            alignas(16) uint16_t p0[ENCODING_BLOCK];
            alignas(16) uint16_t p1[ENCODING_BLOCK];
            alignas(16) uint16_t p2[ENCODING_BLOCK];
            alignas(16) uint16_t p3[ENCODING_BLOCK];

            for (int y = 0; y < (int)encH; y += 4)
            {
                for (int x = 0; x < (int)encW; x += ENCODING_BLOCK)
                {
                    uint16_t b0 = bits[metaIdx + 0], b1 = bits[metaIdx + 1],
                             b2 = bits[metaIdx + 2], b3 = bits[metaIdx + 3];
                    uint16_t r0 = refs[metaIdx + 0], r1 = refs[metaIdx + 1],
                             r2 = refs[metaIdx + 2], r3 = refs[metaIdx + 3];

                    offset += DecodeBlock(p0, b0, input, offset, len);
                    offset += DecodeBlock(p1, b1, input, offset, len);
                    offset += DecodeBlock(p2, b2, input, offset, len);
                    offset += DecodeBlock(p3, b3, input, offset, len);

                    // interleave + add references
                    for (int i = 0; i < ENCODING_BLOCK; i += 2)
                    {
                        int xi = x + i;
                        row0[xi] = p0[i / 2] + r0;
                        row0[xi + 1] = p1[i / 2] + r1;
                        row1[xi] = p2[i / 2] + r2;
                        row1[xi + 1] = p3[i / 2] + r3;
                        row2[xi] = p0[ENCODING_BLOCK / 2 + i / 2] + r0;
                        row2[xi + 1] = p1[ENCODING_BLOCK / 2 + i / 2] + r1;
                        row3[xi] = p2[ENCODING_BLOCK / 2 + i / 2] + r2;
                        row3[xi + 1] = p3[ENCODING_BLOCK / 2 + i / 2] + r3;
                    }
                    metaIdx += 4;
                }
                // copy 4 rows out
                std::memcpy(output, row0.data(), width * sizeof(uint16_t));
                output += width;
                std::memcpy(output, row1.data(), width * sizeof(uint16_t));
                output += width;
                std::memcpy(output, row2.data(), width * sizeof(uint16_t));
                output += width;
                std::memcpy(output, row3.data(), width * sizeof(uint16_t));
                output += width;
            }

            return size_t(output - outStart);
        }

    }
} // namespace motioncam::raw