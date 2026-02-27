const std = @import("std");

/// A generic secure Centered Binomial Distribution (CBD) sampler.
/// Uses Blake3 to securely generate deterministic random distributions from a uniform seed.
pub const CBD = struct {
    seed: [32]u8,
    nonce: u16,

    pub fn init(seed: [32]u8, nonce: u16) CBD {
        return .{ .seed = seed, .nonce = nonce };
    }

    /// Samples a polynomial of arbitrary length using a cryptographic PRF bounds.
    pub fn samplePoly(self: *CBD, comptime eta: u3, poly: []i32) void {
        var hasher = std.crypto.hash.Blake3.init(.{});
        hasher.update(&self.seed);

        var nonce_buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &nonce_buf, self.nonce, .little);
        hasher.update(&nonce_buf);

        // Calculate bounds. Each coefficient needs 2 * eta bits.
        const bits_needed = poly.len * 2 * eta;
        const bytes_needed = (bits_needed + 7) / 8;

        // Heap allocate exactly the requested bounds to prevent bounds mismatches
        var random_bytes = std.heap.page_allocator.alloc(u8, bytes_needed) catch unreachable;
        defer std.heap.page_allocator.free(random_bytes);

        var offset: usize = 0;
        var block_ctr: u32 = 0;
        while (offset < random_bytes.len) {
            var block_hasher = hasher;
            var ctr_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &ctr_buf, block_ctr, .little);
            block_hasher.update(&ctr_buf);

            var block: [32]u8 = undefined;
            block_hasher.final(&block);

            const to_copy = @min(block.len, random_bytes.len - offset);
            @memcpy(random_bytes[offset .. offset + to_copy], block[0..to_copy]);
            offset += to_copy;
            block_ctr += 1;
        }

        var byte_idx: usize = 0;
        var bit_idx: u3 = 0;

        const get_bit = struct {
            fn f(bytes: []const u8, b_idx: *usize, bit_i: *u3) u1 {
                const bit = @as(u1, @truncate(bytes[b_idx.*] >> bit_i.*));
                if (bit_i.* == 7) {
                    bit_i.* = 0;
                    b_idx.* += 1;
                } else {
                    bit_i.* += 1;
                }
                return bit;
            }
        }.f;

        for (poly) |*item| {
            var a: u32 = 0;
            var b: u32 = 0;
            for (0..eta) |_| {
                a += get_bit(random_bytes, &byte_idx, &bit_idx);
                b += get_bit(random_bytes, &byte_idx, &bit_idx);
            }
            item.* = @as(i32, @intCast(a)) - @as(i32, @intCast(b));
        }
    }
};
