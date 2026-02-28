const std = @import("std");
const zml = @import("zml");
const params = @import("params.zig");

pub fn sampleSmall(allocator: std.mem.Allocator, seed: [32]u8, bound: u64) ![]u64 {
    var seed_slice: [8]u8 = undefined;
    @memcpy(&seed_slice, seed[0..8]);
    var prng = std.Random.DefaultPrng.init(std.mem.readInt(u64, &seed_slice, .little));
    const random = prng.random();

    const poly = try allocator.alloc(u64, params.ntt.N);
    for (poly) |*v| {
        const val = random.intRangeLessThan(u64, 0, 2 * bound + 1);
        if (val <= bound) {
            v.* = val;
        } else {
            v.* = params.ntt.Q - (val - bound);
        }
    }
    return poly;
}

pub fn expandA(allocator: std.mem.Allocator, seed: [32]u8) ![]u64 {
    const poly = try allocator.alloc(u64, params.ntt.M * params.ntt.N);
    params.labrador.Prover.expandA(seed, poly);
    return poly;
}

// Check if vector is within [-limit, limit]
pub fn isSmall(poly: []const u64, limit: u64, out_max: ?*u64) bool {
    const q_half = params.ntt.Q / 2;
    var max_val: u64 = 0;
    for (poly) |v| {
        const abs_v = if (v > q_half) params.ntt.Q - v else v;
        if (abs_v > max_val) {
            max_val = abs_v;
        }
    }
    if (out_max) |m| {
        m.* = max_val;
    }
    return max_val <= limit;
}

// Compute A * x mod Q in ZML
pub fn matVecMul(io: std.Io, platform: *const zml.Platform, allocator: std.mem.Allocator, exe: *zml.Exe, a_mat: []const u64, x_poly: []const u64) ![]u64 {
    var a_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ params.ntt.M, params.ntt.N }, .u64), std.mem.sliceAsBytes(a_mat)));
    defer a_buf.deinit();

    var x_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{params.ntt.N}, .u64), std.mem.sliceAsBytes(x_poly)));
    defer x_buf.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ a_buf, x_buf });
    exe.call(args, &results);

    var res_buf = results.get(zml.Buffer);
    defer res_buf.deinit();

    const out_slice = try res_buf.toSliceAlloc(allocator, io);
    defer out_slice.free(allocator);

    const result_poly = try allocator.alloc(u64, params.ntt.M);
    @memcpy(result_poly, out_slice.items(u64));
    return result_poly;
}

// Compute (y + c * s) mod Q or (w + c * t) mod Q in pure zig since scalar operations are fast
pub fn vecAddScalarMul(allocator: std.mem.Allocator, y: []const u64, c_val: u64, s: []const u64) ![]u64 {
    const len = y.len; // Supports both N and M sized vectors
    const result = try allocator.alloc(u64, len);
    for (y, 0..) |yi, i| {
        const cs = (@as(u128, c_val) * @as(u128, s[i])) % params.ntt.Q;
        result[i] = @as(u64, @intCast((@as(u128, yi) + cs) % params.ntt.Q));
    }
    return result;
}

test "sampleSmall respects bounds" {
    const allocator = std.testing.allocator;
    const seed: [32]u8 = [_]u8{0} ** 32;
    const poly = try sampleSmall(allocator, seed, params.B);
    defer allocator.free(poly);

    try std.testing.expectEqual(params.ntt.N, poly.len);
    try std.testing.expect(isSmall(poly, params.B, null));
}

test "vecAddScalarMul correctness" {
    const allocator = std.testing.allocator;
    const y = [_]u64{ 1, params.ntt.Q - 2, 3 };
    const s = [_]u64{ 4, 5, 6 };
    const c_val: u64 = 2;

    const result = try vecAddScalarMul(allocator, &y, c_val, &s);
    defer allocator.free(result);

    try std.testing.expectEqual(y.len, result.len);
    try std.testing.expectEqual((1 + 2 * 4) % params.ntt.Q, result[0]);
    try std.testing.expectEqual((params.ntt.Q - 2 + 2 * 5) % params.ntt.Q, result[1]);
    try std.testing.expectEqual((3 + 2 * 6) % params.ntt.Q, result[2]);
}

test "isSmall logic" {
    const poly1 = [_]u64{ 10, 0, params.ntt.Q - 10 };
    try std.testing.expect(isSmall(&poly1, 10, null));
    try std.testing.expect(!isSmall(&poly1, 9, null));

    var max_val: u64 = 0;
    _ = isSmall(&poly1, 10, &max_val);
    try std.testing.expectEqual(@as(u64, 10), max_val);
}
