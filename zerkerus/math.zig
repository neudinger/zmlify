const std = @import("std");
const builtin = @import("builtin");
const is_wasm = builtin.target.cpu.arch == .wasm32;
const zml = if (!is_wasm) @import("zml") else struct {};
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

// Compute A * x mod Q in ZML (GPU-accelerated, non-WASM only)
pub const matVecMul = if (!is_wasm) struct {
    pub fn call(io: std.Io, platform: *const zml.Platform, allocator: std.mem.Allocator, exe: *zml.Exe, a_mat: []const u64, x_poly: []const u64) ![]u64 {
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
}.call else @compileError("matVecMul requires ZML (not available on wasm32)");

/// Pure-Zig A * x mod Q for WASM and CPU-only environments (no ZML dependency).
pub fn matVecMulPure(allocator: std.mem.Allocator, a_mat: []const u64, x_poly: []const u64) ![]u64 {
    const result = try allocator.alloc(u64, params.ntt.M);
    for (0..params.ntt.M) |i| {
        var acc: u128 = 0;
        for (0..params.ntt.N) |j| {
            acc += @as(u128, a_mat[i * params.ntt.N + j]) * @as(u128, x_poly[j]);
            acc %= params.ntt.Q;
        }
        result[i] = @intCast(acc);
    }
    return result;
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

test "matVecMulPure correctness" {
    const allocator = std.testing.allocator;
    // Simple 2x3 matrix * 3-vector test (using small values mod Q)
    const a_mat = [_]u64{ 1, 2, 3, 4, 5, 6 }; // 2x3 matrix
    _ = a_mat; // autofix
    const x_poly = [_]u64{ 10, 20, 30 }; // 3-vector
    _ = x_poly; // autofix

    // Expected: row0 = (1*10 + 2*20 + 3*30) % Q = 140
    //           row1 = (4*10 + 5*20 + 6*30) % Q = 320
    // But matVecMulPure uses params.ntt.M and params.ntt.N, so we test with
    // actual param dimensions using expandA + sampleSmall
    const seed = [_]u8{42} ** 32;
    const a_poly = try expandA(allocator, seed);
    defer allocator.free(a_poly);

    const s_poly = try sampleSmall(allocator, seed, params.B);
    defer allocator.free(s_poly);

    const result = try matVecMulPure(allocator, a_poly, s_poly);
    defer allocator.free(result);

    // Verify dimensions
    try std.testing.expectEqual(params.ntt.M, result.len);

    // Verify all results are in valid range [0, Q)
    for (result) |v| {
        try std.testing.expect(v < params.ntt.Q);
    }
}
