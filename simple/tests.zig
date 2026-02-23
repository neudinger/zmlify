const std = @import("std");
const ref = @import("reference.zig");
const opt = @import("optimized.zig");

fn expectApproxEq(expected: []const f32, actual: []const f32, epsilon: f32) !void {
    if (expected.len != actual.len) return error.LengthMismatch;
    for (expected, actual) |e, a| {
        if (@abs(e - a) > epsilon) {
            std.log.err("Mismatch: expected {d}, got {d}", .{ e, a });
            return error.ValueMismatch;
        }
    }
}

fn expectEq(expected: []const i32, actual: []const i32) !void {
    if (expected.len != actual.len) return error.LengthMismatch;
    for (expected, actual) |e, a| {
        if (e != a) {
            std.log.err("Mismatch: expected {d}, got {d}", .{ e, a });
            return error.ValueMismatch;
        }
    }
}

test "matmul" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    const M = 16;
    const K = 16;
    const N = 16;

    const lhs_data = try allocator.alloc(f32, M * K);
    defer allocator.free(lhs_data);
    const rhs_data = try allocator.alloc(f32, K * N);
    defer allocator.free(rhs_data);
    const ref_out_matmul = try allocator.alloc(f32, M * N);
    defer allocator.free(ref_out_matmul);
    const opt_out_matmul = try allocator.alloc(f32, M * N);
    defer allocator.free(opt_out_matmul);

    for (lhs_data) |*v| v.* = random.float(f32);
    for (rhs_data) |*v| v.* = random.float(f32);

    ref.matmul(M, K, N, lhs_data, rhs_data, ref_out_matmul);
    opt.matmul(M, K, N, lhs_data, rhs_data, opt_out_matmul);
    try expectApproxEq(ref_out_matmul, opt_out_matmul, 1e-4);
}

test "saxpy" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    const SAXPY_N = 1024;
    const a_val: f32 = 2.0;

    const x_data = try allocator.alloc(f32, SAXPY_N);
    defer allocator.free(x_data);
    const y_data = try allocator.alloc(f32, SAXPY_N);
    defer allocator.free(y_data);
    const ref_out_saxpy = try allocator.alloc(f32, SAXPY_N);
    defer allocator.free(ref_out_saxpy);
    const opt_out_saxpy = try allocator.alloc(f32, SAXPY_N);
    defer allocator.free(opt_out_saxpy);

    for (x_data) |*v| v.* = random.float(f32);
    for (y_data) |*v| v.* = random.float(f32);
    ref.saxpy(SAXPY_N, a_val, x_data, y_data, ref_out_saxpy);
    opt.saxpy(SAXPY_N, a_val, x_data, y_data, opt_out_saxpy);
    try expectApproxEq(ref_out_saxpy, opt_out_saxpy, 1e-4);
}

test "mod_matmul" {
    const allocator = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    const MOD_K = 16;

    const lhs_mod_data = try allocator.alloc(i32, MOD_K * MOD_K);
    defer allocator.free(lhs_mod_data);
    const rhs_mod_data = try allocator.alloc(i32, MOD_K * MOD_K);
    defer allocator.free(rhs_mod_data);
    const ref_out_mod = try allocator.alloc(i32, MOD_K * MOD_K);
    defer allocator.free(ref_out_mod);
    const opt_out_mod = try allocator.alloc(i32, MOD_K * MOD_K);
    defer allocator.free(opt_out_mod);

    for (lhs_mod_data) |*v| v.* = @intCast(random.intRangeAtMost(i32, 0, 100));
    for (rhs_mod_data) |*v| v.* = @intCast(random.intRangeAtMost(i32, 0, 100));

    ref.mod_matmul(MOD_K, MOD_K, MOD_K, lhs_mod_data, rhs_mod_data, ref_out_mod);
    opt.mod_matmul(MOD_K, MOD_K, MOD_K, lhs_mod_data, rhs_mod_data, opt_out_mod);
    try expectEq(ref_out_mod, opt_out_mod);

    // Verify ModMatMul logic with a small known example
    const MOD_TEST_N = 2;
    const lhs_small = try allocator.alloc(i32, MOD_TEST_N * MOD_TEST_N);
    defer allocator.free(lhs_small);
    const rhs_small = try allocator.alloc(i32, MOD_TEST_N * MOD_TEST_N);
    defer allocator.free(rhs_small);
    const out_small = try allocator.alloc(i32, MOD_TEST_N * MOD_TEST_N);
    defer allocator.free(out_small);
    const out_small_opt = try allocator.alloc(i32, MOD_TEST_N * MOD_TEST_N);
    defer allocator.free(out_small_opt);

    // LHS: [[1, 2], [3, 4]]
    lhs_small[0] = 1;
    lhs_small[1] = 2;
    lhs_small[2] = 3;
    lhs_small[3] = 4;

    // RHS: [[5, 6], [7, 8]]
    rhs_small[0] = 5;
    rhs_small[1] = 6;
    rhs_small[2] = 7;
    rhs_small[3] = 8;

    // Expected: [[19, 22], [43, 50]]

    ref.mod_matmul(MOD_TEST_N, MOD_TEST_N, MOD_TEST_N, lhs_small, rhs_small, out_small);
    opt.mod_matmul(MOD_TEST_N, MOD_TEST_N, MOD_TEST_N, lhs_small, rhs_small, out_small_opt);

    try std.testing.expectEqual(@as(i32, 19), out_small[0]);
    try std.testing.expectEqual(@as(i32, 22), out_small[1]);
    try std.testing.expectEqual(@as(i32, 43), out_small[2]);
    try std.testing.expectEqual(@as(i32, 50), out_small[3]);
}

test "heat_transfer" {
    const allocator = std.testing.allocator;
    const H = 4;
    const W = 4;

    const grid = try allocator.alloc(f32, H * W);
    defer allocator.free(grid);
    @memset(grid, 0);
    for (0..W) |j| grid[j] = 100.0;

    const ref_out = try allocator.alloc(f32, H * W);
    defer allocator.free(ref_out);
    const opt_out = try allocator.alloc(f32, H * W);
    defer allocator.free(opt_out);

    ref.heat_transfer(H, W, grid, ref_out);
    opt.heat_transfer(H, W, grid, opt_out);

    try expectApproxEq(ref_out, opt_out, 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 25.0), ref_out[1 * W + 1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), ref_out[1 * W + 2], 1e-4);
}
