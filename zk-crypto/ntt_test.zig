const std = @import("std");
const zml = @import("zml");
const crypto = @import("crypto.zig");
const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 2048;
    pub const Q: u64 = 0xFFFFFFFF00000001; // Goldilocks prime
    pub const PSI: u64 = 17492915097719143606; // Primitive 4096-th root
    pub const ZETA: u64 = 455906449640507599; // Primitive 2048-th root
    pub const N_INV: u64 = 18437736870161940481; // Inverse of 2048 mod Q
});

const ButterflyNTT = ntt.ButterflyNTT;

// Larger NTT-friendly prime for ZML NTTModel tests (dot-product safe with N=1024)
const ntt_small = crypto.MakeNTT(struct {
    pub const N: usize = 1024;
    pub const Q: u64 = 104857601; // ~27-bit NTT-friendly prime (Q ≡ 1 mod 2048)
    pub const PSI: u64 = 4354736; // Primitive 2048-th root of unity mod Q
    pub const ZETA: u64 = 18773644; // Primitive 1024-th root of unity mod Q (PSI²)
    pub const N_INV: u64 = 104755201; // Inverse of 1024 mod Q
});

const NTTModel = ntt_small.NTTModel;

// =====================================================================
// ButterflyNTT Tests (Goldilocks prime, CPU-side)
// =====================================================================

fn testButterflyForward() !void {
    std.debug.print("Running testButterflyForward...\n", .{});

    var poly: [ntt.N]u64 = undefined;
    @memset(&poly, 1);

    ButterflyNTT.forward(&poly);

    // All values should be in range [0, Q)
    for (poly) |v| {
        try std.testing.expect(v < ntt.Q);
    }
}

fn testButterflyIdentity() !void {
    std.debug.print("Running testButterflyIdentity...\n", .{});

    var poly: [ntt.N]u64 = undefined;
    for (0..ntt.N) |i| {
        poly[i] = @as(u64, @intCast(i));
    }

    var original: [ntt.N]u64 = undefined;
    @memcpy(&original, &poly);

    ButterflyNTT.identity(&poly);

    for (0..ntt.N) |i| {
        try std.testing.expectEqual(original[i], poly[i]);
    }
}

fn testButterflyIdentityEdgeCases() !void {
    std.debug.print("Running testButterflyIdentityEdgeCases...\n", .{});

    var poly: [ntt.N]u64 = undefined;
    @memset(&poly, 0);
    poly[0] = 42;
    poly[1] = 100;
    poly[ntt.N - 1] = ntt.Q - 1;

    var original: [ntt.N]u64 = undefined;
    @memcpy(&original, &poly);

    ButterflyNTT.identity(&poly);

    for (0..ntt.N) |i| {
        try std.testing.expectEqual(original[i], poly[i]);
    }
}

fn testButterflyPolyMul() !void {
    std.debug.print("Running testButterflyPolyMul...\n", .{});

    // p(x) = 5, q(x) = 7  →  p·q = 35 (constant × constant)
    var a: [ntt.N]u64 = undefined;
    var b: [ntt.N]u64 = undefined;
    @memset(&a, 0);
    @memset(&b, 0);
    a[0] = 5;
    b[0] = 7;

    ButterflyNTT.forward(&a);
    ButterflyNTT.forward(&b);

    var result: [ntt.N]u64 = undefined;
    ButterflyNTT.polyMul(&a, &b, &result);

    ButterflyNTT.inverse(&result);

    try std.testing.expectEqual(@as(u64, 35), result[0]);
}

fn testButterflyBufferRoundTrip(allocator: std.mem.Allocator, platform: *zml.Platform, io: std.Io) !void {
    std.debug.print("Running testButterflyBufferRoundTrip...\n", .{});

    // Prepare input via ZML Buffer
    const poly_host = try allocator.alloc(u64, ntt.N);
    defer allocator.free(poly_host);
    for (0..ntt.N) |i| {
        poly_host[i] = @as(u64, @intCast(i));
    }

    var poly_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(
        zml.Shape.init(.{ntt.N}, .u64),
        std.mem.sliceAsBytes(poly_host),
    ));
    defer poly_buf.deinit();

    // Forward via ZML Buffer helper
    var fwd_buf = try ButterflyNTT.forwardBuffer(allocator, platform, io, poly_buf);
    defer fwd_buf.deinit();

    // Inverse via ZML Buffer helper
    var inv_buf = try ButterflyNTT.inverseBuffer(allocator, platform, io, fwd_buf);
    defer inv_buf.deinit();

    // Read back and verify identity
    const out_slice = try inv_buf.toSliceAlloc(allocator, io);
    defer out_slice.free(allocator);

    const out_data = out_slice.items(u64);
    for (0..ntt.N) |i| {
        try std.testing.expectEqual(poly_host[i], out_data[i]);
    }
}

// =====================================================================
// NTTModel Tests (Q=104857601, ZML graph compile/execute path)
// =====================================================================

fn testNTTModelDimensions(allocator: std.mem.Allocator, platform: *zml.Platform, io: std.Io) !void {
    std.debug.print("Running testNTTModelDimensions...\n", .{});

    var buffers = try NTTModel.init(allocator, platform, io);
    defer {
        buffers.F_fwd.deinit();
        buffers.F_inv.deinit();
    }

    try std.testing.expectEqualSlices(isize, &.{ ntt_small.N, ntt_small.N }, buffers.F_fwd.shape().dims());
    try std.testing.expectEqualSlices(isize, &.{ ntt_small.N, ntt_small.N }, buffers.F_inv.shape().dims());
}

fn testNTTModelForward(allocator: std.mem.Allocator, platform: *zml.Platform, io: std.Io) !void {
    std.debug.print("Running testNTTModelForward...\n", .{});

    // Compile the forward graph
    const ntt_def = NTTModel{
        .F_fwd = zml.Tensor.init(.{ ntt_small.N, ntt_small.N }, .u64).withTags(.{ ._, .c }),
        .F_inv = zml.Tensor.init(.{ ntt_small.N, ntt_small.N }, .u64).withTags(.{ ._, .c }),
    };

    const input_tensor = zml.Tensor.init(.{ntt_small.N}, .u64).withTags(.{.c});
    const exe = try platform.compile(allocator, io, ntt_def, .forward, .{input_tensor});
    defer exe.deinit();

    // Init buffers
    var buffers = try NTTModel.init(allocator, platform, io);
    defer {
        buffers.F_fwd.deinit();
        buffers.F_inv.deinit();
    }

    // Prepare all-ones input
    const poly_host = try allocator.alloc(u64, ntt_small.N);
    defer allocator.free(poly_host);
    @memset(poly_host, 1);

    var poly_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ntt_small.N}, .u64), std.mem.sliceAsBytes(poly_host)));
    defer poly_buf.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ buffers, poly_buf });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe.call(args, &results);
    var res_buf = results.get(zml.Buffer);
    defer res_buf.deinit();

    const output_slice = try res_buf.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);

    const out_data = output_slice.items(u64);
    try std.testing.expectEqual(ntt_small.N, out_data.len);

    // All values should be in [0, Q)
    for (out_data) |v| {
        try std.testing.expect(v < ntt_small.Q);
    }
}

fn testNTTModelIdentity(allocator: std.mem.Allocator, platform: *zml.Platform, io: std.Io) !void {
    std.debug.print("Running testNTTModelIdentity...\n", .{});

    // Compile the identity graph (forward then inverse)
    const ntt_def = NTTModel{
        .F_fwd = zml.Tensor.init(.{ ntt_small.N, ntt_small.N }, .u64).withTags(.{ ._, .c }),
        .F_inv = zml.Tensor.init(.{ ntt_small.N, ntt_small.N }, .u64).withTags(.{ ._, .c }),
    };

    const input_tensor = zml.Tensor.init(.{ntt_small.N}, .u64).withTags(.{.c});
    const exe = try platform.compile(allocator, io, ntt_def, .identity, .{input_tensor});
    defer exe.deinit();

    // Init buffers
    var buffers = try NTTModel.init(allocator, platform, io);
    defer {
        buffers.F_fwd.deinit();
        buffers.F_inv.deinit();
    }

    // Prepare input: 0, 1, 2, ..., N-1
    const poly_host = try allocator.alloc(u64, ntt_small.N);
    defer allocator.free(poly_host);
    for (0..ntt_small.N) |i| {
        poly_host[i] = @as(u64, @intCast(i));
    }

    var poly_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ntt_small.N}, .u64), std.mem.sliceAsBytes(poly_host)));
    defer poly_buf.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ buffers, poly_buf });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe.call(args, &results);
    var res_buf = results.get(zml.Buffer);
    defer res_buf.deinit();

    const output_slice = try res_buf.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);

    const out_data = output_slice.items(u64);
    for (0..ntt_small.N) |i| {
        try std.testing.expectEqual(poly_host[i], out_data[i]);
    }
}

// =====================================================================
// Cross-validation: ButterflyNTT vs NTTModel on the small prime
// =====================================================================

fn testCrossValidation(allocator: std.mem.Allocator, platform: *zml.Platform, io: std.Io) !void {
    std.debug.print("Running testCrossValidation...\n", .{});

    const SmallButterflyNTT = ntt_small.ButterflyNTT;

    // --- ButterflyNTT path ---
    var poly_butterfly: [ntt_small.N]u64 = undefined;
    for (0..ntt_small.N) |i| {
        poly_butterfly[i] = @as(u64, @intCast(i % ntt_small.Q));
    }
    SmallButterflyNTT.forward(&poly_butterfly);

    // --- NTTModel (ZML graph) path ---
    const ntt_def = NTTModel{
        .F_fwd = zml.Tensor.init(.{ ntt_small.N, ntt_small.N }, .u64).withTags(.{ ._, .c }),
        .F_inv = zml.Tensor.init(.{ ntt_small.N, ntt_small.N }, .u64).withTags(.{ ._, .c }),
    };
    const input_tensor = zml.Tensor.init(.{ntt_small.N}, .u64).withTags(.{.c});
    const exe = try platform.compile(allocator, io, ntt_def, .forward, .{input_tensor});
    defer exe.deinit();

    var buffers = try NTTModel.init(allocator, platform, io);
    defer {
        buffers.F_fwd.deinit();
        buffers.F_inv.deinit();
    }

    const poly_host = try allocator.alloc(u64, ntt_small.N);
    defer allocator.free(poly_host);
    for (0..ntt_small.N) |i| {
        poly_host[i] = @as(u64, @intCast(i % ntt_small.Q));
    }

    var poly_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ntt_small.N}, .u64), std.mem.sliceAsBytes(poly_host)));
    defer poly_buf.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ buffers, poly_buf });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe.call(args, &results);
    var res_buf = results.get(zml.Buffer);
    defer res_buf.deinit();

    const output_slice = try res_buf.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);

    const out_zml = output_slice.items(u64);

    // Both paths should produce identical NTT output
    for (0..ntt_small.N) |i| {
        try std.testing.expectEqual(poly_butterfly[i], out_zml[i]);
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    // ButterflyNTT tests (Goldilocks)
    try testButterflyForward();
    try testButterflyIdentity();
    try testButterflyIdentityEdgeCases();
    try testButterflyPolyMul();
    try testButterflyBufferRoundTrip(allocator, platform, io);

    // NTTModel tests (small prime, ZML graph path)
    try testNTTModelDimensions(allocator, platform, io);
    try testNTTModelForward(allocator, platform, io);
    try testNTTModelIdentity(allocator, platform, io);

    // Cross-validation: butterfly == matrix on small prime
    try testCrossValidation(allocator, platform, io);

    std.debug.print("All ZML NTT Tests Passed!\n", .{});
}
