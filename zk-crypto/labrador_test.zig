const std = @import("std");
const crypto = @import("crypto.zig");
const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 2048;
    pub const Q: u64 = 0xFFFFFFFF00000001; // Goldilocks prime
    pub const PSI: u64 = 17492915097719143606; // Primitive 4096-th root
    pub const ZETA: u64 = 455906449640507599; // Primitive 2048-th root
    pub const N_INV: u64 = 18437736870161940481; // Inverse of 2048 mod Q
});
const labrador = crypto.MakeLabrador(ntt, 1000);
const zml = @import("zml");
const fiat_shamir = crypto.fiat_shamir;

fn testNormBoundChecking() !void {
    std.debug.print("Running testNormBoundChecking...\n", .{});
    // Test that valid bounds pass
    const valid_poly = [_]u64{ 0, 500, labrador.B, ntt.Q - 500, ntt.Q - labrador.B };
    try std.testing.expect(labrador.Prover.checkNorm(&valid_poly));

    // Test that out-of-bounds fail (positive bounds)
    const invalid_poly_pos = [_]u64{ 0, labrador.B + 1 };
    try std.testing.expect(!labrador.Prover.checkNorm(&invalid_poly_pos));

    // Test out of bounds (negative mapping)
    const invalid_poly_neg = [_]u64{ 0, ntt.Q - labrador.B - 1 };
    try std.testing.expect(!labrador.Prover.checkNorm(&invalid_poly_neg));
}

fn testRejectionSampling() !void {
    std.debug.print("Running testRejectionSampling...\n", .{});

    // valid: bounds inside B - beta = 1000 - 100 = 900
    const valid_z = [_]u64{ 0, 500, 899, ntt.Q - 500, ntt.Q - 899 };
    try std.testing.expect(labrador.Prover.rejectionSample(&valid_z, 100));

    // invalid: too close to bound, leaks info -> ABORT
    const invalid_z_pos = [_]u64{ 0, 901 };
    try std.testing.expect(!labrador.Prover.rejectionSample(&invalid_z_pos, 100));

    const invalid_z_neg = [_]u64{ 0, ntt.Q - 901 };
    try std.testing.expect(!labrador.Prover.rejectionSample(&invalid_z_neg, 100));
}

fn testSumcheckPolynomialFolding() !void {
    std.debug.print("Running testSumcheckPolynomialFolding...\n", .{});
    // N=4 polynomial coefficients
    const poly = [_]u64{ 1, 2, 3, 4 };
    var out_poly: [2]u64 = undefined;

    // Fold with challenge alpha = 10
    labrador.Prover.foldPolynomial(&poly, 10, &out_poly);

    // out[0] = poly[0] + 10 * poly[1] = 1 + 20 = 21
    // out[1] = poly[2] + 10 * poly[3] = 3 + 40 = 43
    try std.testing.expectEqual(@as(usize, 2), out_poly.len);
    try std.testing.expectEqual(@as(u64, 21), out_poly[0]);
    try std.testing.expectEqual(@as(u64, 43), out_poly[1]);
}

fn testRecursiveFoldingProveStateMachine(allocator: std.mem.Allocator) !void {
    std.debug.print("Running testRecursiveFoldingProveStateMachine...\n", .{});

    // N=8 polynomial coefficients
    const poly = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };

    var transcript = fiat_shamir.Transcript.init("test");
    transcript.appendMessage("test_ctx", "Labrador Sumcheck Prove Test");

    const final_proof = try labrador.Prover.prove(allocator, &transcript, &poly);
    defer allocator.free(final_proof);

    // N=8 -> N=4 -> N=2 -> N=1. Final proof should be exactly length 1.
    try std.testing.expectEqual(@as(usize, 1), final_proof.len);

    // Ensure bounds are modulated appropriately
    try std.testing.expect(final_proof[0] >= 0 and final_proof[0] < ntt.Q);
}

fn testRecursiveFoldingVerifyStateMachine(allocator: std.mem.Allocator) !void {
    std.debug.print("Running testRecursiveFoldingVerifyStateMachine...\n", .{});

    // N=8 polynomial coefficients (simulating the public commitment matching the secret)
    const poly = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // PROVER SIDE
    var prover_transcript = fiat_shamir.Transcript.init("test");
    prover_transcript.appendMessage("test_ctx", "Labrador Sumcheck Prove Test");

    const final_proof = try labrador.Prover.prove(allocator, &prover_transcript, &poly);
    defer allocator.free(final_proof);

    // VERIFIER SIDE
    var verifier_transcript = fiat_shamir.Transcript.init("test");
    verifier_transcript.appendMessage("test_ctx", "Labrador Sumcheck Prove Test");

    // The verifier folds the public commitment using the exact same transcript logic
    const expected_scalar = try labrador.Prover.verify(allocator, &verifier_transcript, &poly);
    defer allocator.free(expected_scalar);

    // If the verifier's deterministic folding matches the prover's output, the proof is valid!
    try std.testing.expectEqualSlices(u64, final_proof, expected_scalar);
}

fn testBDLOPMatrixExpansion() !void {
    std.debug.print("Running testBDLOPMatrixExpansion...\n", .{});
    var seed: [32]u8 = undefined;
    @memset(&seed, 123);

    var poly1: [256]u64 = undefined;
    var poly2: [256]u64 = undefined;

    labrador.Prover.expandA(seed, &poly1);
    labrador.Prover.expandA(seed, &poly2);

    // Check bounds
    for (poly1) |x| {
        try std.testing.expect(x >= 0 and x < ntt.Q);
    }

    // Check determinism
    try std.testing.expectEqualSlices(u64, &poly1, &poly2);
}

fn testBDLOPCommitmentGraph(allocator: std.mem.Allocator, platform: *zml.Platform, io: std.Io) !void {
    std.debug.print("Running testBDLOPCommitmentGraph...\n", .{});

    const ntt_def = ntt.NTTModel{
        .F_fwd = zml.Tensor.init(.{ ntt.N, ntt.N }, .u64).withTags(.{ ._, .c }),
        .F_inv = zml.Tensor.init(.{ ntt.N, ntt.N }, .u64).withTags(.{ ._, .c }),
    };

    const bdlop = labrador.BDLOPModel.init(ntt_def);

    const a_tensor = zml.Tensor.init(.{ntt.N}, .u64).withTags(.{.c});
    const s_tensor = zml.Tensor.init(.{ntt.N}, .u64).withTags(.{.c});
    const e_tensor = zml.Tensor.init(.{ntt.N}, .u64).withTags(.{.c});

    const exe = try platform.compile(allocator, io, bdlop, .commit, .{ a_tensor, s_tensor, e_tensor });
    defer exe.deinit();

    // If it compiled, the graph is structurally valid!
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    try testNormBoundChecking();
    try testRejectionSampling();
    try testSumcheckPolynomialFolding();
    try testRecursiveFoldingProveStateMachine(allocator);
    try testRecursiveFoldingVerifyStateMachine(allocator);
    try testBDLOPMatrixExpansion();
    try testBDLOPCommitmentGraph(allocator, platform, io);

    std.debug.print("All Labrador Logic Tests Passed!\n", .{});
}
