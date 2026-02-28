const std = @import("std");
const log = std.log;
const zml = @import("zml");
const c = @import("c");

const params = @import("params.zig");
const utils = @import("utils.zig");
const math = @import("math.zig");
const flatbuf_tools = @import("flatbuf_tools.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn commitPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    a_poly: []const u64,
    secret_seed: [32]u8,
    attempts: u32,
) !struct { y_poly: []const u64, w_poly: []const u64 } {
    var y_seed_input: [36]u8 = undefined;
    @memcpy(y_seed_input[0..32], &secret_seed);
    std.mem.writeInt(u32, y_seed_input[32..36], attempts, .little);

    const iter_seed = utils.extractSeed(&y_seed_input);
    const limit_y = 8192 * params.B;
    const y_poly = try math.sampleSmall(allocator, iter_seed, limit_y); // Mask

    // w = A * y mod Q
    const w_poly = try math.matVecMul(io, platform, allocator, mul_exe, a_poly, y_poly);

    return .{ .y_poly = y_poly, .w_poly = w_poly };
}

pub fn challengeResponsePhase(
    allocator: std.mem.Allocator,
    y_poly: []const u64,
    w_poly: []const u64,
    s_poly: []const u64,
) !struct { challenge: u64, z_poly: []const u64, is_safe: bool, actual_max: u64 } {
    var transcript = params.Transcript.init("LabradorZKP");
    transcript.appendMessage("w", std.mem.sliceAsBytes(w_poly));
    const challenge = transcript.getChallengeU64("c") % 2; // Binary challenge for simplicity/safety

    // z = y + c * s
    const z_poly = try math.vecAddScalarMul(allocator, y_poly, challenge, s_poly);

    const limit_y = 8192 * params.B;
    const max_c = 1;
    const beta = max_c * params.B;
    const safe_limit = limit_y - beta;

    var actual_max: u64 = 0;
    const is_safe = math.isSmall(z_poly, safe_limit, &actual_max);
    return .{ .challenge = challenge, .z_poly = z_poly, .is_safe = is_safe, .actual_max = actual_max };
}

pub fn verificationPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    a_poly: []const u64,
    t_poly: []const u64,
    w_poly: []const u64,
    z_poly: []const u64,
    challenge: u64,
) !bool {
    const limit_y = 8192 * params.B;

    // LHS = A * z mod Q
    const lhs = try math.matVecMul(io, platform, allocator, mul_exe, a_poly, z_poly);
    defer allocator.free(lhs);

    // RHS = w + c * t mod Q
    const rhs = try math.vecAddScalarMul(allocator, w_poly, challenge, t_poly);
    defer allocator.free(rhs);

    var is_equal = true;
    for (lhs, 0..) |val, i| {
        if (val != rhs[i]) {
            is_equal = false;
            log.err("Mismatch at {d}: LHS={d}, RHS={d}", .{ i, val, rhs[i] });
            if (i > 5) break;
        }
    }

    if (!is_equal) {
        log.err("FAILURE: Matrix equation does not hold.", .{});
        return false;
    } else if (!math.isSmall(z_poly, limit_y + params.B, null)) {
        log.err("FAILURE: Z vector too large.", .{});
        return false;
    }
    return true;
}

pub fn setupZmlEngine(
    allocator: std.mem.Allocator,
    io: std.Io,
) !struct { platform: *zml.Platform, mul_exe: zml.Exe } {
    log.info("Initializing ZML engine...", .{});
    const platform: *zml.Platform = try .auto(allocator, io, .{});

    // Compile matVecMul ZML graph once
    const Model = struct {
        pub fn forward(self: @This(), a: zml.Tensor, x: zml.Tensor) zml.Tensor {
            _ = self;
            const a_tagged = a.withTags(.{ .m, .n });
            const x_tagged = x.withTags(.{.n});
            const as_time = a_tagged.dot(x_tagged, .n);
            const q_tensor = zml.Tensor.scalar(params.ntt.Q, .u64).broad(as_time.shape());
            return as_time.remainder(q_tensor);
        }
    };
    log.info("Compiling ZML Graph for A * x mod Q...", .{});
    const mul_exe = try platform.compile(allocator, io, Model{}, .forward, .{ zml.Tensor.init(.{ params.ntt.M, params.ntt.N }, .u64), zml.Tensor.init(.{params.ntt.N}, .u64) });

    return .{ .platform = platform, .mul_exe = mul_exe };
}

pub fn registrationPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
) !struct { public_seed: [32]u8, a_poly: []const u64, secret_seed: [32]u8, s_poly: []const u64, t_poly: []const u64 } {
    const public_seed = utils.extractSeed("PublicSharedMatrixSeed");
    const a_poly = try math.expandA(allocator, public_seed);

    log.info("=== 1. REGISTRATION PHASE ===", .{});
    // Secret s
    const secret_seed = utils.extractSeed("UserPasswordSalt");
    const s_poly = try math.sampleSmall(allocator, secret_seed, params.B);

    // Public key t = A * s mod Q
    const t_poly = try math.matVecMul(io, platform, allocator, mul_exe, a_poly, s_poly);

    return .{
        .public_seed = public_seed,
        .a_poly = a_poly,
        .secret_seed = secret_seed,
        .s_poly = s_poly,
        .t_poly = t_poly,
    };
}

pub fn outputProofPhase(
    w_poly: []const u64,
    public_seed: [32]u8,
    t_poly: []const u64,
    attempts: u32,
    challenge: u64,
    z_poly: []const u64,
) !void {
    // --- Serialize proof as FlatBuffer ---
    log.info("=== FLATBUFFER SERIALIZATION ===", .{});
    var hash_transcript = params.Transcript.init("LabradorZKP");
    hash_transcript.appendMessage("w", std.mem.sliceAsBytes(w_poly));
    const transcript_hash = hash_transcript.getChallenge("hash");

    const fb_builder = std.heap.c_allocator.create(c.flatcc_builder_t) catch {
        log.err("FlatBuffer builder alloc failed", .{});
        return error.BuilderInitFailed;
    };
    defer std.heap.c_allocator.destroy(fb_builder);
    if (c.flatcc_builder_init(fb_builder) != 0) {
        log.err("FlatBuffer builder init failed", .{});
        return error.BuilderInitFailed;
    }
    defer c.flatcc_builder_clear(fb_builder);

    var buf_size: usize = 0;
    const proof_buf = flatbuf_tools.serializeProof(
        fb_builder,
        &public_seed,
        t_poly,
        w_poly,
        attempts,
        challenge,
        &transcript_hash,
        z_poly,
        true,
        &buf_size,
    ) catch |err| {
        log.err("FlatBuffer serialization failed: {}", .{err});
        return err;
    };
    defer std.c.free(proof_buf);

    log.info("[FlatBuf] Proof serialized: {d} bytes", .{buf_size});

    // --- Deserialize and verify round-trip ---
    log.info("=== FLATBUFFER DESERIALIZATION ===", .{});
    flatbuf_tools.deserializeAndLogProof(proof_buf);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var setup = try setupZmlEngine(allocator, io);
    defer setup.platform.deinit(allocator);
    defer setup.mul_exe.deinit();

    const registration = try registrationPhase(allocator, io, setup.platform, &setup.mul_exe);
    defer allocator.free(registration.a_poly);
    defer allocator.free(registration.s_poly);
    defer allocator.free(registration.t_poly);

    var proof_generated = false;
    var attempts: u32 = 0;

    log.info("Starting ZKP generation loop...", .{});
    while (!proof_generated) {
        attempts += 1;
        // 1. Prover: Commitment y and w
        const commitment = try commitPhase(allocator, io, setup.platform, &setup.mul_exe, registration.a_poly, registration.secret_seed, attempts);
        defer allocator.free(commitment.y_poly);
        defer allocator.free(commitment.w_poly);

        // 2 & 3. Verifier: Challenge & Prover: Response
        const challenge_response = try challengeResponsePhase(allocator, commitment.y_poly, commitment.w_poly, registration.s_poly);
        defer allocator.free(challenge_response.z_poly);

        if (challenge_response.is_safe) {
            log.info("--- Verification (Attempt {d}) ---", .{attempts});
            // 4. Verifier: Verify
            const is_verified = try verificationPhase(allocator, io, setup.platform, &setup.mul_exe, registration.a_poly, registration.t_poly, commitment.w_poly, challenge_response.z_poly, challenge_response.challenge);

            if (is_verified) {
                log.info("SUCCESS: Zero Knowledge Proof Accepted. (Attempts: {d})", .{attempts});
                proof_generated = true;

                try outputProofPhase(
                    commitment.w_poly,
                    registration.public_seed,
                    registration.t_poly,
                    attempts,
                    challenge_response.challenge,
                    challenge_response.z_poly,
                );
            } else {
                break;
            }
        } else {
            if (attempts % 10 == 0) {
                const limit_y = 8192 * params.B;
                const safe_limit = limit_y - (1 * params.B);
                log.info("... Client restarted protocol (Rejection Sampling Limit {d}, Max was {d}) ...", .{ safe_limit, resp_res.actual_max });
            }
        }
    }
}

test {
    _ = @import("utils.zig");
    _ = @import("math.zig");
    _ = @import("flatbuf_tools.zig");
}
