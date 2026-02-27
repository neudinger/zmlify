const std = @import("std");
const log = std.log;
const zml = @import("zml");
const c = @import("c");

const crypto = @import("zk-crypto");
const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 1024;
    pub const M: usize = 256;
    pub const Q: u64 = 8380417; // Kyber/Dilithium prime
    pub const PSI: u64 = 17492915097719143606; // Primitive 4096-th root of unity modulo Q
    pub const ZETA: u64 = 455906449640507599; // Primitive 2048-th root modulo Q (PSI^2)
    pub const N_INV: u64 = 18437736870161940481; // Inverse of 2048 mod Q
});
const NTTModel = ntt.NTTModel;
const Transcript = crypto.fiat_shamir.Transcript;
const labrador = crypto.MakeLabrador(ntt, 1000);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const B: u64 = 1000; // Infinity norm bound for small vectors

// --- Math Helpers ---
fn extractSeed(data: []const u8) [32]u8 {
    var hasher = std.crypto.hash.Blake3.init(.{});
    hasher.update(data);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

fn sampleSmall(allocator: std.mem.Allocator, seed: [32]u8, bound: u64) ![]u64 {
    var seed_slice: [8]u8 = undefined;
    @memcpy(&seed_slice, seed[0..8]);
    var prng = std.Random.DefaultPrng.init(std.mem.readInt(u64, &seed_slice, .little));
    const random = prng.random();

    const poly = try allocator.alloc(u64, ntt.N);
    for (poly) |*v| {
        const val = random.intRangeLessThan(u64, 0, 2 * bound + 1);
        if (val <= bound) {
            v.* = val;
        } else {
            v.* = ntt.Q - (val - bound);
        }
    }
    return poly;
}

fn expandA(allocator: std.mem.Allocator, seed: [32]u8) ![]u64 {
    const poly = try allocator.alloc(u64, ntt.M * ntt.N);
    labrador.Prover.expandA(seed, poly);
    return poly;
}

// Check if vector is within [-limit, limit]
fn isSmall(poly: []const u64, limit: u64, out_max: ?*u64) bool {
    const q_half = ntt.Q / 2;
    var max_val: u64 = 0;
    for (poly) |v| {
        const abs_v = if (v > q_half) ntt.Q - v else v;
        if (abs_v > max_val) {
            max_val = abs_v;
        }
    }
    if (out_max) |m| {
        m.* = max_val;
    }
    return max_val <= limit;
}

// --- FlatBuffers Serialization (via flatcc C API) ---

const FlatbError = error{
    BuilderInitFailed,
    BufferCreationFailed,
};

/// Serialize a complete ZKP proof into a FlatBuffer.
/// The builder must be initialized by the caller.
/// Returns a finalized buffer (caller must free with c allocator) and its size.
fn serializeProof(
    builder: *c.flatcc_builder_t,
    public_seed: []const u8,
    t_poly: []const u64,
    w_poly: []const u64,
    attempt: u32,
    challenge: u64,
    transcript_hash: []const u8,
    z_poly: []const u64,
    verified: bool,
    out_size: *usize,
) FlatbError!*anyopaque {
    // --- Registration table (vectors inline via _create) ---
    _ = c.zerkerus_Registration_start(builder);
    _ = c.zerkerus_Registration_public_seed_create(builder, public_seed.ptr, public_seed.len);
    _ = c.zerkerus_Registration_public_key_t_create(builder, t_poly.ptr, t_poly.len);
    const registration = c.zerkerus_Registration_end(builder);

    // --- Commitment table ---
    _ = c.zerkerus_Commitment_start(builder);
    _ = c.zerkerus_Commitment_attempt_add(builder, attempt);
    _ = c.zerkerus_Commitment_w_create(builder, w_poly.ptr, w_poly.len);
    const commitment = c.zerkerus_Commitment_end(builder);

    // --- Challenge table ---
    _ = c.zerkerus_Challenge_start(builder);
    _ = c.zerkerus_Challenge_c_add(builder, challenge);
    _ = c.zerkerus_Challenge_transcript_hash_create(builder, transcript_hash.ptr, transcript_hash.len);
    const challenge_ref = c.zerkerus_Challenge_end(builder);

    // --- Proof table (root) ---
    _ = c.zerkerus_Proof_start(builder);
    _ = c.zerkerus_Proof_registration_add(builder, registration);
    _ = c.zerkerus_Proof_commitment_add(builder, commitment);
    _ = c.zerkerus_Proof_challenge_add(builder, challenge_ref);
    _ = c.zerkerus_Proof_verified_add(builder, @intFromBool(verified));
    _ = c.zerkerus_Proof_z_create(builder, z_poly.ptr, z_poly.len);
    const proof = c.zerkerus_Proof_end(builder);

    // Create buffer with root pointer framing (create_buffer returns ref_t: 0=failure)
    const buf_ref = c.flatcc_builder_create_buffer(builder, null, 0, proof, 0, 0);
    if (buf_ref == 0) return FlatbError.BufferCreationFailed;

    out_size.* = 0;
    const buf = c.flatcc_builder_finalize_buffer(builder, out_size);
    if (buf == null) return FlatbError.BufferCreationFailed;
    return buf.?;
}

/// Deserialize and print the contents of a Proof FlatBuffer for verification.
fn deserializeAndLogProof(buf: *const anyopaque) void {
    const proof = c.zerkerus_Proof_as_root(buf);

    // Registration
    if (c.zerkerus_Proof_registration_is_present(proof) != 0) {
        const reg = c.zerkerus_Proof_registration(proof);
        const seed_vec = c.zerkerus_Registration_public_seed(reg);
        const seed_len = c.flatbuffers_uint8_vec_len(seed_vec);
        log.info("[FlatBuf] Registration: public_seed length = {d}", .{seed_len});

        const key_vec = c.zerkerus_Registration_public_key_t(reg);
        const key_len = c.flatbuffers_uint64_vec_len(key_vec);
        log.info("[FlatBuf] Registration: public_key_t length = {d}", .{key_len});
        if (key_len > 0) {
            log.info("[FlatBuf]   t[0] = {d}, t[{d}] = {d}", .{
                c.flatbuffers_uint64_vec_at(key_vec, 0),
                key_len - 1,
                c.flatbuffers_uint64_vec_at(key_vec, key_len - 1),
            });
        }
    }

    // Commitment
    if (c.zerkerus_Proof_commitment_is_present(proof) != 0) {
        const com = c.zerkerus_Proof_commitment(proof);
        const attempt = c.zerkerus_Commitment_attempt(com);
        const w_vec = c.zerkerus_Commitment_w(com);
        const w_len = c.flatbuffers_uint64_vec_len(w_vec);
        log.info("[FlatBuf] Commitment: attempt = {d}, w length = {d}", .{ attempt, w_len });
    }

    // Challenge
    if (c.zerkerus_Proof_challenge_is_present(proof) != 0) {
        const chal = c.zerkerus_Proof_challenge(proof);
        const c_val = c.zerkerus_Challenge_c(chal);
        const hash_vec = c.zerkerus_Challenge_transcript_hash(chal);
        const hash_len = c.flatbuffers_uint8_vec_len(hash_vec);
        log.info("[FlatBuf] Challenge: c = {d}, transcript_hash length = {d}", .{ c_val, hash_len });
    }

    // Response z
    if (c.zerkerus_Proof_z_is_present(proof) != 0) {
        const z_vec = c.zerkerus_Proof_z(proof);
        const z_len = c.flatbuffers_uint64_vec_len(z_vec);
        log.info("[FlatBuf] Response: z length = {d}", .{z_len});
        if (z_len > 0) {
            log.info("[FlatBuf]   z[0] = {d}, z[{d}] = {d}", .{
                c.flatbuffers_uint64_vec_at(z_vec, 0),
                z_len - 1,
                c.flatbuffers_uint64_vec_at(z_vec, z_len - 1),
            });
        }
    }

    // Verified
    const verified = c.zerkerus_Proof_verified(proof);
    log.info("[FlatBuf] Verified: {}", .{verified != 0});
}

// Compute A * x mod Q in ZML
fn matVecMul(io: std.Io, platform: *const zml.Platform, allocator: std.mem.Allocator, exe: *zml.Exe, a_mat: []const u64, x_poly: []const u64) ![]u64 {
    var a_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ ntt.M, ntt.N }, .u64), std.mem.sliceAsBytes(a_mat)));
    defer a_buf.deinit();

    var x_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ntt.N}, .u64), std.mem.sliceAsBytes(x_poly)));
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

    const result_poly = try allocator.alloc(u64, ntt.M);
    @memcpy(result_poly, out_slice.items(u64));
    return result_poly;
}

// Compute (y + c * s) mod Q or (w + c * t) mod Q in pure zig since scalar operations are fast
fn vecAddScalarMul(allocator: std.mem.Allocator, y: []const u64, c_val: u64, s: []const u64) ![]u64 {
    const len = y.len; // Supports both N and M sized vectors
    const result = try allocator.alloc(u64, len);
    for (y, 0..) |yi, i| {
        const cs = (@as(u128, c_val) * @as(u128, s[i])) % ntt.Q;
        result[i] = @as(u64, @intCast((@as(u128, yi) + cs) % ntt.Q));
    }
    return result;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    log.info("Initializing ZML engine...", .{});
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    // Compile matVecMul ZML graph once
    const Model = struct {
        pub fn forward(self: @This(), a: zml.Tensor, x: zml.Tensor) zml.Tensor {
            _ = self;
            const a_tagged = a.withTags(.{ .m, .n });
            const x_tagged = x.withTags(.{.n});
            const as_time = a_tagged.dot(x_tagged, .n);
            const q_tensor = zml.Tensor.scalar(ntt.Q, .u64).broad(as_time.shape());
            return as_time.remainder(q_tensor);
        }
    };
    log.info("Compiling ZML Graph for A * x mod Q...", .{});
    var mul_exe = try platform.compile(allocator, io, Model{}, .forward, .{ zml.Tensor.init(.{ ntt.M, ntt.N }, .u64), zml.Tensor.init(.{ntt.N}, .u64) });
    defer mul_exe.deinit();

    const public_seed = extractSeed("PublicSharedMatrixSeed");
    const a_poly = try expandA(allocator, public_seed);
    defer allocator.free(a_poly);

    log.info("=== 1. REGISTRATION PHASE ===", .{});
    // Secret s
    const secret_seed = extractSeed("UserPasswordSalt");
    const s_poly = try sampleSmall(allocator, secret_seed, B);
    defer allocator.free(s_poly);

    // Public key t = A * s mod Q
    const t_poly = try matVecMul(io, platform, allocator, &mul_exe, a_poly, s_poly);
    defer allocator.free(t_poly);

    var proof_generated = false;
    var attempts: u32 = 0;

    log.info("Starting ZKP generation loop...", .{});
    while (!proof_generated) {
        attempts += 1;
        // 1. Prover: Commitment y and w
        var y_seed_input: [36]u8 = undefined;
        @memcpy(y_seed_input[0..32], &secret_seed);
        std.mem.writeInt(u32, y_seed_input[32..36], attempts, .little);

        const iter_seed = extractSeed(&y_seed_input);
        const limit_y = 8192 * B;
        const y_poly = try sampleSmall(allocator, iter_seed, limit_y); // Mask
        defer allocator.free(y_poly);

        // w = A * y mod Q
        const w_poly = try matVecMul(io, platform, allocator, &mul_exe, a_poly, y_poly);
        defer allocator.free(w_poly);

        // 2. Verifier: Challenge c
        var transcript = Transcript.init("LabradorZKP");
        transcript.appendMessage("w", std.mem.sliceAsBytes(w_poly));
        const challenge = transcript.getChallengeU64("c") % 2; // Binary challenge for simplicity/safety

        // 3. Prover: Response z
        // z = y + c * s
        const z_poly = try vecAddScalarMul(allocator, y_poly, challenge, s_poly);
        defer allocator.free(z_poly);

        // Security check: bounds of z
        // Rejection sample if z leaks information about s
        const max_c = 1;
        const beta = max_c * B;
        const safe_limit = limit_y - beta;

        var actual_max: u64 = 0;
        if (isSmall(z_poly, safe_limit, &actual_max)) {
            // 4. Verifier: Verify
            log.info("--- Verification (Attempt {d}) ---", .{attempts});

            // LHS = A * z mod Q
            const lhs = try matVecMul(io, platform, allocator, &mul_exe, a_poly, z_poly);
            defer allocator.free(lhs);

            // RHS = w + c * t mod Q
            const rhs = try vecAddScalarMul(allocator, w_poly, challenge, t_poly);
            defer allocator.free(rhs);

            var is_equal = true;
            for (lhs, 0..) |val, i| {
                if (val != rhs[i]) {
                    is_equal = false;
                    log.err("Mismatch at {d}: LHS={d}, RHS={d}", .{ i, val, rhs[i] });
                    if (i > 5) break; // just print a few
                }
            }

            if (!is_equal) {
                log.err("FAILURE: Matrix equation does not hold.", .{});
                break;
            } else if (!isSmall(z_poly, limit_y + B, null)) {
                log.err("FAILURE: Z vector too large.", .{});
                break;
            } else {
                log.info("SUCCESS: Zero Knowledge Proof Accepted. (Attempts: {d})", .{attempts});
                proof_generated = true;

                // --- Serialize proof as FlatBuffer ---
                log.info("=== FLATBUFFER SERIALIZATION ===", .{});
                var hash_transcript = Transcript.init("LabradorZKP");
                hash_transcript.appendMessage("w", std.mem.sliceAsBytes(w_poly));
                const transcript_hash = hash_transcript.getChallenge("hash");

                const fb_builder = std.heap.c_allocator.create(c.flatcc_builder_t) catch {
                    log.err("FlatBuffer builder alloc failed", .{});
                    break;
                };
                defer std.heap.c_allocator.destroy(fb_builder);
                if (c.flatcc_builder_init(fb_builder) != 0) {
                    log.err("FlatBuffer builder init failed", .{});
                    break;
                }
                defer c.flatcc_builder_clear(fb_builder);

                var buf_size: usize = 0;
                const proof_buf = serializeProof(
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
                    break;
                };
                defer std.c.free(proof_buf);

                log.info("[FlatBuf] Proof serialized: {d} bytes", .{buf_size});

                // --- Deserialize and verify round-trip ---
                log.info("=== FLATBUFFER DESERIALIZATION ===", .{});
                deserializeAndLogProof(proof_buf);
            }
        } else {
            if (attempts % 10 == 0) {
                log.info("... Client restarted protocol (Rejection Sampling Limit {d}, Max was {d}) ...", .{ safe_limit, actual_max });
            }
        }
    }
}
