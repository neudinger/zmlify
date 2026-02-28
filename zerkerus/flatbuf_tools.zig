const std = @import("std");
const log = std.log;
const c_interface = @import("c");

pub const FlatbError = error{
    BuilderInitFailed,
    BufferCreationFailed,
};

/// Serialize a complete ZKP proof into a FlatBuffer.
/// The builder must be initialized by the caller.
/// Returns a finalized buffer (caller must free with c allocator) and its size.
pub fn serializeProof(
    builder: *c_interface.flatcc_builder_t,
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
    _ = c_interface.zerkerus_Registration_start(builder);
    _ = c_interface.zerkerus_Registration_public_seed_create(builder, public_seed.ptr, public_seed.len);
    _ = c_interface.zerkerus_Registration_public_key_t_create(builder, t_poly.ptr, t_poly.len);
    const registration = c_interface.zerkerus_Registration_end(builder);

    // --- Commitment table ---
    _ = c_interface.zerkerus_Commitment_start(builder);
    _ = c_interface.zerkerus_Commitment_attempt_add(builder, attempt);
    _ = c_interface.zerkerus_Commitment_w_create(builder, w_poly.ptr, w_poly.len);
    const commitment = c_interface.zerkerus_Commitment_end(builder);

    // --- Challenge table ---
    _ = c_interface.zerkerus_Challenge_start(builder);
    _ = c_interface.zerkerus_Challenge_c_add(builder, challenge);
    _ = c_interface.zerkerus_Challenge_transcript_hash_create(builder, transcript_hash.ptr, transcript_hash.len);
    const challenge_ref = c_interface.zerkerus_Challenge_end(builder);

    // --- Proof table (root) ---
    _ = c_interface.zerkerus_Proof_start(builder);
    _ = c_interface.zerkerus_Proof_registration_add(builder, registration);
    _ = c_interface.zerkerus_Proof_commitment_add(builder, commitment);
    _ = c_interface.zerkerus_Proof_challenge_add(builder, challenge_ref);
    _ = c_interface.zerkerus_Proof_verified_add(builder, @intFromBool(verified));
    _ = c_interface.zerkerus_Proof_z_create(builder, z_poly.ptr, z_poly.len);
    const proof = c_interface.zerkerus_Proof_end(builder);

    // Create buffer with root pointer framing (create_buffer returns ref_t: 0=failure)
    const buf_ref = c_interface.flatcc_builder_create_buffer(builder, null, 0, proof, 0, 0);
    if (buf_ref == 0) return FlatbError.BufferCreationFailed;

    out_size.* = 0;
    const buf = c_interface.flatcc_builder_finalize_buffer(builder, out_size);
    if (buf == null) return FlatbError.BufferCreationFailed;
    return buf.?;
}

/// Deserialize and print the contents of a Proof FlatBuffer for verification.
pub fn deserializeAndLogProof(buf: *const anyopaque) void {
    const proof = c_interface.zerkerus_Proof_as_root(buf);

    // Registration
    if (c_interface.zerkerus_Proof_registration_is_present(proof) != 0) {
        const reg = c_interface.zerkerus_Proof_registration(proof);
        const seed_vec = c_interface.zerkerus_Registration_public_seed(reg);
        const seed_len = c_interface.flatbuffers_uint8_vec_len(seed_vec);
        log.info("[FlatBuf] Registration: public_seed length = {d}", .{seed_len});

        const key_vec = c_interface.zerkerus_Registration_public_key_t(reg);
        const key_len = c_interface.flatbuffers_uint64_vec_len(key_vec);
        log.info("[FlatBuf] Registration: public_key_t length = {d}", .{key_len});
        if (key_len > 0) {
            log.info("[FlatBuf]   t[0] = {d}, t[{d}] = {d}", .{
                c_interface.flatbuffers_uint64_vec_at(key_vec, 0),
                key_len - 1,
                c_interface.flatbuffers_uint64_vec_at(key_vec, key_len - 1),
            });
        }
    }

    // Commitment
    if (c_interface.zerkerus_Proof_commitment_is_present(proof) != 0) {
        const com = c_interface.zerkerus_Proof_commitment(proof);
        const attempt = c_interface.zerkerus_Commitment_attempt(com);
        const w_vec = c_interface.zerkerus_Commitment_w(com);
        const w_len = c_interface.flatbuffers_uint64_vec_len(w_vec);
        log.info("[FlatBuf] Commitment: attempt = {d}, w length = {d}", .{ attempt, w_len });
    }

    // Challenge
    if (c_interface.zerkerus_Proof_challenge_is_present(proof) != 0) {
        const chal = c_interface.zerkerus_Proof_challenge(proof);
        const c_val = c_interface.zerkerus_Challenge_c(chal);
        const hash_vec = c_interface.zerkerus_Challenge_transcript_hash(chal);
        const hash_len = c_interface.flatbuffers_uint8_vec_len(hash_vec);
        log.info("[FlatBuf] Challenge: c = {d}, transcript_hash length = {d}", .{ c_val, hash_len });
    }

    // Response z
    if (c_interface.zerkerus_Proof_z_is_present(proof) != 0) {
        const z_vec = c_interface.zerkerus_Proof_z(proof);
        const z_len = c_interface.flatbuffers_uint64_vec_len(z_vec);
        log.info("[FlatBuf] Response: z length = {d}", .{z_len});
        if (z_len > 0) {
            log.info("[FlatBuf]   z[0] = {d}, z[{d}] = {d}", .{
                c_interface.flatbuffers_uint64_vec_at(z_vec, 0),
                z_len - 1,
                c_interface.flatbuffers_uint64_vec_at(z_vec, z_len - 1),
            });
        }
    }

    // Verified
    const verified = c_interface.zerkerus_Proof_verified(proof);
    log.info("[FlatBuf] Verified: {}", .{verified != 0});
}

test "flatbuf init builder" {
    const fb_builder = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);

    const init_res = c_interface.flatcc_builder_init(fb_builder);
    try std.testing.expectEqual(@as(c_int, 0), init_res);
    c_interface.flatcc_builder_clear(fb_builder);
}

test "serializeProof and deserializeAndLogProof" {
    const fb_builder = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);

    const init_res = c_interface.flatcc_builder_init(fb_builder);
    try std.testing.expectEqual(@as(c_int, 0), init_res);
    defer c_interface.flatcc_builder_clear(fb_builder);

    const public_seed = [_]u8{ 1, 2, 3, 4 };
    const t_poly = [_]u64{ 10, 20, 30 };
    const w_poly = [_]u64{ 100, 200, 300 };
    const transcript_hash = [_]u8{ 5, 6, 7, 8 };
    const z_poly = [_]u64{ 1000, 2000, 3000 };

    var buf_size: usize = 0;
    const proof_buf = try serializeProof(
        fb_builder,
        &public_seed,
        &t_poly,
        &w_poly,
        1,
        42,
        &transcript_hash,
        &z_poly,
        true,
        &buf_size,
    );
    defer std.c.free(proof_buf);

    try std.testing.expect(buf_size > 0);

    // Ensure we can deserialize without crashing
    deserializeAndLogProof(proof_buf);
}
