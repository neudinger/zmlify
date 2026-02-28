const std = @import("std");
const log = std.log;
const c_interface = @import("c");

pub const FlatbError = error{
    BuilderInitFailed,
    BufferCreationFailed,
    TableStartFailed,
    FieldAddFailed,
    TableEndFailed,
    VectorCreateFailed,
    InvalidSeedLength,
    BuilderResetFailed,
};

/// Serialize Registration phase data (public seed and key t).
pub fn serializeRegistration(
    builder: *c_interface.flatcc_builder_t,
    public_seed: []const u8,
    t_poly: []const u64,
    out_size: *usize,
) FlatbError!*anyopaque {
    if (c_interface.zerkerus_Registration_start_as_root(builder) != 0) return FlatbError.TableStartFailed;
    if (c_interface.zerkerus_Registration_public_seed_create(builder, public_seed.ptr, public_seed.len) != 0) return FlatbError.VectorCreateFailed;
    if (c_interface.zerkerus_Registration_public_key_t_create(builder, t_poly.ptr, t_poly.len) != 0) return FlatbError.VectorCreateFailed;
    const registration = c_interface.zerkerus_Registration_end_as_root(builder);
    if (registration == 0) return FlatbError.TableEndFailed;

    out_size.* = 0;
    const buf = c_interface.flatcc_builder_finalize_buffer(builder, out_size);
    if (buf == null) return FlatbError.BufferCreationFailed;
    return buf.?;
}

pub fn deserializeAndLogRegistration(buf: *const anyopaque) void {
    const reg = c_interface.zerkerus_Registration_as_root(buf);
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

pub const RegistrationData = struct {
    public_seed: [32]u8,
    public_key_t: []u64,
};

pub fn parseRegistration(allocator: std.mem.Allocator, buf: *const anyopaque) !RegistrationData {
    const reg = c_interface.zerkerus_Registration_as_root(buf);

    const seed_vec = c_interface.zerkerus_Registration_public_seed(reg);
    const seed_len = c_interface.flatbuffers_uint8_vec_len(seed_vec);
    var public_seed: [32]u8 = undefined;
    if (seed_len != 32) return FlatbError.InvalidSeedLength;
    for (0..32) |i| {
        public_seed[i] = c_interface.flatbuffers_uint8_vec_at(seed_vec, i);
    }

    const key_vec = c_interface.zerkerus_Registration_public_key_t(reg);
    const key_len = c_interface.flatbuffers_uint64_vec_len(key_vec);
    var t_poly = try allocator.alloc(u64, key_len);
    for (0..key_len) |i| {
        t_poly[i] = c_interface.flatbuffers_uint64_vec_at(key_vec, i);
    }
    return .{ .public_seed = public_seed, .public_key_t = t_poly };
}

/// Serialize Commitment phase data (w and attempt count).
pub fn serializeCommitment(
    builder: *c_interface.flatcc_builder_t,
    attempt: u32,
    w_poly: []const u64,
    out_size: *usize,
) FlatbError!*anyopaque {
    if (c_interface.zerkerus_Commitment_start_as_root(builder) != 0) return FlatbError.TableStartFailed;
    if (c_interface.zerkerus_Commitment_attempt_add(builder, attempt) != 0) return FlatbError.FieldAddFailed;
    if (c_interface.zerkerus_Commitment_w_create(builder, w_poly.ptr, w_poly.len) != 0) return FlatbError.VectorCreateFailed;
    const commitment = c_interface.zerkerus_Commitment_end_as_root(builder);
    if (commitment == 0) return FlatbError.TableEndFailed;

    out_size.* = 0;
    const buf = c_interface.flatcc_builder_finalize_buffer(builder, out_size);
    if (buf == null) return FlatbError.BufferCreationFailed;
    return buf.?;
}

pub fn deserializeAndLogCommitment(buf: *const anyopaque) void {
    const com = c_interface.zerkerus_Commitment_as_root(buf);
    const attempt = c_interface.zerkerus_Commitment_attempt(com);
    const w_vec = c_interface.zerkerus_Commitment_w(com);
    const w_len = c_interface.flatbuffers_uint64_vec_len(w_vec);
    log.info("[FlatBuf] Commitment: attempt = {d}, w length = {d}", .{ attempt, w_len });
}

pub const CommitmentData = struct {
    attempt: u32,
    w_poly: []u64,
};

pub fn parseCommitment(allocator: std.mem.Allocator, buf: *const anyopaque) !CommitmentData {
    const com = c_interface.zerkerus_Commitment_as_root(buf);
    const attempt = c_interface.zerkerus_Commitment_attempt(com);

    const w_vec = c_interface.zerkerus_Commitment_w(com);
    const w_len = c_interface.flatbuffers_uint64_vec_len(w_vec);
    var w_poly = try allocator.alloc(u64, w_len);
    for (0..w_len) |i| {
        w_poly[i] = c_interface.flatbuffers_uint64_vec_at(w_vec, i);
    }
    return .{ .attempt = attempt, .w_poly = w_poly };
}

/// Serialize Challenge phase data (c and transcript hash).
pub fn serializeChallenge(
    builder: *c_interface.flatcc_builder_t,
    challenge: u64,
    transcript_hash: []const u8,
    out_size: *usize,
) FlatbError!*anyopaque {
    if (c_interface.zerkerus_Challenge_start_as_root(builder) != 0) return FlatbError.TableStartFailed;
    if (c_interface.zerkerus_Challenge_c_add(builder, challenge) != 0) return FlatbError.FieldAddFailed;
    if (c_interface.zerkerus_Challenge_transcript_hash_create(builder, transcript_hash.ptr, transcript_hash.len) != 0) return FlatbError.VectorCreateFailed;
    const challenge_ref = c_interface.zerkerus_Challenge_end_as_root(builder);
    if (challenge_ref == 0) return FlatbError.TableEndFailed;

    out_size.* = 0;
    const buf = c_interface.flatcc_builder_finalize_buffer(builder, out_size);
    if (buf == null) return FlatbError.BufferCreationFailed;
    return buf.?;
}

pub fn deserializeAndLogChallenge(buf: *const anyopaque) void {
    const chal = c_interface.zerkerus_Challenge_as_root(buf);
    const c_val = c_interface.zerkerus_Challenge_c(chal);
    const hash_vec = c_interface.zerkerus_Challenge_transcript_hash(chal);
    const hash_len = c_interface.flatbuffers_uint8_vec_len(hash_vec);
    log.info("[FlatBuf] Challenge: c = {d}, transcript_hash length = {d}", .{ c_val, hash_len });
}

pub const ChallengeData = struct {
    challenge: u64,
    transcript_hash: [32]u8,
};

pub fn parseChallenge(buf: *const anyopaque) !ChallengeData {
    const chal = c_interface.zerkerus_Challenge_as_root(buf);
    const challenge = c_interface.zerkerus_Challenge_c(chal);

    const hash_vec = c_interface.zerkerus_Challenge_transcript_hash(chal);
    const hash_len = c_interface.flatbuffers_uint8_vec_len(hash_vec);
    var transcript_hash: [32]u8 = undefined;
    if (hash_len != 32) return FlatbError.InvalidSeedLength;
    for (0..32) |i| {
        transcript_hash[i] = c_interface.flatbuffers_uint8_vec_at(hash_vec, i);
    }
    return .{ .challenge = challenge, .transcript_hash = transcript_hash };
}

/// Serialize Response phase data (z).
pub fn serializeResponse(
    builder: *c_interface.flatcc_builder_t,
    z_poly: []const u64,
    out_size: *usize,
) FlatbError!*anyopaque {
    if (c_interface.zerkerus_Response_start_as_root(builder) != 0) return FlatbError.TableStartFailed;
    if (c_interface.zerkerus_Response_z_create(builder, z_poly.ptr, z_poly.len) != 0) return FlatbError.VectorCreateFailed;
    const response_ref = c_interface.zerkerus_Response_end_as_root(builder);
    if (response_ref == 0) return FlatbError.TableEndFailed;

    out_size.* = 0;
    const buf = c_interface.flatcc_builder_finalize_buffer(builder, out_size);
    if (buf == null) return FlatbError.BufferCreationFailed;
    return buf.?;
}

pub fn deserializeAndLogResponse(buf: *const anyopaque) void {
    const resp = c_interface.zerkerus_Response_as_root(buf);
    const z_vec = c_interface.zerkerus_Response_z(resp);
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

pub const ResponseData = struct {
    z_poly: []u64,
};

pub fn parseResponse(allocator: std.mem.Allocator, buf: *const anyopaque) !ResponseData {
    const resp = c_interface.zerkerus_Response_as_root(buf);
    const z_vec = c_interface.zerkerus_Response_z(resp);
    const z_len = c_interface.flatbuffers_uint64_vec_len(z_vec);
    var z_poly = try allocator.alloc(u64, z_len);
    for (0..z_len) |i| {
        z_poly[i] = c_interface.flatbuffers_uint64_vec_at(z_vec, i);
    }
    return .{ .z_poly = z_poly };
}

test "flatbuf init builder" {
    const fb_builder = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);

    const init_res = c_interface.flatcc_builder_init(fb_builder);
    try std.testing.expectEqual(@as(c_int, 0), init_res);
    c_interface.flatcc_builder_clear(fb_builder);
}

test "serialize phases individually" {
    const fb_builder = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);

    const init_res = c_interface.flatcc_builder_init(fb_builder);
    try std.testing.expectEqual(@as(c_int, 0), init_res);
    defer c_interface.flatcc_builder_clear(fb_builder);

    var buf_size: usize = 0;

    // 1. Registration
    const public_seed = [_]u8{ 1, 2, 3 };
    const t_poly = [_]u64{ 10, 20 };
    if (c_interface.flatcc_builder_reset(fb_builder) != 0) return error.BuilderResetFailed;
    const reg_buf = try serializeRegistration(fb_builder, &public_seed, &t_poly, &buf_size);
    defer std.c.free(reg_buf);
    deserializeAndLogRegistration(reg_buf);

    // 2. Commitment
    const w_poly = [_]u64{ 100, 200, 300 };
    if (c_interface.flatcc_builder_reset(fb_builder) != 0) return error.BuilderResetFailed;
    const com_buf = try serializeCommitment(fb_builder, 1, &w_poly, &buf_size);
    defer std.c.free(com_buf);
    deserializeAndLogCommitment(com_buf);

    // 3. Challenge
    const transcript_hash = [_]u8{ 5, 6, 7, 8 };
    if (c_interface.flatcc_builder_reset(fb_builder) != 0) return error.BuilderResetFailed;
    const chal_buf = try serializeChallenge(fb_builder, 42, &transcript_hash, &buf_size);
    defer std.c.free(chal_buf);
    deserializeAndLogChallenge(chal_buf);

    // 4. Response
    const z_poly = [_]u64{ 1000, 2000, 3000 };
    if (c_interface.flatcc_builder_reset(fb_builder) != 0) return error.BuilderResetFailed;
    const resp_buf = try serializeResponse(fb_builder, &z_poly, &buf_size);
    defer std.c.free(resp_buf);
    deserializeAndLogResponse(resp_buf);
}

pub fn printJsonRegistration(buf: *const anyopaque, size: usize) void {
    c_interface.print_registration_json_stdout(buf, size);
    log.info("\n", .{}); // print an empty line to flush properly visually
}

pub fn printJsonCommitment(buf: *const anyopaque, size: usize) void {
    c_interface.print_commitment_json_stdout(buf, size);
    log.info("\n", .{});
}

pub fn printJsonChallenge(buf: *const anyopaque, size: usize) void {
    c_interface.print_challenge_json_stdout(buf, size);
    log.info("\n", .{});
}

pub fn printJsonResponse(buf: *const anyopaque, size: usize) void {
    c_interface.print_response_json_stdout(buf, size);
    log.info("\n", .{});
}
