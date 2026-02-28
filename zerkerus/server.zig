const std = @import("std");
const log = std.log;
const zml = @import("zml");
const params = @import("params.zig");
const utils = @import("utils.zig");
const math = @import("math.zig");
const flatbuf_tools = @import("flatbuf_tools.zig");
const c_interface = flatbuf_tools.c_interface;

// Server State
// This struct represents the database stored by the Verifier (Server).
// - `a_poly` is mathematically expanded from the identical public seed.
// - `t_poly` is the Public Key derived uniquely from the Client's Password (`A*s mod Q`) stored in the database.
// - The Server NEVER possesses the literal `secret_seed` or the underlying `s_poly`.
pub const ServerState = struct {
    a_poly: []const u64,
    t_poly: []const u64,
};

pub fn serverReceiveRegistrationPhase(
    allocator: std.mem.Allocator,
    reg_buf: *const anyopaque,
) !ServerState {
    log.info("=== 2. SERVER: RECEIVE REGISTRATION ===", .{});
    flatbuf_tools.deserializeAndLogRegistration(reg_buf);

    const parsed = try flatbuf_tools.parseRegistration(allocator, reg_buf);
    const a_poly = try math.expandA(allocator, parsed.public_seed);

    return .{
        .a_poly = a_poly,
        .t_poly = parsed.public_key_t,
    };
}

pub fn serverChallengePhase(
    allocator: std.mem.Allocator,
    com_buf: *const anyopaque,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { chal_buf: *anyopaque } {
    log.info("=== 4. SERVER: CHALLENGE ===", .{});
    flatbuf_tools.deserializeAndLogCommitment(com_buf);

    const parsed = try flatbuf_tools.parseCommitment(allocator, com_buf);
    defer allocator.free(parsed.w_poly);

    var transcript = params.Transcript.init("LabradorZKP");
    transcript.appendMessage("w", std.mem.sliceAsBytes(parsed.w_poly));
    const challenge = transcript.getChallengeU64("c") % 2;

    var transcript_hash: [32]u8 = undefined;
    const hash_slice = transcript.getChallenge("hash");
    @memcpy(&transcript_hash, &hash_slice);

    var buf_size: usize = 0;
    _ = c_interface.flatcc_builder_reset(fb_builder);
    const chal_buf = try flatbuf_tools.serializeChallenge(fb_builder, challenge, &transcript_hash, &buf_size);

    return .{ .chal_buf = chal_buf };
}

pub fn serverVerifyPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    server_state: *const ServerState,
    com_buf: *const anyopaque,
    chal_buf: *const anyopaque,
    resp_buf: *const anyopaque,
) !bool {
    log.info("=== 6. SERVER: VERIFICATION ===", .{});
    flatbuf_tools.deserializeAndLogResponse(resp_buf);

    const parsed_com = try flatbuf_tools.parseCommitment(allocator, com_buf);
    defer allocator.free(parsed_com.w_poly);

    const parsed_chal = try flatbuf_tools.parseChallenge(chal_buf);
    const challenge = parsed_chal.challenge;

    const parsed_resp = try flatbuf_tools.parseResponse(allocator, resp_buf);
    defer allocator.free(parsed_resp.z_poly);

    const limit_y = 8192 * params.B;

    const lhs = try math.matVecMul(io, platform, allocator, mul_exe, server_state.a_poly, parsed_resp.z_poly);
    defer allocator.free(lhs);

    const rhs = try math.vecAddScalarMul(allocator, parsed_com.w_poly, challenge, server_state.t_poly);
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
    } else if (!math.isSmall(parsed_resp.z_poly, limit_y + params.B, null)) {
        log.err("FAILURE: Z vector too large.", .{});
        return false;
    }
    return true;
}
