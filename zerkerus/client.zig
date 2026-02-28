const std = @import("std");
const log = std.log;
const zml = @import("zml");
const params = @import("params.zig");
const utils = @import("utils.zig");
const math = @import("math.zig");
const flatbuf_tools = @import("flatbuf_tools.zig");
const c_interface = @import("c");

// Client State
// This struct represents the exclusive memory space of the Prover (Client).
// - `secret_seed` and `s_poly` (the User's Password) MUST NEVER leave the client device securely.
// - `a_poly` is mathematically expanded from the public seed uniformly.
pub const ClientState = struct {
    secret_seed: [32]u8,
    s_poly: []const u64,
    a_poly: []const u64,
};

pub fn clientRegistrationPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { state: ClientState, reg_buf: *anyopaque, reg_buf_size: usize } {
    log.info("=== 1. CLIENT: REGISTRATION ===", .{});
    // The public_seed is conceptually established ahead of time (e.g. Server's static prime parameter).
    const public_seed = utils.extractSeed("PublicSharedMatrixSeed");
    const a_poly = try math.expandA(allocator, public_seed);

    // The Client derives their highly confidential `s_poly` (their password).
    const secret_seed = utils.extractSeed("UserPasswordSalt");
    const s_poly = try math.sampleSmall(allocator, secret_seed, params.B);

    // The Client generates their Public Key `t` to register themselves with the Server.
    const t_poly = try math.matVecMul(io, platform, allocator, mul_exe, a_poly, s_poly);
    defer allocator.free(t_poly);

    var buf_size: usize = 0;
    if (c_interface.flatcc_builder_reset(fb_builder) != 0) return flatbuf_tools.FlatbError.BuilderResetFailed;
    const reg_buf = try flatbuf_tools.serializeRegistration(fb_builder, &public_seed, t_poly, &buf_size);

    return .{
        .state = .{
            .secret_seed = secret_seed,
            .s_poly = s_poly,
            .a_poly = a_poly,
        },
        .reg_buf = reg_buf,
        .reg_buf_size = buf_size,
    };
}

pub fn clientCommitPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    client_state: *const ClientState,
    attempts: u32,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { y_poly: []const u64, com_buf: *anyopaque, com_buf_size: usize } {
    log.info("=== 3. CLIENT: COMMITMENT ===", .{});
    var y_seed_input: [36]u8 = undefined;
    @memcpy(y_seed_input[0..32], &client_state.secret_seed);
    std.mem.writeInt(u32, y_seed_input[32..36], attempts, .little);

    const iter_seed = utils.extractSeed(&y_seed_input);

    // --- REJECTION SAMPLING MATH (THE MASK `y`) ---
    // The Client generates a mask `y`.
    // It is mathematically paramount this mask is significantly larger than their password magnitude `params.B`.
    // We multiply `params.B` by 8192 (or 2^13).
    // This multiplier is statistically defined in BDLOP commitments (like Dilithium) to ensure that when `y`
    // is added to the secret `s` later, the shape of `y` completely mathematically overrides and "hides" `s`.
    const limit_y = 8192 * params.B;
    const y_poly = try math.sampleSmall(allocator, iter_seed, limit_y);

    // The Client generates their Commitment `w` to lock in their mask without revealing `y`.
    const w_poly = try math.matVecMul(io, platform, allocator, mul_exe, client_state.a_poly, y_poly);
    defer allocator.free(w_poly);

    var buf_size: usize = 0;
    if (c_interface.flatcc_builder_reset(fb_builder) != 0) return flatbuf_tools.FlatbError.BuilderResetFailed;
    const com_buf = try flatbuf_tools.serializeCommitment(fb_builder, attempts, w_poly, &buf_size);

    return .{ .y_poly = y_poly, .com_buf = com_buf, .com_buf_size = buf_size };
}

pub fn clientResponsePhase(
    allocator: std.mem.Allocator,
    client_state: *const ClientState,
    y_poly: []const u64,
    chal_buf: *const anyopaque,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { is_safe: bool, actual_max: u64, resp_buf: ?*anyopaque, resp_buf_size: usize } {
    log.info("=== 5. CLIENT: RESPONSE ===", .{});
    flatbuf_tools.deserializeAndLogChallenge(chal_buf);

    const parsed = try flatbuf_tools.parseChallenge(chal_buf);
    const challenge = parsed.challenge;

    // The Client computes `z = y + c * s`.
    // Because `z` explicitly mathematically contains `s`, the sheer numeric size of `z` could accidentally leak
    // the value of `s` to the Server if `y` wasn't perfectly uniform across the top edges of its limits.
    const z_poly = try math.vecAddScalarMul(allocator, y_poly, challenge, client_state.s_poly);
    defer allocator.free(z_poly);

    // --- REJECTION SAMPLING ABORT CONDITIONS ---
    // We expect the mask `y` to max out randomly at `limit_y` (8192 * B).
    // The secret `s` maxes out at `B`. Our challenge `c` is binary (0 or 1).
    // If the calculation `z = y + (1 * s)` pushes `z` strictly higher than `limit_y - (1 * B)`,
    // it mathematically proves `y` was dangerously close to the boundary, and `s` is statistically leaking.
    // The equation thus becomes strictly bounded at `Limit_Y - (max_c * max_s) = Limit_Y - Beta`
    const limit_y = 8192 * params.B;
    const safe_limit = limit_y - (1 * params.B);

    var actual_max: u64 = 0;
    // IF z is numerically larger than safe_limit, the Prover MUST silently throw `z` away completely
    // (return a structurally "unsafe" status) to force generation of an entirely new mask `y` from scratch.
    const is_safe = math.isSmall(z_poly, safe_limit, &actual_max);

    if (is_safe) {
        var buf_size: usize = 0;
        if (c_interface.flatcc_builder_reset(fb_builder) != 0) return flatbuf_tools.FlatbError.BuilderResetFailed;
        const resp_buf = try flatbuf_tools.serializeResponse(fb_builder, z_poly, &buf_size);
        return .{ .is_safe = is_safe, .actual_max = actual_max, .resp_buf = resp_buf, .resp_buf_size = buf_size };
    }

    return .{ .is_safe = is_safe, .actual_max = actual_max, .resp_buf = null, .resp_buf_size = 0 };
}
