const std = @import("std");
const log = std.log;
const zml = @import("zml");
const params = @import("params.zig");
const utils = @import("utils.zig");
const math = @import("math.zig");
const flatbuf_tools = @import("flatbuf_tools.zig");
const c_interface = flatbuf_tools.c_interface;

pub const std_options: std.Options = .{
    .log_level = .info,
};

// Client State
const ClientState = struct {
    secret_seed: [32]u8,
    s_poly: []const u64,
    a_poly: []const u64,
};

// Server State
const ServerState = struct {
    a_poly: []const u64,
    t_poly: []const u64,
};

pub fn clientRegistrationPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { state: ClientState, reg_buf: *anyopaque } {
    log.info("=== 1. CLIENT: REGISTRATION ===", .{});
    const public_seed = utils.extractSeed("PublicSharedMatrixSeed");
    const a_poly = try math.expandA(allocator, public_seed);

    const secret_seed = utils.extractSeed("UserPasswordSalt");
    const s_poly = try math.sampleSmall(allocator, secret_seed, params.B);

    const t_poly = try math.matVecMul(io, platform, allocator, mul_exe, a_poly, s_poly);
    defer allocator.free(t_poly);

    var buf_size: usize = 0;
    _ = c_interface.flatcc_builder_reset(fb_builder);
    const reg_buf = try flatbuf_tools.serializeRegistration(fb_builder, &public_seed, t_poly, &buf_size);

    return .{
        .state = .{
            .secret_seed = secret_seed,
            .s_poly = s_poly,
            .a_poly = a_poly,
        },
        .reg_buf = reg_buf,
    };
}

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

pub fn clientCommitPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    client: *const ClientState,
    attempts: u32,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { y_poly: []const u64, com_buf: *anyopaque } {
    log.info("=== 3. CLIENT: COMMITMENT ===", .{});
    var y_seed_input: [36]u8 = undefined;
    @memcpy(y_seed_input[0..32], &client.secret_seed);
    std.mem.writeInt(u32, y_seed_input[32..36], attempts, .little);

    const iter_seed = utils.extractSeed(&y_seed_input);
    const limit_y = 8192 * params.B;
    const y_poly = try math.sampleSmall(allocator, iter_seed, limit_y);

    const w_poly = try math.matVecMul(io, platform, allocator, mul_exe, client.a_poly, y_poly);
    defer allocator.free(w_poly);

    var buf_size: usize = 0;
    _ = c_interface.flatcc_builder_reset(fb_builder);
    const com_buf = try flatbuf_tools.serializeCommitment(fb_builder, attempts, w_poly, &buf_size);

    return .{ .y_poly = y_poly, .com_buf = com_buf };
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

pub fn clientResponsePhase(
    allocator: std.mem.Allocator,
    client: *const ClientState,
    y_poly: []const u64,
    chal_buf: *const anyopaque,
    fb_builder: *c_interface.flatcc_builder_t,
) !struct { is_safe: bool, actual_max: u64, resp_buf: ?*anyopaque } {
    log.info("=== 5. CLIENT: RESPONSE ===", .{});
    flatbuf_tools.deserializeAndLogChallenge(chal_buf);

    const parsed = try flatbuf_tools.parseChallenge(chal_buf);
    const challenge = parsed.challenge;

    const z_poly = try math.vecAddScalarMul(allocator, y_poly, challenge, client.s_poly);
    defer allocator.free(z_poly);

    const limit_y = 8192 * params.B;
    const safe_limit = limit_y - (1 * params.B);

    var actual_max: u64 = 0;
    const is_safe = math.isSmall(z_poly, safe_limit, &actual_max);

    if (is_safe) {
        var buf_size: usize = 0;
        _ = c_interface.flatcc_builder_reset(fb_builder);
        const resp_buf = try flatbuf_tools.serializeResponse(fb_builder, z_poly, &buf_size);
        return .{ .is_safe = is_safe, .actual_max = actual_max, .resp_buf = resp_buf };
    }

    return .{ .is_safe = is_safe, .actual_max = actual_max, .resp_buf = null };
}

pub fn serverVerifyPhase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mul_exe: *zml.Exe,
    server: *const ServerState,
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

    const lhs = try math.matVecMul(io, platform, allocator, mul_exe, server.a_poly, parsed_resp.z_poly);
    defer allocator.free(lhs);

    const rhs = try math.vecAddScalarMul(allocator, parsed_com.w_poly, challenge, server.t_poly);
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var setup = try setupZmlEngine(allocator, io);
    defer setup.platform.deinit(allocator);
    defer setup.mul_exe.deinit();

    const fb_builder = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);
    if (c_interface.flatcc_builder_init(fb_builder) != 0) return error.BuilderInitFailed;
    defer c_interface.flatcc_builder_clear(fb_builder);

    // 1. Client creates registration flatbuffer
    const client_reg = try clientRegistrationPhase(allocator, io, setup.platform, &setup.mul_exe, fb_builder);
    defer std.c.free(client_reg.reg_buf);
    defer allocator.free(client_reg.state.a_poly);
    defer allocator.free(client_reg.state.s_poly);

    // 2. Server processes registration flatbuffer
    const server_state = try serverReceiveRegistrationPhase(allocator, client_reg.reg_buf);
    defer allocator.free(server_state.a_poly);
    defer allocator.free(server_state.t_poly);

    var proof_generated = false;
    var attempts: u32 = 0;

    log.info("Starting ZKP generation loop...", .{});
    while (!proof_generated) {
        attempts += 1;

        // 3. Client creates commitment flatbuffer
        const client_com = try clientCommitPhase(allocator, io, setup.platform, &setup.mul_exe, &client_reg.state, attempts, fb_builder);
        defer allocator.free(client_com.y_poly);
        defer std.c.free(client_com.com_buf);

        // 4. Server creates challenge flatbuffer
        const server_chal = try serverChallengePhase(allocator, client_com.com_buf, fb_builder);
        defer std.c.free(server_chal.chal_buf);

        // 5. Client creates response flatbuffer
        const client_resp = try clientResponsePhase(allocator, &client_reg.state, client_com.y_poly, server_chal.chal_buf, fb_builder);
        if (client_resp.resp_buf) |resp_buf| {
            defer std.c.free(resp_buf);

            log.info("--- Verification (Attempt {d}) ---", .{attempts});
            // 6. Server verifies all flatbuffers
            const is_verified = try serverVerifyPhase(allocator, io, setup.platform, &setup.mul_exe, &server_state, client_com.com_buf, server_chal.chal_buf, resp_buf);

            if (is_verified) {
                log.info("SUCCESS: Zero Knowledge Proof Accepted. (Attempts: {d})", .{attempts});
                proof_generated = true;
            } else {
                break;
            }
        } else {
            if (attempts % 10 == 0) {
                const limit_y = 8192 * params.B;
                const safe_limit = limit_y - (1 * params.B);
                log.info("... Client restarted protocol (Rejection Sampling Limit {d}, Max was {d}) ...", .{ safe_limit, client_resp.actual_max });
            }
        }
    }
}

test {
    _ = @import("utils.zig");
    _ = @import("math.zig");
    _ = @import("flatbuf_tools.zig");
}
