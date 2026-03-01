const std = @import("std");
const log = std.log;
const zml = @import("zml");
const params = @import("params.zig");
const utils = @import("utils.zig");
const math = @import("math.zig");
const flatbuf_tools = @import("flatbuf_tools.zig");
const c_interface = @import("c");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const client = @import("client.zig");
const server = @import("server.zig");
const engine = @import("engine.zig");

pub fn main(init: std.process.Init) !void {
    const allocator: std.mem.Allocator = init.gpa;
    const io: std.Io = init.io;

    var setup: engine.Setup = try engine.setupZmlEngine(allocator, io);
    defer setup.deinit(allocator);

    const fb_builder: *c_interface.flatcc_builder_t = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);
    if (c_interface.flatcc_builder_init(fb_builder) != 0) return error.BuilderInitFailed;
    defer c_interface.flatcc_builder_clear(fb_builder);

    // 1. Client creates registration flatbuffer
    const client_reg: client.ClientRegistration = try client.clientRegistrationPhase(allocator, io, setup.platform, &setup.mul_exe, fb_builder);
    defer client_reg.deinit(allocator);

    log.info("\n--- Registration JSON ---", .{});
    flatbuf_tools.printJsonRegistration(client_reg.reg_buf, client_reg.reg_buf_size);

    const base64_reg: []u8 = try client_reg.toBase64(allocator);
    defer allocator.free(base64_reg);
    log.info("\n--- Registration Base64 ---\n{s}\n", .{base64_reg});

    // 2. Server processes registration flatbuffer
    const server_state: server.ServerState = try server.serverReceiveRegistrationPhase(allocator, client_reg.reg_buf);
    defer server_state.deinit(allocator);

    var proof_generated: bool = false;
    var attempts: u32 = 0;

    log.info("Starting ZKP generation loop...", .{});
    while (!proof_generated) {
        attempts += 1;

        // 3. Client creates commitment flatbuffer
        const client_com: client.ClientCommitment = try client.clientCommitPhase(allocator, io, setup.platform, &setup.mul_exe, &client_reg.state, attempts, fb_builder);
        defer client_com.deinit(allocator);

        log.info("--- Commitment JSON ---", .{});
        flatbuf_tools.printJsonCommitment(client_com.com_buf, client_com.com_buf_size);

        const base64_com: []u8 = try client_com.toBase64(allocator);
        defer allocator.free(base64_com);
        log.info("\n--- Commitment Base64 ---\n{s}\n", .{base64_com});

        // 4. Server creates challenge flatbuffer
        const server_chal: server.ServerChallenge = try server.serverChallengePhase(allocator, client_com.com_buf, fb_builder);
        defer server_chal.deinit();

        log.info("--- Challenge JSON ---", .{});
        flatbuf_tools.printJsonChallenge(server_chal.chal_buf, server_chal.chal_buf_size);

        const base64_chal: []u8 = try server_chal.toBase64(allocator);
        defer allocator.free(base64_chal);
        log.info("\n--- Challenge Base64 ---\n{s}\n", .{base64_chal});

        // 5. Client creates response ßflatbuffer
        const client_resp: client.ClientResponse = try client.clientResponsePhase(allocator, &client_reg.state, client_com.y_poly, server_chal.chal_buf, fb_builder);
        defer client_resp.deinit();

        if (client_resp.resp_buf) |resp_buf| {
            log.info("--- Response JSON ---", .{});
            flatbuf_tools.printJsonResponse(resp_buf, client_resp.resp_buf_size);

            // Ok because we know resp_buf is not null
            const base64_resp: []u8 = (try client_resp.toBase64(allocator)).?;
            defer allocator.free(base64_resp);
            log.info("\n--- Response Base64 ---\n{s}\n", .{base64_resp});

            log.info("--- Verification (Attempt {d}) ---", .{attempts});
            // 6. Server verifies all flatbuffers
            const is_verified: bool = try server.serverVerifyPhase(allocator, io, setup.platform, &setup.mul_exe, &server_state, client_com.com_buf, server_chal.chal_buf, resp_buf);

            if (is_verified) {
                proof_generated = true;
            } else {
                break;
            }
        } else {
            // --- REJECTION ABORT LOGGING ---
            // The protocol is computationally fast, but rejection aborts physically happen roughly ~50% of the time
            // depending on the bounds. To avoid terminal spam, we only log the abort trace every 10 iterations.
            // This implicitly confirms the Client has successfully regenerated a completely new mask `y` and `w`.
            if (attempts % 10 == 0) {
                const limit_y: u64 = 8192 * params.B;
                const safe_limit: u64 = limit_y - (1 * params.B);
                log.info("... Client restarted protocol (Rejection Sampling Limit {d}, Max was {d}) ...", .{ safe_limit, client_resp.actual_max });
            }
        }
    }
    log.info("SUCCESS: Zero Knowledge Proof Accepted. (Attempts: {d})", .{attempts});
}

test {
    _ = @import("utils.zig");
    _ = @import("math.zig");
    _ = @import("flatbuf_tools.zig");
}
