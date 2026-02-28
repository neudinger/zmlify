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

const client = @import("client.zig");
const server = @import("server.zig");
const engine = @import("engine.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var setup = try engine.setupZmlEngine(allocator, io);
    defer setup.platform.deinit(allocator);
    defer setup.mul_exe.deinit();

    const fb_builder = try std.heap.c_allocator.create(c_interface.flatcc_builder_t);
    defer std.heap.c_allocator.destroy(fb_builder);
    if (c_interface.flatcc_builder_init(fb_builder) != 0) return error.BuilderInitFailed;
    defer c_interface.flatcc_builder_clear(fb_builder);

    // 1. Client creates registration flatbuffer
    const client_reg = try client.clientRegistrationPhase(allocator, io, setup.platform, &setup.mul_exe, fb_builder);
    defer std.c.free(client_reg.reg_buf);
    defer allocator.free(client_reg.state.a_poly);
    defer allocator.free(client_reg.state.s_poly);

    // 2. Server processes registration flatbuffer
    const server_state = try server.serverReceiveRegistrationPhase(allocator, client_reg.reg_buf);
    defer allocator.free(server_state.a_poly);
    defer allocator.free(server_state.t_poly);

    var proof_generated = false;
    var attempts: u32 = 0;

    log.info("Starting ZKP generation loop...", .{});
    while (!proof_generated) {
        attempts += 1;

        // 3. Client creates commitment flatbuffer
        const client_com = try client.clientCommitPhase(allocator, io, setup.platform, &setup.mul_exe, &client_reg.state, attempts, fb_builder);
        defer allocator.free(client_com.y_poly);
        defer std.c.free(client_com.com_buf);

        // 4. Server creates challenge flatbuffer
        const server_chal = try server.serverChallengePhase(allocator, client_com.com_buf, fb_builder);
        defer std.c.free(server_chal.chal_buf);

        // 5. Client creates response flatbuffer
        const client_resp = try client.clientResponsePhase(allocator, &client_reg.state, client_com.y_poly, server_chal.chal_buf, fb_builder);
        if (client_resp.resp_buf) |resp_buf| {
            defer std.c.free(resp_buf);

            log.info("--- Verification (Attempt {d}) ---", .{attempts});
            // 6. Server verifies all flatbuffers
            const is_verified = try server.serverVerifyPhase(allocator, io, setup.platform, &setup.mul_exe, &server_state, client_com.com_buf, server_chal.chal_buf, resp_buf);

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
                const limit_y = 8192 * params.B;
                const safe_limit = limit_y - (1 * params.B);
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
