const std = @import("std");
const log = std.log;
const zml = @import("zml");
const params = @import("params.zig");

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
