const std = @import("std");
const zml = @import("zml");

pub const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Context {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = try .auto(allocator, io, .{}),
            .rng = std.Random.DefaultPrng.init(0),
        };
    }

    pub fn deinit(self: *Context) void {
        self.platform.deinit(self.allocator);
    }

    pub fn random(self: *Context) std.Random {
        return self.rng.random();
    }

    pub fn allocAndInit(self: *Context, comptime T: type, len: usize, factory: fn (std.Random) T) ![]T {
        const data = try self.allocator.alloc(T, len);
        const r = self.random();
        for (data) |*v| v.* = factory(r);
        return data;
    }

    pub fn bufferFromSlice(self: Context, shape: zml.Shape, data: anytype) !zml.Buffer {
        return zml.Buffer.fromSlice(self.io, self.platform, zml.Slice.init(shape, std.mem.sliceAsBytes(data)));
    }

    pub fn sliceFromBuffer(self: Context, buffer: zml.Buffer, out: anytype) !void {
        try buffer.toSlice(self.io, zml.Slice.init(buffer.shape(), std.mem.sliceAsBytes(out)));
    }
};
