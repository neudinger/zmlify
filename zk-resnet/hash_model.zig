const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const CliArgs = struct {
    model: []const u8 = "/Users/kevin/models/resnet-18/model.safetensors",
    out: []const u8 = "vk_hash.zig",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli_args = stdx.flags.parse(init.minimal.args, CliArgs);
    const model_path = cli_args.model;
    const out_path = cli_args.out;

    const file = try std.Io.Dir.openFile(.cwd(), io, model_path, .{ .mode = .read_only });
    defer file.close(io);

    const stat = try file.stat(io);
    const content = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(content);

    _ = try file.readPositionalAll(io, content, 0);

    var hasher = std.crypto.hash.Blake3.init(.{});
    hasher.update(content);

    var digest: [32]u8 = undefined;
    hasher.final(&digest);

    const out_file = try std.Io.Dir.createFile(.cwd(), io, out_path, .{});
    defer out_file.close(io);

    var formatted_buf: [1024]u8 = undefined;
    var offset: usize = 0;

    const prefix = "pub const vk_hash = [32]u8{\n    ";
    @memcpy(formatted_buf[offset .. offset + prefix.len], prefix);
    offset += prefix.len;

    for (digest, 0..) |b, i| {
        const str = try std.fmt.bufPrint(formatted_buf[offset..], "{d}, ", .{b});
        offset += str.len;
        if (i % 8 == 7 and i != 31) {
            const newline = "\n    ";
            @memcpy(formatted_buf[offset .. offset + newline.len], newline);
            offset += newline.len;
        }
    }
    const suffix = "\n};\n";
    @memcpy(formatted_buf[offset .. offset + suffix.len], suffix);
    offset += suffix.len;

    _ = try out_file.writePositionalAll(io, formatted_buf[0..offset], 0);
}
