const std = @import("std");

pub fn extractSeed(data: []const u8) [32]u8 {
    var hasher = std.crypto.hash.Blake3.init(.{});
    hasher.update(data);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return digest;
}

test "extractSeed produces deterministic output" {
    const seed1 = extractSeed("test_data");
    const seed2 = extractSeed("test_data");
    try std.testing.expectEqualSlices(u8, &seed1, &seed2);

    const seed3 = extractSeed("different_data");
    var is_different = false;
    for (seed1, 0..) |b, i| {
        if (b != seed3[i]) {
            is_different = true;
            break;
        }
    }
    try std.testing.expect(is_different);
}
