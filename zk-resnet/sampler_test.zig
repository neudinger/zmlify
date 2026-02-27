const std = @import("std");
const sampler = @import("sampler.zig");

test "CBD Sampling bounds and determinism" {
    var seed: [32]u8 = undefined;
    @memset(&seed, 42);

    var cbd1 = sampler.CBD.init(seed, 0);
    const eta: u3 = 2; // B_2 bounds: [-2, 2]
    var poly1: [256]i32 = undefined;
    cbd1.samplePoly(eta, &poly1);

    // Check bounds
    var has_positive = false;
    var has_negative = false;
    for (poly1) |x| {
        try std.testing.expect(x >= -2 and x <= 2);
        if (x > 0) has_positive = true;
        if (x < 0) has_negative = true;
    }
    // With 256 samples, we should overwhelmingly see both positive and negative values.
    try std.testing.expect(has_positive);
    try std.testing.expect(has_negative);

    // Check determinism
    var cbd2 = sampler.CBD.init(seed, 0);
    var poly2: [256]i32 = undefined;
    cbd2.samplePoly(eta, &poly2);

    try std.testing.expectEqualSlices(i32, &poly1, &poly2);

    // Check nonce separation
    var cbd3 = sampler.CBD.init(seed, 1);
    var poly3: [256]i32 = undefined;
    cbd3.samplePoly(eta, &poly3);

    // It's statistically improbable they are equal
    var diff_count: usize = 0;
    for (0..256) |i| {
        if (poly1[i] != poly3[i]) diff_count += 1;
    }
    try std.testing.expect(diff_count > 0);
}
