pub const fiat_shamir = @import("fiat_shamir.zig");

pub fn MakeNTT(comptime config: type) type {
    return @import("ntt.zig").MakeNTT(config);
}

pub fn MakeLabrador(comptime ntt: type, comptime B_val: u64) type {
    return @import("labrador.zig").MakeLabrador(ntt, B_val);
}
