/// Fiat-Shamir transcript implementation for non-interactive proofs.
pub const fiat_shamir = @import("fiat_shamir.zig");

/// Generates an NTT implementation tailored to a specific configuration.
/// The `config` struct must provide the following fields:
/// - `N`: The degree of the polynomial ring (e.g., 1024 or 2048).
/// - `Q`: The modular arithmetic prime modulus.
///
/// Additionally, these fields are optional and will be derived at `comptime` if missing:
/// - `M`: The rank of the module (defaults to `N`).
/// - `PSI`: A primitive (2*N)-th root of unity modulo Q.
/// - `ZETA`: A primitive N-th root of unity modulo Q (calculated as `PSI^2 mod Q`).
/// - `N_INV`: The modular inverse of `N` modulo Q (1/N mod Q).
pub fn MakeNTT(comptime config: type) type {
    return @import("ntt.zig").MakeNTT(config);
}

/// Generates a Labrador lattice-based commitment scheme instance.
/// Requires a previously initialized NTT type and an infinity norm bound B_val.
pub fn MakeLabrador(comptime ntt_type: type, comptime B_val: u64) type {
    return @import("labrador.zig").MakeLabrador(ntt_type, B_val);
}
