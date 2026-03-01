const crypto = @import("zk-crypto");

/// Debug profile — tiny ring size for fast iteration during development.
/// N=64, M=16 keeps computation trivially fast; same Kyber prime ensures
/// all field arithmetic is identical to production.
pub const debug = struct {
    pub const ntt = crypto.MakeNTT(struct {
        pub const N: usize = 64;
        pub const M: usize = 16;
        pub const Q: u64 = 8380417; // Kyber/Dilithium prime
    });

    pub const NTTModel = debug.ntt.NTTModel;
    pub const Transcript = crypto.fiat_shamir.Transcript;
    pub const labrador = crypto.MakeLabrador(debug.ntt, 64);

    /// Loose norm bound — avoids rejection-sampling retries in debug runs.
    pub const B: u64 = 64;
};

/// Production profile — full-strength Kyber/Dilithium parameters.
/// N=1024, M=256, Q=8380417. All NTT constants auto-derived at comptime.
pub const production = struct {
    pub const ntt = crypto.MakeNTT(struct {
        pub const N: usize = 1024;
        pub const M: usize = 256;
        pub const Q: u64 = 8380417; // Kyber/Dilithium prime
    });

    pub const NTTModel = production.ntt.NTTModel;
    pub const Transcript = crypto.fiat_shamir.Transcript;
    pub const labrador = crypto.MakeLabrador(production.ntt, 1000);

    /// Tight infinity-norm bound for Zero-Knowledge security.
    pub const B: u64 = 1000;
};

// --- Convenience re-exports (default to production) ---

// pub const ntt = production.ntt;
// pub const NTTModel = production.NTTModel;
// pub const Transcript = production.Transcript;
// pub const labrador = production.labrador;
// pub const B = production.B;


pub const ntt = debug.ntt;
pub const NTTModel = debug.NTTModel;
pub const Transcript = debug.Transcript;
pub const labrador = debug.labrador;
pub const B = debug.B;
