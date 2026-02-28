const std = @import("std");
const crypto = @import("zk-crypto");

pub const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 1024;
    pub const M: usize = 256;
    pub const Q: u64 = 8380417; // Kyber/Dilithium prime
    pub const PSI: u64 = 17492915097719143606; // Primitive 4096-th root of unity modulo Q
    pub const ZETA: u64 = 455906449640507599; // Primitive 2048-th root modulo Q (PSI^2)
    pub const N_INV: u64 = 18437736870161940481; // Inverse of 2048 mod Q
});

pub const NTTModel = ntt.NTTModel;
pub const Transcript = crypto.fiat_shamir.Transcript;
pub const labrador = crypto.MakeLabrador(ntt, 1000);

pub const B: u64 = 1000; // Infinity norm bound for small vectors
