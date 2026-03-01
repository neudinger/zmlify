# zk-crypto: Hardware-Accelerated Lattice Cryptography

`zk-crypto` is a standalone, performance-driven Bazel module providing generalized Number Theoretic Transforms (NTT), Labrador Polynomial Commitments, and Fiat-Shamir transcript algorithms designed specifically to cross-compile against hardware accelerators via **Zig Machine Learning (ZML)**.

Instead of hardcoding specific structural fields directly into algorithmic logic, `zk-crypto` is designed using Zig's powerful `comptime` metaprogramming execution. This allows diverse cryptographic projects (such as `zk-resnet` and `zerkerus`) to import the identical logic module but mathematically configure it with entirely divergent Prime Fields ($Q$) and Dimensions ($N$) with absolutely zero runtime penalty.

## 🚀 Features

*   **Comptime Modularity:** Algorithms mathematically map to any $Q$ and $N$ configuration injected at compilation.
*   **Automatic Constant Generation:** `PSI`, `ZETA`, and `N_INV` are derived automatically at `comptime` via primitive root discovery — no manual calculation needed. You only need to supply `N` and `Q`.
*   **Dual NTT Backends:**
    *   **`NTTModel`** — ZML graph-compiled matrix NTT, accelerated via MLIR on Apple Silicon / TPUs / GPUs. Works for primes where the dot-product accumulation fits in `u64` (see [Prime Selection](#-prime-selection) below).
    *   **`ButterflyNTT`** — CPU-side Cooley-Tukey butterfly NTT using `u128` intermediate arithmetic. Safe for any prime size, including Goldilocks ($Q = 2^{64} - 2^{32} + 1$). Runs in $O(N \log N)$ versus the matrix approach's $O(N^2)$.
*   **ZML Buffer Integration:** `ButterflyNTT` provides `forwardBuffer` / `inverseBuffer` helpers for seamless round-trip through ZML Buffers.
*   **Constant-Time Norm Check:** `checkNorm` and `rejectionSample` in `labrador.zig` compute the signed absolute value in $\mathbb{Z}_Q$ using fully branch-free arithmetic — no `if` on secret data.

## 📦 Usage

Add `@zk-crypto` to your Bazel dependencies, and import the module dynamically passing your security configuration structs.

### Example 1: NTTModel (ZML Graph Path)

Only `N` and `Q` are required. All NTT twiddle factors are derived automatically:

```zig
const crypto = @import("zk-crypto");

const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 1024;
    pub const Q: u64 = 104857601; // ~27-bit NTT-friendly prime (Q ≡ 1 mod 2048)
    // PSI, ZETA, N_INV derived automatically at comptime
});

// Use NTTModel for ZML graph compilation (hardware-accelerated)
const NTTModel = ntt.NTTModel;

const labrador = crypto.MakeLabrador(ntt, 1000);
```

### Example 2: ButterflyNTT (Goldilocks Path)

```zig
const crypto = @import("zk-crypto");

const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 2048;
    pub const Q: u64 = 0xFFFFFFFF00000001; // Goldilocks prime
    // PSI, ZETA, N_INV derived automatically at comptime
});

// Use ButterflyNTT for large primes (CPU-side, u128-safe)
const ButterflyNTT = ntt.ButterflyNTT;

// In-place forward NTT
var poly: [2048]u64 = undefined;
ButterflyNTT.forward(&poly);

// Or via ZML Buffers
var result_buf = try ButterflyNTT.forwardBuffer(allocator, platform, io, input_buf);
```

### Example 3: Debug vs. Production Profiles

You can define multiple NTT instances in the same file for different environments:

```zig
// Debug: tiny ring, fast iteration, no rejection-sampling retries
pub const debug = struct {
    pub const ntt = crypto.MakeNTT(struct {
        pub const N: usize = 64;
        pub const Q: u64 = 8380417;
    });
    pub const labrador = crypto.MakeLabrador(ntt, 64);
    pub const B: u64 = 64;
};

// Production: full cryptographic strength
pub const production = struct {
    pub const ntt = crypto.MakeNTT(struct {
        pub const N: usize = 1024;
        pub const Q: u64 = 8380417;
    });
    pub const labrador = crypto.MakeLabrador(ntt, 1000);
    pub const B: u64 = 1000;
};
```

## 🔑 Prime Selection

The ZML `NTTModel` uses a `u64` dot-product internally. To avoid overflow, Q must satisfy:

$$N \times (Q - 1)^2 < 2^{64}  \implies  Q < \sqrt{2^{64} / N} + 1$$

| N | Max Q for NTTModel | Backend |
|---|---------------------|---------| 
| 1024 | **~134 million** (~27 bit) | `NTTModel` ✅ |
| 2048 | **~95 million** (~26.5 bit) | `NTTModel` ✅ |
| Any | **Unlimited** | `ButterflyNTT` ✅ |

For primes above the NTTModel limit (e.g. Goldilocks ≈ 2⁶⁴), use `ButterflyNTT`.

## 🧪 Testing

```bash
bazel test //zk-crypto:fiat_shamir_test
bazel run //zk-crypto:ntt_test_bin
bazel run //zk-crypto:labrador_test_bin
```

The NTT test suite validates both backends and includes a **cross-validation** test confirming `ButterflyNTT` and `NTTModel` produce identical outputs on Q = 104857601.
