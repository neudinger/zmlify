# zk-crypto: Hardware-Accelerated Lattice Cryptography

`zk-crypto` is a standalone, performance-driven Bazel module providing generalized Number Theoretic Transforms (NTT), Labrador Polynomial Commitments, and Fiat-Shamir transcript algorithms designed specifically to cross-compile against hardware accelerators via **Zig Machine Learning (ZML)**.

Instead of hardcoding specific structural fields directly into algorithmic logic, `zk-crypto` is designed using Zig's powerful `comptime` metaprogramming execution. This allows diverse cryptographic projects (such as `zk-resnet` and `zerkerus`) to import the identical logic module but mathematically configure it with entirely divergent Prime Fields ($Q$) and Dimensions ($N$) with absolutely zero runtime penalty.

## 🚀 Features

*   **Comptime Modularity:** Algorithms mathematically map to any $Q$ and $N$ configuration injected at compilation.
*   **Dual NTT Backends:**
    *   **`NTTModel`** — ZML graph-compiled matrix NTT, accelerated via MLIR on Apple Silicon / TPUs / GPUs. Works for primes where the dot-product accumulation fits in `u64` (see [Prime Selection](#-prime-selection) below).
    *   **`ButterflyNTT`** — CPU-side Cooley-Tukey butterfly NTT using `u128` intermediate arithmetic. Safe for any prime size, including Goldilocks ($Q = 2^{64} - 2^{32} + 1$). Runs in $O(N \log N)$ versus the matrix approach's $O(N^2)$.
*   **ZML Buffer Integration:** `ButterflyNTT` provides `forwardBuffer` / `inverseBuffer` helpers for seamless round-trip through ZML Buffers.
*   **Zero-Knowledge Probabilistic Bounds:** Cryptographic Rejection Sampling strictly guarantees zero structural execution leakage.

## 📦 Usage

Add `@zk-crypto` to your Bazel dependencies, and import the module dynamically passing your security configuration structs.

### Example 1: NTTModel (ZML Graph Path)

```zig
const crypto = @import("zk-crypto");

const ntt = crypto.MakeNTT(struct {
    pub const N: usize = 1024;
    pub const Q: u64 = 104857601; // ~27-bit NTT-friendly prime (Q ≡ 1 mod 2048)
    pub const PSI: u64 = 4354736; // Primitive 2048-th root of unity mod Q
    pub const ZETA: u64 = 18773644; // Primitive 1024-th root of unity mod Q (PSI²)
    pub const N_INV: u64 = 104755201; // Inverse of 1024 mod Q
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
    pub const PSI: u64 = 17492915097719143606;
    pub const ZETA: u64 = 455906449640507599;
    pub const N_INV: u64 = 18437736870161940481;
});

// Use ButterflyNTT for large primes (CPU-side, u128-safe)
const ButterflyNTT = ntt.ButterflyNTT;

// In-place forward NTT
var poly: [2048]u64 = undefined;
ButterflyNTT.forward(&poly);

// Or via ZML Buffers
var result_buf = try ButterflyNTT.forwardBuffer(allocator, platform, io, input_buf);
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
