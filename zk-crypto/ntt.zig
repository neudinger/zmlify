const std = @import("std");
const builtin = @import("builtin");
const is_wasm = builtin.target.cpu.arch == .wasm32;
const zml = if (!is_wasm) @import("zml") else struct {};

pub fn MakeNTT(comptime config: type) type {
    const math = struct {
        fn modMul(a: u64, b: u64, q: u64) u64 {
            const prod = @as(u128, a) * @as(u128, b);
            return @as(u64, @intCast(prod % q));
        }

        fn modPow(base: u64, exp: u64, q: u64) u64 {
            var res: u64 = 1;
            var b = base;
            var e = exp;
            while (e > 0) {
                if (e % 2 == 1) res = modMul(res, b, q);
                b = modMul(b, b, q);
                e /= 2;
            }
            return res;
        }

        fn modInv(a: u64, q: u64) u64 {
            return modPow(a, q - 2, q);
        }

        fn findPrimitiveRoot(q: u64) u64 {
            var factors: [64]u64 = undefined;
            var num_factors: usize = 0;

            var n = q - 1;
            var i: u64 = 2;
            while (i * i <= n) : (i += 1) {
                if (n % i == 0) {
                    factors[num_factors] = i;
                    num_factors += 1;
                    while (n % i == 0) n /= i;
                }
            }
            if (n > 1) {
                factors[num_factors] = n;
                num_factors += 1;
            }

            var g: u64 = 2;
            while (g < q) : (g += 1) {
                var is_primitive = true;
                for (0..num_factors) |k| {
                    const factor = factors[k];
                    if (modPow(g, (q - 1) / factor, q) == 1) {
                        is_primitive = false;
                        break;
                    }
                }
                if (is_primitive) return g;
            }
            unreachable;
        }

        fn getPrimitiveRootOfUnity(n: u64, q: u64) u64 {
            const g = findPrimitiveRoot(q);
            if ((q - 1) % n != 0) @compileError("n does not divide q - 1");
            return modPow(g, (q - 1) / n, q);
        }
    };

    return struct {
        /// Degree of the polynomial ring (length of the NTT vector).
        /// All NTT operations are performed on vectors of size N.
        pub const N: usize = config.N;

        /// Rank of the module (number of polynomials per vector).
        /// Used by higher-level primitives like Labrador.
        pub const M: usize = if (@hasDecl(config, "M")) config.M else config.N;

        /// Modular arithmetic prime modulus Q.
        pub const Q: u64 = config.Q;

        /// Primitive (2*N)-th root of unity modulo Q.
        /// Primarily used for negacyclic twist factors (psi^j).
        pub const PSI: u64 = if (@hasDecl(config, "PSI")) config.PSI else math.getPrimitiveRootOfUnity(2 * N, Q);

        /// Primitive N-th root of unity modulo Q (calculated as PSI^2 mod Q).
        /// Used as the twiddle factor base for the main butterfly stages.
        pub const ZETA: u64 = if (@hasDecl(config, "ZETA")) config.ZETA else math.modPow(PSI, 2, Q);

        /// Modular inverse of N modulo Q (1/N mod Q).
        /// Used for scaling the final result of the inverse NTT.
        pub const N_INV: u64 = if (@hasDecl(config, "N_INV")) config.N_INV else math.modInv(N, Q);

        /// Number of butterfly stages, equal to ctz(N).
        pub const LOG_N: usize = @ctz(@as(usize, N));

        // --- Modular Arithmetic Helpers (u128 intermediates, no overflow) ---

        pub fn modMul(a: u64, b: u64) u64 {
            const prod = @as(u128, a) * @as(u128, b);
            return @as(u64, @intCast(prod % Q));
        }

        pub fn modAdd(a: u64, b: u64) u64 {
            const sum = @as(u128, a) + @as(u128, b);
            return @as(u64, @intCast(sum % Q));
        }

        pub fn modSub(a: u64, b: u64) u64 {
            if (a >= b) return a - b;
            return Q - (b - a);
        }

        fn constantTimeSelect(mask: u64, a: u64, b: u64) u64 {
            return (a & mask) | (b & ~mask);
        }

        pub fn modPow(base: u64, exp: u64) u64 {
            var res: u64 = 1;
            var b = base;
            for (0..64) |i| {
                const bit = (exp >> @intCast(i)) & 1;
                const mask = if (bit == 1) @as(u64, ~@as(u64, 0)) else 0;
                const multiplier = constantTimeSelect(mask, b, 1);
                res = modMul(res, multiplier);
                b = modMul(b, b);
            }
            return res;
        }

        /// Modular inverse via Fermat's little theorem: a^{-1} = a^{Q-2} mod Q
        pub fn modInv(a: u64) u64 {
            return modPow(a, Q - 2);
        }

        // --- Bit-Reverse Permutation ---

        fn bitReverse(x: usize) usize {
            var v = x;
            var r: usize = 0;
            for (0..LOG_N) |_| {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            return r;
        }

        fn bitReversePermutation(poly: []u64) void {
            std.debug.assert(poly.len == N);
            for (0..N) |i| {
                const j = bitReverse(i);
                if (i < j) {
                    std.mem.swap(u64, &poly[i], &poly[j]);
                }
            }
        }

        // =====================================================================
        // CPU Butterfly NTT — Cooley-Tukey (DIT), safe for any prime size.
        // Uses u128 intermediate arithmetic; works correctly with Goldilocks.
        // =====================================================================

        pub const ButterflyNTT = struct {
            /// In-place forward negacyclic NTT (Cooley-Tukey, DIT).
            ///
            /// Computes: result[k] = Σ_j poly[j] · PSI^{(2j+1)k}  (mod Q)
            ///
            /// Steps:
            ///   1. Pre-multiply by PSI powers (negacyclic twist)
            ///   2. Bit-reverse permutation
            ///   3. Butterfly stages with ZETA = PSI² as primitive N-th root
            pub fn forward(poly: []u64) void {
                std.debug.assert(poly.len == N);

                // Step 1: Negacyclic twist — multiply a[j] by ψ^j
                var psi_pow: u64 = 1;
                for (0..N) |j| {
                    poly[j] = modMul(poly[j], psi_pow);
                    psi_pow = modMul(psi_pow, PSI);
                }

                // Step 2: Bit-reverse permutation
                bitReversePermutation(poly);

                // Step 3: Cooley-Tukey butterfly stages
                var s: usize = 1;
                while (s <= LOG_N) : (s += 1) {
                    const m = @as(usize, 1) << @intCast(s);
                    const half = m / 2;
                    const w_m = modPow(ZETA, N / m);

                    var k: usize = 0;
                    while (k < N) : (k += m) {
                        var w: u64 = 1;
                        for (0..half) |j| {
                            const t = modMul(w, poly[k + j + half]);
                            const u = poly[k + j];
                            poly[k + j] = modAdd(u, t);
                            poly[k + j + half] = modSub(u, t);
                            w = modMul(w, w_m);
                        }
                    }
                }
            }

            /// In-place inverse negacyclic NTT.
            ///
            /// Recovers coefficient-domain polynomial from NTT representation.
            ///
            /// Steps:
            ///   1. Bit-reverse permutation
            ///   2. Butterfly stages with ZETA⁻¹
            ///   3. Post-multiply by N⁻¹ · PSI^{-j}  (untwist + scale)
            pub fn inverse(poly: []u64) void {
                std.debug.assert(poly.len == N);

                const zeta_inv = modInv(ZETA);

                // Step 1: Bit-reverse permutation
                bitReversePermutation(poly);

                // Step 2: Butterfly stages with inverse twiddles
                var s: usize = 1;
                while (s <= LOG_N) : (s += 1) {
                    const m = @as(usize, 1) << @intCast(s);
                    const half = m / 2;
                    const w_m = modPow(zeta_inv, N / m);

                    var k: usize = 0;
                    while (k < N) : (k += m) {
                        var w: u64 = 1;
                        for (0..half) |j| {
                            const t = modMul(w, poly[k + j + half]);
                            const u = poly[k + j];
                            poly[k + j] = modAdd(u, t);
                            poly[k + j + half] = modSub(u, t);
                            w = modMul(w, w_m);
                        }
                    }
                }

                // Step 3: Scale by N⁻¹ and untwist by ψ^{-j}
                const psi_inv = modInv(PSI);
                var psi_inv_pow: u64 = 1;
                for (0..N) |j| {
                    poly[j] = modMul(poly[j], modMul(N_INV, psi_inv_pow));
                    psi_inv_pow = modMul(psi_inv_pow, psi_inv);
                }
            }

            /// Forward then inverse — should recover the original polynomial.
            pub fn identity(poly: []u64) void {
                forward(poly);
                inverse(poly);
            }

            /// Pointwise modular multiplication in NTT domain.
            /// result[i] = a[i] · b[i]  (mod Q)
            pub fn polyMul(a: []const u64, b: []const u64, result: []u64) void {
                std.debug.assert(a.len == N);
                std.debug.assert(b.len == N);
                std.debug.assert(result.len == N);
                for (0..N) |i| {
                    result[i] = modMul(a[i], b[i]);
                }
            }

            // --- ZML Buffer helpers (requires ZML platform) ---

            pub const forwardBuffer = if (is_wasm) void else struct {
                pub fn f(allocator: std.mem.Allocator, platform: *const zml.Platform, io: std.Io, poly_buf: zml.Buffer) !zml.Buffer {
                    const slice = try poly_buf.toSliceAlloc(allocator, io);
                    defer slice.free(allocator);

                    const data = try allocator.alloc(u64, N);
                    defer allocator.free(data);
                    @memcpy(data, slice.items(u64));

                    forward(data);

                    return try zml.Buffer.fromSlice(io, platform, zml.Slice.init(
                        zml.Shape.init(.{N}, .u64),
                        std.mem.sliceAsBytes(data),
                    ));
                }
            }.f;

            pub const inverseBuffer = if (is_wasm) void else struct {
                pub fn f(allocator: std.mem.Allocator, platform: *const zml.Platform, io: std.Io, poly_buf: zml.Buffer) !zml.Buffer {
                    const slice = try poly_buf.toSliceAlloc(allocator, io);
                    defer slice.free(allocator);

                    const data = try allocator.alloc(u64, N);
                    defer allocator.free(data);
                    @memcpy(data, slice.items(u64));

                    inverse(data);

                    return try zml.Buffer.fromSlice(io, platform, zml.Slice.init(
                        zml.Shape.init(.{N}, .u64),
                        std.mem.sliceAsBytes(data),
                    ));
                }
            }.f;
        };

        // =====================================================================
        // ZML Graph-based NTT — Matrix multiply approach.
        // Works correctly for small primes (e.g. Kyber Q=8380417) where the
        // dot-product accumulation doesn't overflow u64.
        // For large primes (Goldilocks), use ButterflyNTT instead.
        // Only available on non-WASM targets.
        // =====================================================================

        pub const NTTModel = if (!is_wasm) struct {
            /// Pre-computed N x N forward negacyclic NTT matrix F_fwd.
            /// result = F_fwd * poly (mod Q)
            F_fwd: zml.Tensor,

            /// Pre-computed N x N inverse negacyclic NTT matrix F_inv.
            /// result = F_inv * poly (mod Q)
            F_inv: zml.Tensor,

            // In ML Frameworks like ZML, we supply the concrete pre-computed Tensors
            // to the execution Host (Cpu, GPU, TPU)
            pub fn init(allocator: std.mem.Allocator, platform: *const zml.Platform, io: std.Io) !zml.Bufferized(@This()) {
                const host_fwd = try allocator.alloc(u64, N * N);
                defer allocator.free(host_fwd);

                const host_inv = try allocator.alloc(u64, N * N);
                defer allocator.free(host_inv);

                // Populate Negacyclic NTT Matrix F and Inverse F^{-1}
                for (0..N) |i| {
                    for (0..N) |j| {
                        const i_u64 = @as(u64, @intCast(i));
                        const j_u64 = @as(u64, @intCast(j));

                        // Forward: PSI^{(2i+1)j} mod Q
                        const fwd_exp = (2 * i_u64 + 1) * j_u64;
                        const fwd_val = modPow(PSI, fwd_exp);
                        host_fwd[i * N + j] = fwd_val;

                        // Inverse: N^{-1} * PSI^{-(2j+1)i} mod Q
                        const inv_exp_pos = (2 * j_u64 + 1) * i_u64;
                        const inv_exp = Q - 1 - (inv_exp_pos % (Q - 1));
                        const inv_psi = modPow(PSI, inv_exp);
                        const inv_val = modMul(inv_psi, N_INV);
                        host_inv[i * N + j] = inv_val;
                    }
                }

                // Push constants to accelerator memory
                const F_fwd_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ N, N }, .u64), std.mem.sliceAsBytes(host_fwd)));
                const F_inv_buf = try zml.Buffer.fromSlice(io, platform, zml.Slice.init(zml.Shape.init(.{ N, N }, .u64), std.mem.sliceAsBytes(host_inv)));

                return .{
                    .F_fwd = F_fwd_buf,
                    .F_inv = F_inv_buf,
                };
            }

            /// Forward ZML Transformation: Time -> Frequency Domain
            pub fn forward(self: @This(), poly: zml.Tensor) zml.Tensor {
                const poly_64 = poly.convert(.u64);

                // F_fwd is generated as row=freq, col=time. Tag as .c (freq) and .n (time)
                const fwd_tagged = self.F_fwd.withTags(.{ .c, .n });

                // Bind poly_64 time dimension to .n implicitly
                const poly_tagged = if (poly_64.rank() == 1) poly_64.withTags(.{.n}) else poly_64.withTags(.{ .b, .n });

                // Batched / 1D Dot product: Sum out exactly the .n axis.
                const res_64 = poly_tagged.dot(fwd_tagged, .n);

                // Modulo Q Reduction
                const q_tensor = zml.Tensor.scalar(Q, .u64);
                const res_mod = res_64.remainder(q_tensor);

                return res_mod;
            }

            /// Inverse ZML Transformation: Frequency -> Time Domain
            pub fn inverse(self: @This(), poly: zml.Tensor) zml.Tensor {
                const poly_64 = poly.convert(.u64);

                // F_inv is generated as row=time, col=freq. Tag as .n (time) and .c (freq)
                const inv_tagged = self.F_inv.withTags(.{ .n, .c });

                // Bind poly_64 freq dimension to .c explicitly
                const poly_tagged = if (poly_64.rank() == 1) poly_64.withTags(.{.c}) else poly_64.withTags(.{ .b, .c });

                // Batched / 1D Dot product: Sum out exactly the .c axis.
                const res_64 = poly_tagged.dot(inv_tagged, .c);

                // Modulo Q Reduction
                const q_tensor = zml.Tensor.scalar(Q, .u64);
                const res_mod = res_64.remainder(q_tensor);

                return res_mod;
            }

            /// Composed Forward and Inverse for testing Identity mapping
            pub fn identity(self: @This(), poly: zml.Tensor) zml.Tensor {
                const fwd = self.forward(poly);
                return self.inverse(fwd.withTags(.{.c}));
            }

            /// Pointwise Matrix Multiplication: Freq * Freq
            pub fn poly_mul(_: @This(), a: zml.Tensor, b: zml.Tensor) zml.Tensor {
                const a_64 = a.convert(.u64);
                const b_64 = b.convert(.u64);

                const prod = a_64.mul(b_64);

                const q_tensor = zml.Tensor.scalar(Q, .u64);
                const res_mod = prod.remainder(q_tensor);

                return res_mod;
            }
        } else void;
    };
}
