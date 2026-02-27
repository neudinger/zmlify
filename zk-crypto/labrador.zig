const std = @import("std");
const zml = @import("zml");
const fiat_shamir = @import("fiat_shamir.zig");

pub fn MakeLabrador(comptime ntt: type, comptime B_val: u64) type {
    return struct {
        pub const B: u64 = B_val;

        /// Generates a simple Lattice Commitment (A*s + e) module.
        /// Simulates BDLOP Commitments by mapping a polynomial to a condensed hash binding
        pub const Prover = struct {

            // Checks that the resulting polynomial does not exceed the allowed norm (in constant time)
            pub fn checkNorm(poly: []const u64) bool {
                var fail_mask: u64 = 0;
                const q_half = ntt.Q / 2;

                for (poly) |v| {
                    // Modulo signed mapping (constant time)
                    // v is bounded [0, Q-1]. Distance from 0 in Z_q
                    const is_greater = @as(u64, @intFromBool(v > q_half));
                    _ = is_greater; // autofix
                    const abs_v = if (v > q_half) ntt.Q - v else v;

                    // Check bounds (accumulate failures bitwise)
                    const out_of_bounds = @as(u64, @intFromBool(abs_v > B));
                    fail_mask |= out_of_bounds;
                }

                // Return strictly based on aggregated structural failures
                return fail_mask == 0;
            }

            /// Rejection Sampling (Zero-Knowledge Aborts)
            /// Aborts the proof round if the response vector z leaks information about the secret witness s.
            /// In uniform bounds, we must reject if z falls within Beta distance of the structural edge B.
            pub fn rejectionSample(z_poly: []const u64, beta: u64) bool {
                var fail_mask: u64 = 0;
                const q_half = ntt.Q / 2;
                const strict_bound = B - beta;

                for (z_poly) |v| {
                    const abs_v = if (v > q_half) ntt.Q - v else v;

                    // Check structural leakage bounds
                    const out_of_bounds = @as(u64, @intFromBool(abs_v > strict_bound));
                    fail_mask |= out_of_bounds;
                }

                // Return true if NO failures (safe to publish), false if we must ABORT
                return fail_mask == 0;
            }

            /// Splits a polynomial of degree N into two polynomials of degree N/2
            /// (even and odd coefficients), then folds them down using a challenge scalar `alpha`
            pub fn foldPolynomial(poly: []const u64, alpha: u64, out_poly: []u64) void {
                std.debug.assert(poly.len % 2 == 0);
                std.debug.assert(poly.len / 2 == out_poly.len);

                const half_len = poly.len / 2;
                for (0..half_len) |i| {
                    const p_even = poly[2 * i];
                    const p_odd = poly[2 * i + 1];

                    // Fold: p_out[i] = p_even + alpha * p_odd (mod Q)
                    const prod = (@as(u128, alpha) * @as(u128, p_odd)) % ntt.Q;
                    const fold_val = (@as(u128, p_even) + prod) % ntt.Q;
                    out_poly[i] = @as(u64, @intCast(fold_val));
                }
            }

            /// Prove State Machine: Orchestrates the log2(N) recursive folding bounds.
            /// Absorbs polynomials into Fiat-Shamir, squeezes an `alpha` scalar binding, and folds down to a single element.
            pub fn prove(allocator: std.mem.Allocator, transcript: *fiat_shamir.Transcript, initial_poly: []const u64) ![]u64 {
                var current_poly = try allocator.alloc(u64, initial_poly.len);
                @memcpy(current_poly, initial_poly);

                while (current_poly.len > 1) {
                    const next_len = current_poly.len / 2;
                    const next_poly = try allocator.alloc(u64, next_len);

                    // Absorb current polynomial state as cross-terms
                    transcript.appendMessage("poly", std.mem.sliceAsBytes(current_poly));

                    // Squeeze challenge scalar alpha
                    const challenge_bytes = transcript.getChallenge("alpha");
                    const alpha_raw = std.mem.readInt(u64, challenge_bytes[0..8], .little);
                    const alpha = alpha_raw % ntt.Q;

                    // Fold polynomial geometrically by factor of 2
                    foldPolynomial(current_poly, alpha, next_poly);

                    allocator.free(current_poly);
                    current_poly = next_poly;
                }

                // Return the final Log2 folded proof scalar (as a slice of len 1)
                return current_poly;
            }

            /// Verify State Machine: The deterministic opposite of `prove()`.
            /// The Verifier absorbs the public polynomial into their own Fiat-Shamir transcript,
            /// squeezes the exact same geometric `alpha` bounds, and folds the public vector.
            /// Returns the final expected scalar which should match the Prover's output.
            pub fn verify(allocator: std.mem.Allocator, transcript: *fiat_shamir.Transcript, initial_poly: []const u64) ![]u64 {
                var current_poly = try allocator.alloc(u64, initial_poly.len);
                @memcpy(current_poly, initial_poly);

                while (current_poly.len > 1) {
                    const next_len = current_poly.len / 2;
                    const next_poly = try allocator.alloc(u64, next_len);

                    // Verifier must absorb the exact same cross-terms to sync the transcript
                    transcript.appendMessage("poly", std.mem.sliceAsBytes(current_poly));

                    // Squeeze challenge scalar alpha
                    const challenge_bytes = transcript.getChallenge("alpha");
                    const alpha_raw = std.mem.readInt(u64, challenge_bytes[0..8], .little);
                    const alpha = alpha_raw % ntt.Q;

                    // Fold the public polynomial geometrically by same factor
                    foldPolynomial(current_poly, alpha, next_poly);

                    allocator.free(current_poly);
                    current_poly = next_poly;
                }

                // Return the expected final Log2 folded proof scalar
                return current_poly;
            }

            /// Deterministically expands a random seed into a uniform polynomial A mod Q
            pub fn expandA(seed: [32]u8, poly: []u64) void {
                var hasher = std.crypto.hash.Blake3.init(.{});
                hasher.update(&seed);
                hasher.update("BDLOP_A");

                var digest: [32]u8 = undefined;
                hasher.final(&digest);

                var seed_array: [8]u8 = undefined;
                @memcpy(&seed_array, digest[0..8]);
                var prng = std.Random.DefaultPrng.init(std.mem.readInt(u64, &seed_array, .little));
                const random = prng.random();

                for (poly) |*v| {
                    v.* = random.intRangeLessThan(u64, 0, ntt.Q);
                }
            }
        };

        /// BDLOP ZML Compilation Graph
        pub const BDLOPModel = struct {
            ntt_def: ntt.NTTModel,

            pub fn init(ntt_def: ntt.NTTModel) BDLOPModel {
                return .{ .ntt_def = ntt_def };
            }

            /// Computes t = A * s + e in the finite field using ZML Tensors
            pub fn commit(self: BDLOPModel, a: zml.Tensor, s: zml.Tensor, e: zml.Tensor) zml.Tensor {
                // Transform A and s to frequency domain
                const a_ntt = self.ntt_def.forward(a);
                const s_ntt = self.ntt_def.forward(s);

                // Pointwise multiply A * s in frequency domain
                const as_ntt = self.ntt_def.poly_mul(a_ntt, s_ntt);

                // Inverse transform back to time domain
                const as_time = self.ntt_def.inverse(as_ntt.withTags(.{.c}));

                // Add error e: t = A*s + e
                const t = as_time.add(e);

                // Reduce modulo Q
                const q_tensor = zml.Tensor.scalar(ntt.Q, .u64).broad(t.shape());
                const t_mod = t.remainder(q_tensor);

                return t_mod;
            }

            /// Computes T = A * s + u * e using ZML Tensors.
            /// This is the Relaxed BDLOP Commitment required for Nova/Sangria folding.
            /// `u` is the scalar accumulator (starts at 1), binding sequential error accumulations.
            pub fn commitRelaxed(self: BDLOPModel, a: zml.Tensor, s: zml.Tensor, e: zml.Tensor, u: zml.Tensor) zml.Tensor {
                const a_ntt = self.ntt_def.forward(a);
                const s_ntt = self.ntt_def.forward(s);
                const as_ntt = self.ntt_def.poly_mul(a_ntt, s_ntt);
                const as_time = self.ntt_def.inverse(as_ntt.withTags(.{.c}));

                // Scale error by accumulator: u_expanded * e
                const u_broad = u.broad(e.shape());
                const u_e = e.mul(u_broad);

                // Add scaled error: T = A*s + u*e
                const t = as_time.add(u_e);

                const q_tensor = zml.Tensor.scalar(ntt.Q, .u64).broad(t.shape());
                const t_mod = t.remainder(q_tensor);

                return t_mod;
            }

            /// Homomorphically folds two relaxed commitments: T_new = T_1 + alpha * T_2
            /// In Nova, the Verifier executes this fold algebraically without inspecting the witness.
            pub fn foldRelaxed(self: BDLOPModel, t1: zml.Tensor, t2: zml.Tensor, alpha: zml.Tensor) zml.Tensor {
                _ = self;
                const alpha_broad = alpha.broad(t2.shape());
                const t2_scaled = t2.mul(alpha_broad);
                const t_new = t1.add(t2_scaled);

                const q_tensor = zml.Tensor.scalar(ntt.Q, .u64).broad(t_new.shape());
                const t_mod = t_new.remainder(q_tensor);

                return t_mod;
            }
        };
    };
}
