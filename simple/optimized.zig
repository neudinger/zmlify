const std = @import("std");

const max_threads = 16;

fn getNumThreads() usize {
    return std.Thread.getCpuCount() catch 1;
}

pub fn saxpy(n: usize, a: f32, x: []const f32, y: []const f32, out: []f32) void {
    const num_threads = @min(getNumThreads(), max_threads);
    if (num_threads <= 1 or n < 1024) {
        saxpyKernel(a, x[0..n], y[0..n], out[0..n]);
        return;
    }

    const chunk_size = (n + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const start = t * chunk_size;
        if (start >= n) break;
        const end = @min(start + chunk_size, n);
        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(a_: f32, x_: []const f32, y_: []const f32, out_: []f32) void {
                saxpyKernel(a_, x_, y_, out_);
            }
        }.worker, .{ a, x[start..end], y[start..end], out[start..end] }) catch {
            // Fallback: do this chunk on the current thread
            saxpyKernel(a, x[start..end], y[start..end], out[start..end]);
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| t.join();
}

fn saxpyKernel(a: f32, x: []const f32, y: []const f32, out: []f32) void {
    const VL = 64;
    const Vec = @Vector(VL, f32);
    const len = x.len;

    var i: usize = 0;
    while (i + VL <= len) : (i += VL) {
        const x_vec: Vec = x[i..][0..VL].*;
        const y_vec: Vec = y[i..][0..VL].*;
        const a_vec: Vec = @splat(a);
        out[i..][0..VL].* = x_vec * a_vec + y_vec;
    }
    while (i < len) : (i += 1) {
        out[i] = a * x[i] + y[i];
    }
}

pub fn matmul(M: usize, K: usize, N: usize, lhs: []const f32, rhs: []const f32, out: []f32) void {
    const MC = 64;
    const num_threads = @min(getNumThreads(), max_threads);

    @memset(out, 0);

    if (num_threads <= 1 or M < MC) {
        matmulKernel(0, M, K, N, lhs, rhs, out);
        return;
    }

    // Each thread gets a range of row-blocks
    const num_row_blocks = (M + MC - 1) / MC;
    const blocks_per_thread = (num_row_blocks + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const block_start = t * blocks_per_thread;
        if (block_start >= num_row_blocks) break;
        const block_end = @min(block_start + blocks_per_thread, num_row_blocks);
        const m_start = block_start * MC;
        const m_end = @min(block_end * MC, M);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(m_s: usize, m_e: usize, K_: usize, N_: usize, lhs_: []const f32, rhs_: []const f32, out_: []f32) void {
                matmulKernel(m_s, m_e, K_, N_, lhs_, rhs_, out_);
            }
        }.worker, .{ m_start, m_end, K, N, lhs, rhs, out }) catch {
            matmulKernel(m_start, m_end, K, N, lhs, rhs, out);
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| t.join();
}

fn matmulKernel(m_start: usize, m_end: usize, K: usize, N: usize, lhs: []const f32, rhs: []const f32, out: []f32) void {
    const NC = 64;
    const KC = 64;

    var n: usize = 0;
    while (n < N) : (n += NC) {
        const n_end = @min(N, n + NC);
        var k: usize = 0;
        while (k < K) : (k += KC) {
            const k_end = @min(K, k + KC);
            for (m_start..m_end) |i| {
                for (n..n_end) |j| {
                    var sum: f32 = out[i * N + j];
                    for (k..k_end) |p| {
                        sum += lhs[i * K + p] * rhs[p * N + j];
                    }
                    out[i * N + j] = sum;
                }
            }
        }
    }
}

pub fn mod_matmul(M: usize, K: usize, N: usize, lhs: []const i32, rhs: []const i32, out: []i32) void {
    const MC = 64;
    const num_threads = @min(getNumThreads(), max_threads);

    @memset(out, 0);

    if (num_threads <= 1 or M < MC) {
        modMatmulKernel(0, M, K, N, lhs, rhs, out);
        return;
    }

    const num_row_blocks = (M + MC - 1) / MC;
    const blocks_per_thread = (num_row_blocks + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const block_start = t * blocks_per_thread;
        if (block_start >= num_row_blocks) break;
        const block_end = @min(block_start + blocks_per_thread, num_row_blocks);
        const m_start = block_start * MC;
        const m_end = @min(block_end * MC, M);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(m_s: usize, m_e: usize, K_: usize, N_: usize, lhs_: []const i32, rhs_: []const i32, out_: []i32) void {
                modMatmulKernel(m_s, m_e, K_, N_, lhs_, rhs_, out_);
            }
        }.worker, .{ m_start, m_end, K, N, lhs, rhs, out }) catch {
            modMatmulKernel(m_start, m_end, K, N, lhs, rhs, out);
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| t.join();
}

fn modMatmulKernel(m_start: usize, m_end: usize, K: usize, N: usize, lhs: []const i32, rhs: []const i32, out: []i32) void {
    const Q: i64 = 3329;
    const VL = 16;
    const Vec = @Vector(VL, i32);
    const Vec64 = @Vector(VL, i64);
    const NC = 64;

    var n: usize = 0;
    while (n < N) : (n += NC) {
        const n_end = @min(N, n + NC);
        for (m_start..m_end) |i| {
            var j: usize = n;
            while (j + VL <= n_end) : (j += VL) {
                var sum_vec: Vec64 = @splat(0);
                var k: usize = 0;
                while (k < K) : (k += 1) {
                    const a_val: i64 = lhs[i * K + k];
                    const a_vec: Vec64 = @splat(a_val);
                    const b_vec: Vec = rhs[k * N + j ..][0..VL].*;
                    var b_vec64: Vec64 = undefined;
                    const b_arr: [VL]i32 = b_vec;
                    inline for (b_arr, 0..) |b, idx| b_vec64[idx] = b;
                    sum_vec += a_vec * b_vec64;
                }
                var out_arr: [VL]i32 = undefined;
                const sum_arr: [VL]i64 = sum_vec;
                for (sum_arr, 0..) |s, idx| {
                    out_arr[idx] = @intCast(@mod(s, Q));
                }
                const out_vec: Vec = out_arr;
                out[i * N + j ..][0..VL].* = out_vec;
            }
            while (j < n_end) : (j += 1) {
                var sum: i64 = 0;
                for (0..K) |k| {
                    sum += @as(i64, lhs[i * K + k]) * @as(i64, rhs[k * N + j]);
                }
                out[i * N + j] = @intCast(@mod(sum, Q));
            }
        }
    }
}

pub fn heat_transfer(H: usize, W: usize, grid: []const f32, out: []f32) void {
    const num_threads = @min(getNumThreads(), max_threads);

    // Copy everything first to preserve boundaries
    @memcpy(out, grid);

    if (num_threads <= 1 or H < 64) {
        heatTransferKernel(1, H - 1, W, grid, out);
        return;
    }

    const rows = H - 2; // Rows to process: 1..H-1
    const chunk_size = (rows + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const r_start = 1 + t * chunk_size;
        if (r_start >= H - 1) break;
        const r_end = @min(r_start + chunk_size, H - 1);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(start_row: usize, end_row: usize, Width: usize, g: []const f32, o: []f32) void {
                heatTransferKernel(start_row, end_row, Width, g, o);
            }
        }.worker, .{ r_start, r_end, W, grid, out }) catch {
            heatTransferKernel(r_start, r_end, W, grid, out);
            continue;
        };
        spawned += 1;
    }
    for (threads[0..spawned]) |t| t.join();
}

fn heatTransferKernel(row_start: usize, row_end: usize, W: usize, grid: []const f32, out: []f32) void {
    const VL = 64;
    const Vec = @Vector(VL, f32);
    const quarter: Vec = @splat(0.25);

    for (row_start..row_end) |i| {
        var j: usize = 1;
        while (j + VL <= W - 1) : (j += VL) {
            const up: Vec = grid[(i - 1) * W + j ..][0..VL].*;
            const down: Vec = grid[(i + 1) * W + j ..][0..VL].*;
            const left: Vec = grid[i * W + j - 1 ..][0..VL].*;
            const right: Vec = grid[i * W + j + 1 ..][0..VL].*;

            out[i * W + j ..][0..VL].* = (up + down + left + right) * quarter;
        }
        while (j < W - 1) : (j += 1) {
            const up = grid[(i - 1) * W + j];
            const down = grid[(i + 1) * W + j];
            const left = grid[i * W + j - 1];
            const right = grid[i * W + j + 1];
            out[i * W + j] = 0.25 * (up + down + left + right);
        }
    }
}

fn stdNormalCdf(comptime VL: usize, x: @Vector(VL, f32)) @Vector(VL, f32) {
    const Vec = @Vector(VL, f32);
    const p: Vec = @splat(0.3275911);
    const a1: Vec = @splat(0.254829592);
    const a2: Vec = @splat(-0.284496736);
    const a3: Vec = @splat(1.421413741);
    const a4: Vec = @splat(-1.453152027);
    const a5: Vec = @splat(1.061405429);
    const one: Vec = @splat(1.0);
    const zero: Vec = @splat(0.0);
    const half: Vec = @splat(0.5);
    const sqrt2_inv: Vec = @splat(1.0 / std.math.sqrt(2.0));

    const v = x * sqrt2_inv;
    const z = @abs(v);

    const t = one / (one + p * z);
    const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    const y = one - poly * @exp(-z * z);

    const sign_mask = v >= zero;
    const sign = @select(f32, sign_mask, one, @as(Vec, @splat(-1.0)));
    const erf = sign * y;

    return half * (one + erf);
}

pub fn black_scholes(n: usize, s: []const f32, k: []const f32, t: []const f32, r: []const f32, sigma: []const f32, call: []f32, put: []f32) void {
    const num_threads = @min(getNumThreads(), max_threads);
    if (num_threads <= 1 or n < 1024) {
        blackScholesKernel(n, s, k, t, r, sigma, call, put);
        return;
    }

    const chunk_size = (n + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |th| {
        const start = th * chunk_size;
        if (start >= n) break;
        const end = @min(start + chunk_size, n);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(len: usize, s_: []const f32, k_: []const f32, t_: []const f32, r_: []const f32, sigma_: []const f32, call_: []f32, put_: []f32) void {
                blackScholesKernel(len, s_, k_, t_, r_, sigma_, call_, put_);
            }
        }.worker, .{ end - start, s[start..end], k[start..end], t[start..end], r[start..end], sigma[start..end], call[start..end], put[start..end] }) catch {
            blackScholesKernel(end - start, s[start..end], k[start..end], t[start..end], r[start..end], sigma[start..end], call[start..end], put[start..end]);
            continue;
        };
        spawned += 1;
    }
    for (threads[0..spawned]) |th| th.join();
}

fn blackScholesKernel(n: usize, s: []const f32, k: []const f32, t: []const f32, r: []const f32, sigma: []const f32, call: []f32, put: []f32) void {
    const VL = 64;
    const Vec = @Vector(VL, f32);

    var i: usize = 0;
    while (i + VL <= n) : (i += VL) {
        const s_vec: Vec = s[i..][0..VL].*;
        const k_vec: Vec = k[i..][0..VL].*;
        const t_vec: Vec = t[i..][0..VL].*;
        const r_vec: Vec = r[i..][0..VL].*;
        const sigma_vec: Vec = sigma[i..][0..VL].*;

        const sqrt_t = @sqrt(t_vec);
        const sigma_sqrt_t = sigma_vec * sqrt_t;
        const half_sigma2 = @as(Vec, @splat(0.5)) * sigma_vec * sigma_vec;
        const d1 = (@log(s_vec / k_vec) + (r_vec + half_sigma2) * t_vec) / sigma_sqrt_t;
        const d2 = d1 - sigma_sqrt_t;

        const cdf_d1 = stdNormalCdf(VL, d1);
        const cdf_d2 = stdNormalCdf(VL, d2);
        const k_exp_rt = k_vec * @exp(-r_vec * t_vec);

        call[i..][0..VL].* = s_vec * cdf_d1 - k_exp_rt * cdf_d2;

        const cdf_neg_d1 = stdNormalCdf(VL, -d1);
        const cdf_neg_d2 = stdNormalCdf(VL, -d2);
        put[i..][0..VL].* = k_exp_rt * cdf_neg_d2 - s_vec * cdf_neg_d1;
    }

    while (i < n) : (i += 1) {
        const sqrt_t = @sqrt(t[i]);
        const sigma_sqrt_t = sigma[i] * sqrt_t;
        const d1 = (@log(s[i] / k[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * t[i]) / sigma_sqrt_t;
        const d2 = d1 - sigma_sqrt_t;

        const cdf_d1 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(d1)))[0];
        const cdf_d2 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(d2)))[0];
        const k_exp_rt = k[i] * @exp(-r[i] * t[i]);

        call[i] = s[i] * cdf_d1 - k_exp_rt * cdf_d2;

        const cdf_neg_d1 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(-d1)))[0];
        const cdf_neg_d2 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(-d2)))[0];
        put[i] = k_exp_rt * cdf_neg_d2 - s[i] * cdf_neg_d1;
    }
}
