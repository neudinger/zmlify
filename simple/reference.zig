const std = @import("std");

pub fn matmul(M: usize, K: usize, N: usize, lhs: []const f32, rhs: []const f32, out: []f32) void {
    for (0..M) |m| {
        for (0..N) |n| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += lhs[m * K + k] * rhs[k * N + n];
            }
            out[m * N + n] = sum;
        }
    }
}

pub fn saxpy(n: usize, a: f32, x: []const f32, y: []const f32, out: []f32) void {
    for (0..n) |i| {
        out[i] = a * x[i] + y[i];
    }
}

pub fn mod_matmul(M: usize, K: usize, N: usize, lhs: []const i32, rhs: []const i32, out: []i32) void {
    const Q: i64 = 3329;
    for (0..M) |m| {
        for (0..N) |n| {
            var sum: i64 = 0;
            for (0..K) |k| {
                const a: i64 = lhs[m * K + k];
                const b: i64 = rhs[k * N + n];
                sum += a * b;
            }
            out[m * N + n] = @intCast(@mod(sum, Q));
        }
    }
}

pub fn heat_transfer(H: usize, W: usize, grid: []const f32, out: []f32) void {
    @memcpy(out, grid);
    for (1..H - 1) |i| {
        for (1..W - 1) |j| {
            const up = grid[(i - 1) * W + j];
            const down = grid[(i + 1) * W + j];
            const left = grid[i * W + j - 1];
            const right = grid[i * W + j + 1];
            out[i * W + j] = 0.25 * (up + down + left + right);
        }
    }
}

fn std_normal_cdf(x: f32) f32 {
    const p = 0.3275911;
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;

    const v = x / @sqrt(2.0);
    const z = @abs(v);

    const t = 1.0 / (1.0 + p * z);
    const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    const y = 1.0 - poly * @exp(-z * z);

    const sign: f32 = if (v >= 0) 1.0 else -1.0;
    const erf = sign * y;

    return 0.5 * (1.0 + erf);
}

pub fn black_scholes(n: usize, s: []const f32, k: []const f32, t: []const f32, r: []const f32, sigma: []const f32, call: []f32, put: []f32) void {
    for (0..n) |i| {
        const sqrt_t = @sqrt(t[i]);
        const sigma_sqrt_t = sigma[i] * sqrt_t;
        const d1 = (@log(s[i] / k[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * t[i]) / sigma_sqrt_t;
        const d2 = d1 - sigma_sqrt_t;

        const cdf_d1 = std_normal_cdf(d1);
        const cdf_d2 = std_normal_cdf(d2);
        const k_exp_rt = k[i] * @exp(-r[i] * t[i]);

        call[i] = s[i] * cdf_d1 - k_exp_rt * cdf_d2;

        const cdf_neg_d1 = std_normal_cdf(-d1);
        const cdf_neg_d2 = std_normal_cdf(-d2);
        put[i] = k_exp_rt * cdf_neg_d2 - s[i] * cdf_neg_d1;
    }
}
