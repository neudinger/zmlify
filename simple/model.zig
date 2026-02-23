const std = @import("std");
const zml = @import("zml");

pub const SimpleModel = struct {
    pub fn matmul(_: SimpleModel, lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
        return lhs.dot(rhs, .c);
    }

    pub fn saxpy(_: SimpleModel, a: zml.Tensor, x: zml.Tensor, y: zml.Tensor) zml.Tensor {
        return x.mul(a).add(y);
    }

    pub fn mod_matmul(_: SimpleModel, lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
        const Q = 3329;
        const lhs_64 = lhs.convert(.i64);
        const rhs_64 = rhs.convert(.i64);
        const res_64 = lhs_64.dot(rhs_64, .c);
        const res_mod = res_64.remainder(zml.Tensor.scalar(Q, .i64));
        return res_mod.convert(.i32);
    }
    pub fn heat_transfer(_: SimpleModel, grid: zml.Tensor) zml.Tensor {

        // Neighbors:
        // up: [0..-2, 1..-1]
        const up = grid.slice(&.{ .{ .end = -2 }, .{ .start = 1, .end = -1 } });
        // down: [2.., 1..-1]
        const down = grid.slice(&.{ .{ .start = 2 }, .{ .start = 1, .end = -1 } });
        // left: [1..-1, 0..-2]
        const left = grid.slice(&.{ .{ .start = 1, .end = -1 }, .{ .end = -2 } });
        // right: [1..-1, 2..]
        const right = grid.slice(&.{ .{ .start = 1, .end = -1 }, .{ .start = 2 } });

        const center = up.add(down).add(left).add(right).mul(zml.Tensor.scalar(0.25, .f32));

        // Reconstruct full grid
        const top_row = grid.slice(&.{ .{ .end = 1 }, .{} });
        const bottom_row = grid.slice(&.{ .{ .start = -1 }, .{} });

        const left_col = grid.slice(&.{ .{ .start = 1, .end = -1 }, .{ .end = 1 } });
        const right_col = grid.slice(&.{ .{ .start = 1, .end = -1 }, .{ .start = -1 } });

        // Middle row: left_col ++ center ++ right_col (dim 1)
        const middle_row = zml.Tensor.concatenate(&.{ left_col, center, right_col }, 1);

        // Full grid: top_row ++ middle_row ++ bottom_row (dim 0)
        return zml.Tensor.concatenate(&.{ top_row, middle_row, bottom_row }, 0);
    }
    pub const BlackScholesResult = struct {
        call: zml.Tensor,
        put: zml.Tensor,
    };

    fn std_normal_cdf(x: zml.Tensor) zml.Tensor {
        const p = 0.3275911;
        const a1 = 0.254829592;
        const a2 = -0.284496736;
        const a3 = 1.421413741;
        const a4 = -1.453152027;
        const a5 = 1.061405429;

        const v = x.scale(1.0 / std.math.sqrt(2.0));
        const z = v.abs();

        const t = zml.Tensor.scalar(1.0, .f32).div(z.scale(p).add(zml.Tensor.scalar(1.0, .f32)));

        const poly = t.scale(a5).add(zml.Tensor.scalar(a4, .f32)).mul(t).add(zml.Tensor.scalar(a3, .f32)).mul(t).add(zml.Tensor.scalar(a2, .f32)).mul(t).add(zml.Tensor.scalar(a1, .f32)).mul(t);
        const y = zml.Tensor.scalar(1.0, .f32).sub(poly.mul(z.mul(z).scale(-1.0).exp()));

        const sign = v.cmp(.GE, zml.Tensor.scalar(0.0, .f32)).select(zml.Tensor.scalar(1.0, .f32), zml.Tensor.scalar(-1.0, .f32));
        const erf = sign.mul(y);

        return zml.Tensor.scalar(0.5, .f32).mul(zml.Tensor.scalar(1.0, .f32).add(erf));
    }

    pub fn black_scholes(_: SimpleModel, s: zml.Tensor, k: zml.Tensor, t: zml.Tensor, r: zml.Tensor, sigma: zml.Tensor) BlackScholesResult {
        const sqrt_t = t.sqrt();
        const sigma_sqrt_t = sigma.mul(sqrt_t);

        const log_s_k = s.div(k).log();
        const r_plus_half_var = r.add(sigma.mul(sigma).scale(0.5));

        const d1 = log_s_k.add(r_plus_half_var.mul(t)).div(sigma_sqrt_t);
        const d2 = d1.sub(sigma_sqrt_t);

        const cdf_d1 = std_normal_cdf(d1);
        const cdf_d2 = std_normal_cdf(d2);

        const k_exp_rt = k.mul(r.mul(t).scale(-1.0).exp());
        const call = s.mul(cdf_d1).sub(k_exp_rt.mul(cdf_d2));

        const cdf_neg_d1 = std_normal_cdf(d1.scale(-1.0));
        const cdf_neg_d2 = std_normal_cdf(d2.scale(-1.0));
        const put = k_exp_rt.mul(cdf_neg_d2).sub(s.mul(cdf_neg_d1));

        return .{ .call = call, .put = put };
    }
};
