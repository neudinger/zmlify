/// Zerkerus ZKP Client — WebAssembly Entry Point
///
/// Exports the client-side protocol functions for JavaScript consumption.
/// Uses pure-Zig math (matVecMulPure) instead of ZML GPU operations.
/// Exchanges raw u64 arrays instead of FlatBuffers serialization.
const std = @import("std");
const params = @import("params.zig");
const utils = @import("utils.zig");
const math = @import("math.zig");

// WASM allocator - thin and efficient for single-threaded WASM
const allocator = std.heap.wasm_allocator;

// --- Client State (kept in WASM memory, referenced by opaque pointer) ---

const WasmClientState = struct {
    secret_seed: [32]u8,
    s_poly: []const u64,
    a_poly: []const u64,
    // Commitment state (kept between commit and response phases)
    y_poly: ?[]const u64,

    pub fn deinit(self: *WasmClientState) void {
        allocator.free(self.s_poly);
        allocator.free(self.a_poly);
        if (self.y_poly) |y| allocator.free(y);
        allocator.destroy(self);
    }
};

// --- Last buffer tracking (for JS to query buffer sizes) ---
var last_buf_ptr: ?[*]u64 = null;
var last_buf_len: usize = 0;

// --- Memory Management (exported to JS) ---

export fn alloc(len: usize) ?[*]u8 {
    const buf = allocator.alloc(u8, len) catch return null;
    return buf.ptr;
}

export fn dealloc(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

export fn free_u64(ptr: [*]u64, len: usize) void {
    allocator.free(ptr[0..len]);
}

// --- Dimension Getters ---

export fn get_n() usize {
    return params.ntt.N;
}

export fn get_m() usize {
    return params.ntt.M;
}

export fn get_q_lo() u32 {
    return @truncate(params.ntt.Q);
}

export fn get_q_hi() u32 {
    return @truncate(params.ntt.Q >> 32);
}

// --- Protocol Phase 1: Client Registration ---

/// Initialize client state: derive secret key and public key.
/// Returns opaque pointer to WasmClientState, or null on failure.
export fn client_init() ?*anyopaque {
    const state = allocator.create(WasmClientState) catch return null;

    // Derive keys from deterministic seeds (in production, these come from user input)
    const public_seed = utils.extractSeed("PublicSharedMatrixSeed");
    const a_poly = math.expandA(allocator, public_seed) catch {
        allocator.destroy(state);
        return null;
    };

    const secret_seed = utils.extractSeed("UserPasswordSalt");
    const s_poly = math.sampleSmall(allocator, secret_seed, params.B) catch {
        allocator.free(a_poly);
        allocator.destroy(state);
        return null;
    };

    state.* = .{
        .secret_seed = secret_seed,
        .s_poly = s_poly,
        .a_poly = a_poly,
        .y_poly = null,
    };

    return state;
}

/// Deinitialize and free client state.
export fn client_deinit(ctx: *anyopaque) void {
    const state: *WasmClientState = @ptrCast(@alignCast(ctx));
    state.deinit();
}

/// Compute public key t = A * s mod Q.
/// Returns pointer to t_poly (M elements of u64), or null on failure.
/// Caller must free with free_u64(ptr, M).
export fn client_registration(ctx: *anyopaque) ?[*]const u64 {
    const state: *WasmClientState = @ptrCast(@alignCast(ctx));

    const t_poly = math.matVecMulPure(allocator, state.a_poly, state.s_poly) catch return null;

    last_buf_ptr = @constCast(t_poly.ptr);
    last_buf_len = t_poly.len;

    return t_poly.ptr;
}

/// Get the 32-byte public seed. Returns pointer to 32 bytes.
export fn client_get_public_seed() ?[*]const u8 {
    const seed = utils.extractSeed("PublicSharedMatrixSeed");
    // Copy to allocated memory so JS can read it
    const buf = allocator.alloc(u8, 32) catch return null;
    @memcpy(buf, &seed);
    return buf.ptr;
}

// --- Protocol Phase 3: Client Commitment ---

/// Generate commitment w = A * y mod Q.
/// Returns pointer to w_poly (M elements of u64), or null on failure.
/// Caller must free with free_u64(ptr, M).
export fn client_commit(ctx: *anyopaque, attempt: u32) ?[*]const u64 {
    const state: *WasmClientState = @ptrCast(@alignCast(ctx));

    // Free previous y_poly if exists
    if (state.y_poly) |old_y| {
        allocator.free(old_y);
        state.y_poly = null;
    }

    // Generate mask y from (secret_seed || attempt)
    var y_seed_input: [36]u8 = undefined;
    @memcpy(y_seed_input[0..32], &state.secret_seed);
    std.mem.writeInt(u32, y_seed_input[32..36], attempt, .little);
    const iter_seed = utils.extractSeed(&y_seed_input);

    const limit_y = 8192 * params.B;
    const y_poly = math.sampleSmall(allocator, iter_seed, limit_y) catch return null;
    state.y_poly = y_poly;

    // Compute w = A * y mod Q
    const w_poly = math.matVecMulPure(allocator, state.a_poly, y_poly) catch return null;

    last_buf_ptr = @constCast(w_poly.ptr);
    last_buf_len = w_poly.len;

    return w_poly.ptr;
}

// --- Protocol Phase 5: Client Response ---

/// Compute response z = y + c * s mod Q, with rejection sampling.
/// Returns pointer to z_poly (N elements of u64), or null if rejected or error.
/// Caller must free with free_u64(ptr, N) on success.
export fn client_response(ctx: *anyopaque, challenge: u64) ?[*]const u64 {
    const state: *WasmClientState = @ptrCast(@alignCast(ctx));

    const y_poly = state.y_poly orelse return null;

    // z = y + c * s mod Q
    const z_poly = math.vecAddScalarMul(allocator, y_poly, challenge, state.s_poly) catch return null;

    // Rejection sampling
    const limit_y = 8192 * params.B;
    const safe_limit = limit_y - (1 * params.B);

    if (!math.isSmall(z_poly, safe_limit, null)) {
        // Unsafe — reject this attempt
        allocator.free(z_poly);
        return null;
    }

    last_buf_ptr = @constCast(z_poly.ptr);
    last_buf_len = z_poly.len;

    return z_poly.ptr;
}

// --- Verification (for testing — normally server-side) ---

/// Verify proof: check A*z == w + c*t (mod Q) and ||z|| <= limit.
/// All arrays are passed as pointers with known sizes from get_n()/get_m().
export fn client_verify(
    a_poly_ptr: [*]const u64,
    t_poly_ptr: [*]const u64,
    w_poly_ptr: [*]const u64,
    challenge: u64,
    z_poly_ptr: [*]const u64,
) bool {
    const a_poly = a_poly_ptr[0 .. params.ntt.M * params.ntt.N];
    const t_poly = t_poly_ptr[0..params.ntt.M];
    const w_poly = w_poly_ptr[0..params.ntt.M];
    const z_poly = z_poly_ptr[0..params.ntt.N];

    // LHS: A * z mod Q
    const lhs = math.matVecMulPure(allocator, a_poly, z_poly) catch return false;
    defer allocator.free(lhs);

    // RHS: w + c * t mod Q
    const rhs = math.vecAddScalarMul(allocator, w_poly, challenge, t_poly) catch return false;
    defer allocator.free(rhs);

    // Check LHS == RHS
    for (lhs, 0..) |val, i| {
        if (val != rhs[i]) return false;
    }

    // Check ||z|| <= limit_y + B
    const limit_y = 8192 * params.B;
    if (!math.isSmall(z_poly, limit_y + params.B, null)) return false;

    return true;
}

// --- Helper Getters ---

export fn get_last_buf_len() usize {
    return last_buf_len;
}
