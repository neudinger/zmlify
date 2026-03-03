/**
 * Zerkerus ZKP WASM — Automated Test Suite
 *
 * Run with: node zerkerus.test.js
 * Expects zerkerus.wasm in the same directory.
 */

const fs = require('fs');
const path = require('path');
const { webcrypto } = require('crypto');

let wasm = null;
let memory = null;
let passed = 0;
let failed = 0;

function assert(cond, msg) {
    if (!cond) {
        console.error(`  ✗ FAIL: ${msg}`);
        failed++;
    } else {
        console.log(`  ✓ ${msg}`);
        passed++;
    }
}

// WASI shim
const wasi = {
    args_sizes_get: () => 0,
    args_get: () => 0,
    environ_sizes_get: () => 0,
    environ_get: () => 0,
    clock_time_get: () => 0,
    fd_write: (fd, iovs, iovsLen, nwritten) => {
        const view = new DataView(memory.buffer);
        let written = 0;
        for (let i = 0; i < iovsLen; i++) {
            const ptr = view.getUint32(iovs + i * 8, true);
            const len = view.getUint32(iovs + i * 8 + 4, true);
            written += len;
        }
        view.setUint32(nwritten, written, true);
        return 0;
    },
    fd_close: () => 0,
    fd_seek: () => 0,
    proc_exit: () => { },
    random_get: (buf, buf_len) => {
        const arr = new Uint8Array(memory.buffer, buf, buf_len);
        webcrypto.getRandomValues(arr);
        return 0;
    }
};

async function loadWasm() {
    const wasmPath = path.join(__dirname, 'zerkerus.wasm');
    const bytes = fs.readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(bytes, {
        env: {},
        wasi_snapshot_preview1: wasi
    });
    wasm = instance.exports;
    memory = wasm.memory;
}

// ─── Tests ───────────────────────────────────────────────────────

function test_01_exports() {
    console.log('\nTest 1: WASM exports expected functions');
    const expected = [
        'alloc', 'dealloc', 'free_u64',
        'get_n', 'get_m', 'get_q_lo', 'get_q_hi',
        'client_init', 'client_deinit',
        'client_registration', 'client_commit', 'client_response',
        'client_verify', 'client_get_public_seed',
        'get_last_buf_len', 'memory'
    ];
    for (const name of expected) {
        assert(name in wasm, `export '${name}' exists`);
    }
}

function test_02_dimensions() {
    console.log('\nTest 2: Dimensions match debug profile');
    const N = wasm.get_n();
    const M = wasm.get_m();
    assert(N === 64, `N = ${N} (expected 64)`);
    assert(M === 16, `M = ${M} (expected 16)`);

    const Q_lo = wasm.get_q_lo();
    const Q_hi = wasm.get_q_hi();
    const Q = (BigInt(Q_hi) << 32n) | BigInt(Q_lo);
    assert(Q === 8380417n, `Q = ${Q} (expected 8380417)`);
}

function test_03_alloc_dealloc() {
    console.log('\nTest 3: alloc/dealloc round-trip');
    const ptr = wasm.alloc(1024);
    assert(ptr !== 0, 'alloc(1024) returns non-null');
    // Write and read back
    const view = new Uint8Array(memory.buffer, ptr, 1024);
    view[0] = 42;
    view[1023] = 99;
    assert(view[0] === 42, 'memory write/read works');
    assert(view[1023] === 99, 'memory boundary write/read works');
    wasm.dealloc(ptr, 1024);
    assert(true, 'dealloc completes without crash');
}

function test_04_client_init_deinit() {
    console.log('\nTest 4: client_init/deinit lifecycle');
    const ctx = wasm.client_init();
    assert(ctx !== 0 && ctx !== null, 'client_init returns non-null context');
    wasm.client_deinit(ctx);
    assert(true, 'client_deinit completes without crash');
}

function test_05_registration() {
    console.log('\nTest 5: client_registration produces M-length key');
    const ctx = wasm.client_init();
    assert(ctx !== 0, 'context created');

    const M = wasm.get_m();
    const tPtr = wasm.client_registration(ctx);
    assert(tPtr !== 0 && tPtr !== null, 'registration returns non-null');

    const t = new BigUint64Array(memory.buffer, tPtr, M);
    assert(t.length === M, `t_poly has ${M} elements`);
    assert(t[0] < 8380417n, `t[0] = ${t[0]} is in valid range [0, Q)`);

    wasm.free_u64(tPtr, M);
    wasm.client_deinit(ctx);
}

function test_06_commitment() {
    console.log('\nTest 6: client_commit produces M-length commitment');
    const ctx = wasm.client_init();
    const M = wasm.get_m();

    const wPtr = wasm.client_commit(ctx, 1);
    assert(wPtr !== 0 && wPtr !== null, 'commitment returns non-null');

    const w = new BigUint64Array(memory.buffer, wPtr, M);
    assert(w.length === M, `w_poly has ${M} elements`);
    assert(w[0] < 8380417n, `w[0] = ${w[0]} is in valid range [0, Q)`);

    wasm.free_u64(wPtr, M);
    wasm.client_deinit(ctx);
}

function test_07_full_protocol() {
    console.log('\nTest 7: Full protocol loop completes');
    const ctx = wasm.client_init();
    const N = wasm.get_n();
    const M = wasm.get_m();
    const MAX = 100;

    // Registration
    const tPtr = wasm.client_registration(ctx);
    assert(tPtr !== 0, 'registration succeeded');

    let success = false;
    let attempts = 0;

    while (!success && attempts < MAX) {
        attempts++;

        // Commitment
        const wPtr = wasm.client_commit(ctx, attempts);
        if (!wPtr) { break; }

        // Simple challenge: just use attempt parity
        const challenge = BigInt(attempts % 2);

        // Response
        const zPtr = wasm.client_response(ctx, challenge);

        if (zPtr) {
            const z = new BigUint64Array(memory.buffer, zPtr, N);
            assert(z[0] < 8380417n, `z[0] = ${z[0]} is in valid range`);
            success = true;
            wasm.free_u64(zPtr, N);
        }

        wasm.free_u64(wPtr, M);
    }

    assert(success, `Protocol completed in ${attempts} attempt(s)`);
    assert(attempts < MAX, `Finished within ${MAX} attempt limit`);

    wasm.free_u64(tPtr, M);
    wasm.client_deinit(ctx);
}

function test_08_deterministic() {
    console.log('\nTest 8: Deterministic registration');
    const N = wasm.get_n();
    const M = wasm.get_m();

    // Two separate init/registration cycles should produce the same t_poly
    // (because seeds are deterministic in the current implementation)
    const ctx1 = wasm.client_init();
    const t1Ptr = wasm.client_registration(ctx1);
    const t1 = Array.from(new BigUint64Array(memory.buffer, t1Ptr, M));

    const ctx2 = wasm.client_init();
    const t2Ptr = wasm.client_registration(ctx2);
    const t2 = Array.from(new BigUint64Array(memory.buffer, t2Ptr, M));

    let same = true;
    for (let i = 0; i < M; i++) {
        if (t1[i] !== t2[i]) { same = false; break; }
    }
    assert(same, 'Same seeds produce identical registration output');

    wasm.free_u64(t1Ptr, M);
    wasm.free_u64(t2Ptr, M);
    wasm.client_deinit(ctx1);
    wasm.client_deinit(ctx2);
}

// ─── Runner ──────────────────────────────────────────────────────

async function main() {
    console.log('═══════════════════════════════════════');
    console.log('  Zerkerus WASM Test Suite');
    console.log('═══════════════════════════════════════');

    try {
        await loadWasm();
        console.log('WASM loaded successfully.');
    } catch (e) {
        console.error('Failed to load WASM:', e.message);
        process.exit(1);
    }

    test_01_exports();
    test_02_dimensions();
    test_03_alloc_dealloc();
    test_04_client_init_deinit();
    test_05_registration();
    test_06_commitment();
    test_07_full_protocol();
    test_08_deterministic();

    console.log('\n═══════════════════════════════════════');
    console.log(`  Results: ${passed} passed, ${failed} failed`);
    console.log('═══════════════════════════════════════');

    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
