# midnight_proofs

Implementation of Plonk proof system with KZG commitments. This repo initially started 
as a fork of [`halo2`](https://github.com/privacy-scaling-explorations/halo2) v0.3.0 – 
by the Privacy Scaling Explorations (PSE) team, itself originally derived from the 
[Zcash Sapling proving system](https://github.com/zcash/halo2). 
### Summary of Changes

- **Generic proof system interface**  
  The proof system is now built on top of a generic trait, `PolynomialCommitmentScheme`, with a simpler
  and more generic interface.
  At the moment, we provide an implementation using KZG commitments with the original Halo2  
  [multi-open argument](https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html).

- **Simplified transcript interface**  
  The `Transcript` API has been modified for easier use in and off-circuit.

- **Additive selector support**  
  Enabled additive selectors (via the `trash` argument), allowing selectors that do not increase  
  the degree of the proof system.

- **Committed instances**  
  Added support for committed instances behind the `committed-instances` feature flag.  
  See Section 4.2 of the [aPLONK paper](https://eprint.iacr.org/2022/1352.pdf) for details.

- **Truncated challenges for recursion**  
  To enable efficient recursion, Fiat–Shamir challenges can now be truncated to 128 bits  
  (via the `truncate-challenges` feature). This halves the size of scalar multiplications  
  in-circuit, resulting in considerable circuit size gains for in-circuit proof verificaiton.
## Minimum Supported Rust Version

Requires Rust **1.85.0** or higher.

Minimum supported Rust version can be changed in the future, but it will be done with a
minor version bump.

## GPU Acceleration

The `gpu` feature enables CUDA-based GPU acceleration for Multi-Scalar Multiplication (MSM) 
operations using the [ICICLE](https://github.com/ingonyama-zk/icicle) library. This provides significant speedups for large circuits.

### Requirements

- NVIDIA GPU with CUDA support
- ICICLE v4.0.0 backend installed (default path: `/opt/icicle/lib/backend`)
- CUDA toolkit and drivers

### Configuration

The GPU backend is configured via environment variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `MIDNIGHT_DEVICE` | `cpu`, `gpu`, `auto` | `auto` | Device selection mode |
| `MIDNIGHT_GPU_MIN_K` | Integer (e.g., `14`) | `14` | Minimum circuit size (K) for GPU usage |
| `ICICLE_BACKEND_INSTALL_DIR` | Path | `/opt/icicle/lib/backend` | ICICLE backend location |

### Device Modes

- **auto** (default): Uses GPU for circuits with K >= `MIDNIGHT_GPU_MIN_K`, CPU otherwise
- **gpu**: Forces GPU for all MSM operations regardless of size
- **cpu**: Forces CPU-only execution (uses BLST library)

### Building with GPU Support

```bash
cargo build --release --features gpu
```

### Usage Example

```bash
# Auto mode: GPU for large circuits (k=14 default), CPU for smaller ones
export MIDNIGHT_DEVICE=auto
export MIDNIGHT_GPU_MIN_K=16

# Force CPU-only execution
export MIDNIGHT_DEVICE=cpu

# Force GPU for all operations
export MIDNIGHT_DEVICE=gpu
```

### Architecture

The GPU module (`src/gpu/`) contains:

- `backend.rs` - ICICLE backend initialization with singleton pattern
- `config.rs` - Configuration and environment variable parsing
- `msm.rs` - GPU-accelerated MSM executor
- `batch.rs` - Batch MSM operations
- `types.rs` - GPU-specific type definitions

The backend initializes once per process and caches GPU bases to minimize memory transfer overhead.

## Controlling parallelism

`midnight_proofs` currently uses [rayon](https://github.com/rayon-rs/rayon) for parallel
computation. The `RAYON_NUM_THREADS` environment variable can be used to set the number of
threads.

When compiling to WASM-targets, notice that since version `1.7`, `rayon` will fallback automatically (with no need to handle features) to require `getrandom` in order to be able to work. For more info related to WASM-compilation.

See: [Rayon: Usage with WebAssembly](https://github.com/rayon-rs/rayon#usage-with-webassembly) for more 

## License

See root directory for Licensing. We have copied the license files of the original [Zcash Sapling proving system](https://github.com/zcash/halo2).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
