//! Example: Eager GPU Backend Initialization
//!
//! Demonstrates how to initialize the GPU backend at application startup
//! to avoid first-call latency during proof generation.
//!
//! Run with:
//! ```bash
//! cargo run --example gpu_init --features gpu --release
//! ```

use midnight_proofs::init_gpu_backend;

fn main() {
    println!("=== GPU Backend Eager Initialization Example ===\n");

    // Initialize GPU backend BEFORE starting the proof server
    // This eliminates the ~500-1000ms initialization latency on first proof
    match init_gpu_backend() {
        Some(duration) => {
            println!("✅ GPU backend initialized and warmed up in {:?}", duration);
            println!("   CUDA functions are now registered and ready");
            println!("   First proof request will NOT pay initialization cost\n");
        }
        None => {
            println!("⚠️  GPU backend not available (running in CPU mode)");
            println!("   This could mean:");
            println!("   - No CUDA device found");
            println!("   - ICICLE backend not installed");
            println!("   - MIDNIGHT_DEVICE=cpu environment variable set\n");
        }
    }

    println!("Application ready to process proof requests!");
    
    // In a real application, you would now:
    // - Start your HTTP/gRPC server
    // - Accept proof generation requests
    // - Generate proofs with zero GPU initialization latency
}
