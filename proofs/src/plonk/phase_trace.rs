//! Per-phase wall-clock timing of the prover (feature `trace-phases`).
//!
//! `create_proof` calls [`begin`] on entry, [`mark`] at the end of each phase
//! and [`end`] once the proof is written; with the feature on, [`end`] prints
//! one line per proof to stderr:
//!
//! ```text
//! [PHASES] k=15 total=1890.2ms instances=0.4ms advice_synthesize=180.2ms ...
//! ```
//!
//! Each phase is the time since the previous mark, so the phases add up to the
//! total; a phase marked more than once (inside the per-circuit loops) is the
//! sum of its intervals. `advice_*` split witness synthesis, the batch
//! inversion and the commitments; `h_*` split the quotient computation into
//! the extended-domain cosets and the gate, permutation, lookup and trash
//! evaluations. GPU work inside a phase (MSM, NTT) is included in that phase: the
//! split shows where a proof's time goes, CPU and GPU together. The timer is
//! thread-local, so proofs on parallel threads (the proof server's workers)
//! each report their own phases. Without the feature every function is an
//! empty `#[inline(always)]` body.

#[cfg(feature = "trace-phases")]
mod imp {
    use std::{
        cell::RefCell,
        time::{Duration, Instant},
    };

    struct Timer {
        start: Instant,
        last: Instant,
        phases: Vec<(&'static str, Duration)>,
    }

    thread_local! {
        static TIMER: RefCell<Option<Timer>> = const { RefCell::new(None) };
    }

    pub(crate) fn begin() {
        let now = Instant::now();
        TIMER.with(|t| {
            *t.borrow_mut() = Some(Timer {
                start: now,
                last: now,
                phases: Vec::with_capacity(16),
            })
        });
    }

    pub(crate) fn mark(phase: &'static str) {
        let now = Instant::now();
        TIMER.with(|t| {
            if let Some(timer) = t.borrow_mut().as_mut() {
                let elapsed = now - timer.last;
                // A phase marked more than once (once per circuit or per proof
                // phase) accumulates under its first position.
                match timer.phases.iter_mut().find(|(name, _)| *name == phase) {
                    Some((_, total)) => *total += elapsed,
                    None => timer.phases.push((phase, elapsed)),
                }
                timer.last = now;
            }
        });
    }

    pub(crate) fn end(k: u32) {
        TIMER.with(|t| {
            if let Some(timer) = t.borrow_mut().take() {
                let ms = |d: Duration| d.as_secs_f64() * 1e3;
                let phases: Vec<String> = timer
                    .phases
                    .iter()
                    .map(|(name, d)| format!("{name}={:.1}ms", ms(*d)))
                    .collect();
                eprintln!(
                    "[PHASES] k={k} total={:.1}ms {}",
                    ms(timer.last - timer.start),
                    phases.join(" ")
                );
            }
        });
    }
}

#[cfg(not(feature = "trace-phases"))]
mod imp {
    #[inline(always)]
    pub(crate) fn begin() {}

    #[inline(always)]
    pub(crate) fn mark(_phase: &'static str) {}

    #[inline(always)]
    pub(crate) fn end(_k: u32) {}
}

pub(crate) use imp::{begin, end, mark};
