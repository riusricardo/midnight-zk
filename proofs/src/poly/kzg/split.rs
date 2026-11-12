//! Sharing a batch of commitments between the GPU and the CPU.
//!
//! A batch commit ([`super::params::ParamsKZG::commit_lagrange_batch`]) runs
//! one MSM per polynomial. The GPU runs its MSMs one after another while the
//! CPU waits; this module lets the CPU take a share. Both devices compute the
//! same group element for an MSM, so the commitments (and the proof) do not
//! depend on which device computed which.
//!
//! The share is planned from measured costs. MSMs are grouped into classes by
//! what the CPU's cost depends on beyond the size: how many scalars are zero
//! and whether the scalars are small (BLST's Pippenger skips zero digits). For
//! each class the process keeps a running estimate of the time one MSM takes on
//! each device, updated after every batch, and the plan moves to the CPU the
//! MSMs it is relatively best at, as long as that shortens the batch. A class
//! the CPU has not run yet is measured by giving the CPU one of its MSMs, in a
//! batch of at least [`PROBE_MIN_BATCH`] where the CPU has nothing else to do.

use std::{collections::HashMap, sync::Mutex, time::Duration};

use ff::Field;
use midnight_curves::Fq;

/// Smallest batch in which an unmeasured class is probed on the CPU.
pub(crate) const PROBE_MIN_BATCH: usize = 4;

/// Weight of a new measurement in the running estimates.
const WEIGHT: f64 = 0.3;

/// What an MSM's relative CPU cost depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MsmClass {
    log_size: u32,
    /// Share of zero scalars, in quarters (0..=3).
    zero_quarter: u8,
    /// Most sampled non-zero scalars are below 2^128.
    small: bool,
}

impl MsmClass {
    /// The class of an MSM over `scalars`.
    pub(crate) fn of(scalars: &[Fq]) -> Self {
        let n = scalars.len();
        let zeros = scalars.iter().filter(|s| s.is_zero_vartime()).count();
        let zero_quarter = ((zeros * 4) / n.max(1)).min(3) as u8;
        let step = (n / 64).max(1);
        let (mut sampled, mut small) = (0usize, 0usize);
        for s in scalars.iter().step_by(step).filter(|s| !s.is_zero_vartime()).take(64) {
            sampled += 1;
            if s.to_bytes_le()[16..].iter().all(|b| *b == 0) {
                small += 1;
            }
        }
        MsmClass {
            log_size: n.next_power_of_two().trailing_zeros(),
            zero_quarter,
            small: sampled > 0 && 2 * small > sampled,
        }
    }
}

/// Seconds one MSM of a class takes on each device, as measured so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Estimate {
    pub(crate) cpu: Option<f64>,
    pub(crate) gpu: Option<f64>,
}

static ESTIMATES: Mutex<Option<HashMap<MsmClass, Estimate>>> = Mutex::new(None);

fn estimates_of(classes: &[MsmClass]) -> Vec<Estimate> {
    let table = ESTIMATES.lock().unwrap_or_else(|e| e.into_inner());
    classes
        .iter()
        .map(|c| table.as_ref().and_then(|t| t.get(c)).copied().unwrap_or_default())
        .collect()
}

/// Which MSMs of a batch of these classes the CPU computes (`true`).
pub(crate) fn plan(classes: &[MsmClass]) -> Vec<bool> {
    plan_with(&estimates_of(classes))
}

/// [`plan`] for given per-MSM estimates.
pub(crate) fn plan_with(estimates: &[Estimate]) -> Vec<bool> {
    let n = estimates.len();
    let mut on_cpu = vec![false; n];
    if n < 2 {
        return on_cpu;
    }
    // GPU time of each MSM; an unmeasured one counts as the batch's mean.
    let known: Vec<f64> = estimates.iter().filter_map(|e| e.gpu).collect();
    let mean = if known.is_empty() { None } else { Some(known.iter().sum::<f64>() / known.len() as f64) };
    if let Some(mean) = mean {
        let gpu: Vec<f64> = estimates.iter().map(|e| e.gpu.unwrap_or(mean)).collect();
        // Move MSMs to the CPU, lowest CPU-to-GPU ratio first, while the time
        // until both devices are done keeps dropping.
        let mut order: Vec<(f64, usize)> =
            (0..n).filter_map(|i| estimates[i].cpu.map(|c| (c / gpu[i].max(f64::MIN_POSITIVE), i))).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut cpu_total, mut gpu_total) = (0.0f64, gpu.iter().sum::<f64>());
        for (_, i) in order {
            let cpu_cost = estimates[i].cpu.expect("ordered MSMs have a CPU estimate");
            let (c, g) = (cpu_total + cpu_cost, gpu_total - gpu[i]);
            if c.max(g) >= cpu_total.max(gpu_total) {
                break;
            }
            on_cpu[i] = true;
            (cpu_total, gpu_total) = (c, g);
        }
    }
    // Probe an unmeasured class when the CPU would otherwise be idle.
    if n >= PROBE_MIN_BATCH && !on_cpu.contains(&true) {
        if let Some(i) = (0..n).rev().find(|&i| estimates[i].cpu.is_none()) {
            on_cpu[i] = true;
        }
    }
    on_cpu
}

fn blend(old: Option<f64>, new: f64) -> f64 {
    old.map_or(new, |o| o + WEIGHT * (new - o))
}

/// Records a batch: the time of each CPU MSM, and the GPU's MSMs with the time
/// they took together (none when the GPU took no MSM or one of them failed).
pub(crate) fn record(cpu: &[(MsmClass, Duration)], gpu: Option<(&[MsmClass], Duration)>) {
    let mut table = ESTIMATES.lock().unwrap_or_else(|e| e.into_inner());
    let table = table.get_or_insert_with(HashMap::new);
    for (class, t) in cpu {
        let e = table.entry(*class).or_default();
        e.cpu = Some(blend(e.cpu, t.as_secs_f64()));
    }
    if let Some((classes, total)) = gpu {
        if !classes.is_empty() {
            let each = total.as_secs_f64() / classes.len() as f64;
            for class in classes {
                let e = table.entry(*class).or_default();
                e.gpu = Some(blend(e.gpu, each));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn est(cpu: Option<f64>, gpu: Option<f64>) -> Estimate {
        Estimate { cpu, gpu }
    }

    fn cpu_count(plan: &[bool]) -> usize {
        plan.iter().filter(|c| **c).count()
    }

    #[test]
    fn devices_of_equal_speed_share_a_batch() {
        let plan = plan_with(&vec![est(Some(1.0), Some(1.0)); 10]);
        assert_eq!(cpu_count(&plan), 5);
    }

    #[test]
    fn a_slower_cpu_takes_a_smaller_share() {
        // CPU three times slower: the CPU's share is about a quarter.
        let plan = plan_with(&vec![est(Some(3.0), Some(1.0)); 12]);
        assert_eq!(cpu_count(&plan), 3);
        // Much slower: the CPU takes nothing (it would finish last).
        let plan = plan_with(&vec![est(Some(50.0), Some(1.0)); 12]);
        assert_eq!(cpu_count(&plan), 0);
    }

    #[test]
    fn a_faster_cpu_takes_most_of_the_batch() {
        let plan = plan_with(&vec![est(Some(0.5), Some(1.0)); 9]);
        assert_eq!(cpu_count(&plan), 6);
    }

    #[test]
    fn the_cpu_takes_the_msms_it_is_best_at() {
        let mut estimates = vec![est(Some(4.0), Some(1.0)); 6];
        estimates[2] = est(Some(1.0), Some(1.0));
        estimates[4] = est(Some(1.0), Some(1.0));
        let plan = plan_with(&estimates);
        assert!(plan[2] && plan[4]);
        assert_eq!(cpu_count(&plan), 2);
    }

    #[test]
    fn unmeasured_classes_are_probed_once_in_large_enough_batches() {
        let fresh = vec![Estimate::default(); PROBE_MIN_BATCH];
        assert_eq!(cpu_count(&plan_with(&fresh)), 1);
        let small = vec![Estimate::default(); PROBE_MIN_BATCH - 1];
        assert_eq!(cpu_count(&plan_with(&small)), 0);
        // GPU measured, CPU not: still one probe.
        let gpu_only = vec![est(None, Some(1.0)); 8];
        assert_eq!(cpu_count(&plan_with(&gpu_only)), 1);
        // No probe while the CPU already has work.
        let mut mixed = vec![est(Some(1.0), Some(1.0)); 8];
        mixed[7] = est(None, Some(1.0));
        let plan = plan_with(&mixed);
        assert!(!plan[7]);
    }

    #[test]
    fn single_msms_stay_on_the_gpu() {
        assert_eq!(plan_with(&[est(Some(0.1), Some(1.0))]), vec![false]);
        assert!(plan_with(&[]).is_empty());
    }

    #[test]
    fn classes_separate_sparse_small_and_dense_scalars() {
        use rand_core::OsRng;
        let n = 1 << 10;
        let dense: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
        let sparse_small: Vec<Fq> =
            (0..n).map(|i| if i % 4 == 0 { Fq::from(i as u64) } else { Fq::ZERO }).collect();
        let d = MsmClass::of(&dense);
        let s = MsmClass::of(&sparse_small);
        assert_eq!((d.log_size, d.zero_quarter, d.small), (10, 0, false));
        assert_eq!((s.log_size, s.zero_quarter, s.small), (10, 3, true));
        assert_ne!(d, s);
    }
}
