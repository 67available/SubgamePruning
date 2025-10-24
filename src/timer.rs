use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Duration;
use std::time::Instant;

/// 在线均值 (Welford)
#[derive(Default, Debug)]
pub struct OnlineMean {
    n: u64,
    mean: f64,
}
impl OnlineMean {
    pub fn add(&mut self, x: f64) {
        self.n += 1;
        self.mean += (x - self.mean) / self.n as f64;
    }
    pub fn count(&self) -> u64 {
        self.n
    }
    pub fn mean(&self) -> f64 {
        self.mean
    }
}

/// 支持箱型图的计时器：
/// - add(duration, iterations)：提交该次迭代耗时（秒）与包含的迭代数（通常为1）
/// - 内部维护在线均值
/// - 以 reservoir 抽样保存样本（单位：毫秒），可导出 CSV 以绘图
pub struct CfrTimerWithSamples {
    avg: OnlineMean,
    // reservoir sampling
    samples_ms: Vec<f64>,
    cap: usize,
    seen: u64,
    rng: StdRng,
}

impl CfrTimerWithSamples {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            avg: OnlineMean::default(),
            samples_ms: Vec::with_capacity(capacity),
            cap: capacity,
            seen: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 记录一次测量结果
    /// `duration`: 该次测得的总耗时（建议为单次迭代的耗时）
    /// `iterations`: 本次结果包含的迭代数（通常=1，用于摊销平均值；样本按“每次迭代”入库）
    pub fn add(&mut self, duration: Duration, iterations: u64) {
        if iterations == 0 {
            return;
        }

        let per_iter_secs = duration.as_secs_f64() / iterations as f64;
        let per_iter_ms = per_iter_secs * 1000.0;

        // 在线均值（按“每次迭代”的耗时）
        self.avg.add(per_iter_ms);

        // reservoir：保存“每次迭代”样本（毫秒）
        self.seen += 1;
        if self.samples_ms.len() < self.cap {
            self.samples_ms.push(per_iter_ms);
        } else {
            let j = self.rng.gen_range(0..self.seen);
            if (j as usize) < self.cap {
                self.samples_ms[j as usize] = per_iter_ms;
            }
        }
    }

    /// 平均每次迭代耗时（毫秒）
    pub fn average_ms(&self) -> f64 {
        self.avg.mean()
    }

    /// 返回样本（拷贝一份，安全起见）
    pub fn samples(&self) -> Vec<f64> {
        let mut s = self.samples_ms.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    }

    /// 近似分位数（基于样本）
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.samples_ms.is_empty() {
            return None;
        }
        let s = self.samples();
        let t = p.clamp(0.0, 1.0) * (s.len() as f64 - 1.0);
        let lo = t.floor() as usize;
        let hi = t.ceil() as usize;
        if lo == hi {
            Some(s[lo])
        } else {
            let w = t - lo as f64;
            Some(s[lo] * (1.0 - w) + s[hi] * w)
        }
    }

    /// 导出 CSV（单列：ms）
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        writeln!(w, "iter_ms")?;
        for &x in &self.samples_ms {
            writeln!(w, "{x}")?;
        }
        w.flush()
    }
}

#[test]
fn test_timer() -> std::io::Result<()> {
    let mut timer = CfrTimerWithSamples::new(50_000, 42); // 最多保留5万样本

    for _ in 0..10_000 {
        let t0 = Instant::now();
        // 你的CFR单次迭代
        // cfr.iterate_once(...);
        std::thread::sleep(Duration::from_micros(800)); // 示例
        let dt = t0.elapsed();

        timer.add(dt, 1);
        // 如果你某次测到的是“批量K次迭代的总时长”，传 iterations=K 即可摊销
    }

    println!("平均每次迭代：{:.3} ms", timer.average_ms());
    println!(
        "P50≈{:.3} ms, P90≈{:.3} ms, P99≈{:.3} ms",
        timer.percentile(0.5).unwrap_or(f64::NAN),
        timer.percentile(0.9).unwrap_or(f64::NAN),
        timer.percentile(0.99).unwrap_or(f64::NAN),
    );

    // 导出CSV，后续画箱型图
    timer.export_csv("cfr_iter_samples.csv")?;
    Ok(())
}
