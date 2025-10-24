use chrono::Local;
use csv::Writer;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// 实验日志记录器
pub struct Logger {
    run_dir: PathBuf,
    csv: Writer<File>,
    headers: Vec<String>, // 保存列名顺序（第一列固定为 "iter"）
}

impl Logger {
    /// 创建一个新的 Logger。
    /// base_dir: 根目录（例如 "./experiments"）
    /// exp_name: 实验名称（用于一级目录）
    /// purpose:  实验目的，会写入 purpose.txt
    /// returns:  Logger 实例，以及本次运行的目录路径
    pub fn new<P: AsRef<Path>>(
        base_dir: P,
        exp_name: &str,
        purpose: &str,
        extra_notes: Option<&str>, // 可选，记录更多信息（例如命令行参数）
    ) -> std::io::Result<Self> {
        // 目录： ./experiments/<exp_name>/<YYYY-MM-DD>/
        let date = Local::now().format("%Y-%m-%d").to_string();
        let run_dir = base_dir.as_ref().join(exp_name).join(date);
        fs::create_dir_all(&run_dir)?;

        // 写 purpose.txt
        {
            let mut f = File::create(run_dir.join("purpose.txt"))?;
            writeln!(f, "Purpose: {}", purpose)?;
            if let Some(notes) = extra_notes {
                writeln!(f, "\nNotes: {}", notes)?;
            }
            // 也顺便记一下时间戳
            writeln!(f, "\nCreated at: {}", Local::now().to_rfc3339())?;
        }

        // 打开 CSV（不存在则创建）
        let csv_path = run_dir.join("metrics.csv");
        let file = File::create(&csv_path)?;
        let csv = Writer::from_writer(file);

        Ok(Self {
            run_dir,
            csv,
            headers: Vec::new(),
        })
    }

    /// 记录一次迭代的指标。
    /// `metrics` 使用 BTreeMap 以保证列的稳定顺序（键名按字典序）。
    /// 首次调用会写表头：iter,<keys...>
    pub fn log_step(
        &mut self,
        iter: usize,
        metrics: &BTreeMap<String, f64>,
    ) -> std::io::Result<()> {
        // 初始化表头
        if self.headers.is_empty() {
            self.headers.push("iter".to_string());
            self.headers.extend(metrics.keys().cloned());

            // 写表头
            self.csv.write_record(&self.headers)?;
        } else {
            // 校验列一致性（除了 iter）
            let expected: Vec<&str> = self.headers.iter().skip(1).map(|s| s.as_str()).collect();
            let incoming: Vec<&str> = metrics.keys().map(|s| s.as_str()).collect();
            if expected != incoming {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "metrics keys changed.\nexpected: {:?}\nincoming: {:?}",
                        expected, incoming
                    ),
                ));
            }
        }

        // 组装一行：iter + 指标值（按 headers 的顺序）
        let mut row: Vec<String> = Vec::with_capacity(self.headers.len());
        row.push(iter.to_string());
        for k in self.headers.iter().skip(1) {
            // 安全：上面已保证存在
            let v = metrics.get(k).copied().unwrap_or(f64::NAN);
            row.push(v.to_string());
        }

        self.csv.write_record(&row)?;
        self.csv.flush()?; // 需要实时落盘就留着；若追求性能可以去掉，改为在 drop 时 flush
        Ok(())
    }

    /// 返回本次运行目录路径： ./experiments/<exp_name>/<YYYY-MM-DD>/
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }
}

#[test]
fn test_logger() {
    // 假设命令行：cargo run -- <exp_name> <purpose...>
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let exp_name = args.get(0).cloned().unwrap_or_else(|| "default_exp".into());
    let purpose = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "no purpose provided".into()
    };
    let notes = format!("Command line: {:?}", std::env::args().collect::<Vec<_>>());
    let mut logger = Logger::new("./experiments", &exp_name, &purpose, Some(&notes)).unwrap();

    for iter in 0..5 {
        // 你的训练/算法逻辑……
        // 拼指标
        let mut m = BTreeMap::new();
        m.insert("loss".to_string(), 1.0 / (iter as f64 + 1.0));
        m.insert("acc".to_string(), 0.2 * iter as f64);

        logger.log_step(iter, &m).unwrap();
    }

    println!("Logs at: {}", logger.run_dir().display());

    // cfr::cfr("kuhn_poker");
}
