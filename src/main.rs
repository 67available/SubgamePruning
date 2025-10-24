mod br;
mod cfr;
mod dfs;
mod load_tree;
mod logger;
mod subgame_prune_check_free_only;
mod subgame_prune_check_free_only_c;
mod subgame_prune_general;
mod subgame_prune_general_c;
mod timer;
use clap::{Parser, ValueEnum};

#[derive(Clone, Debug, ValueEnum)]
#[value(rename_all = "kebab-case")] // 让命令行写法用 kebab-case：cfr, cfr-plus, dcfr
enum Algo {
    Cfr,
    SP,
    SPGeneral,
    SPC,
    SPGeneralC,
}

#[derive(Parser, Debug)]
#[command(name = "runner", about = "Run CFR experiments")]
struct Args {
    /// 实验名（原来的第一个位置参数）
    exp_name: Option<String>,

    /// 目的/备注（原来的后续所有位置参数，自动用空格拼接）
    purpose: Vec<String>,

    /// 是否使用 CFR+
    #[arg(long)] // 字段若没有 #[arg(long)]/#[arg(short)] 等标注，就被当作位置参数。
    cfr_plus: bool,

    /// 算法（枚举）
    #[arg(long, value_enum, default_value_t = Algo::Cfr)]
    algo: Algo,

    /// 算法名称，例如：cfr、cfr_plus、dcfr...
    // #[arg(long, default_value = "cfr")]
    // algo: String,

    /// 训练轮数
    #[arg(long, default_value_t = 1000usize)]
    epoch: usize,

    /// env
    #[arg(long, default_value = "kuhn_poker")]
    env: String,
}

/// cfg
#[derive(Debug)]
struct Config {
    cfr_plus: bool,
    algo: Algo,
    epoch: usize,
    force_compensate: bool,  // 用于dbg或者计算exploit，强制进行compensate
    record_step: Vec<usize>, // 需要输出信息的step
    delta: [f64; 2],
}

impl ToString for Config {
    fn to_string(&self) -> String {
        format!(
            "algo:{:?}\ncfr+:{}\ndelta:{:?}\nepoch:{}\n",
            self.algo, self.cfr_plus, self.delta, self.epoch
        )
    }
}

const DELTA_RP: f64 = 1e-15;
const DELTA_REGRET: f64 = 1e-15;
const WARMUP: usize = 50; // 前50代用于热身
const JUMPTHEGUN: usize = 2;

fn record_step(epoch: usize) -> Vec<usize> {
    let mut record_step = vec![0];
    let tens = (2..10)
        .into_iter()
        .map(|x| 10_i32.pow(x) as usize)
        .collect::<Vec<_>>();
    let mut step = tens[0];
    let mut idx = 0;
    while step < epoch {
        record_step.push(step);
        step += tens[idx].min(10000);
        if step >= tens[idx + 1] {
            idx += 1;
        }
    }
    record_step.push(epoch - 1);
    record_step
}

fn main() {
    // 支持：cargo run -- <exp_name> <purpose...> --cfr-plus --algo cfr --epoch 500
    // cargo run "kuhn/vanilla_cfr" "测试代码输出一致性" --env kuhn_poker --epoch 1000
    // cargo run "kuhn/cfr_w_sp_check-free-only" "测试代码输出一致性" --env kuhn_poker --epoch 1000 --algo sp -- --release
    let args = Args::parse();

    let exp_name = args.exp_name.unwrap_or_else(|| "default_exp".into());
    let purpose = if args.purpose.is_empty() {
        "no purpose provided".to_string()
    } else {
        args.purpose.join(" ")
    };
    let notes = format!("Command line: {:?}", std::env::args().collect::<Vec<_>>());
    let cfg = Config {
        cfr_plus: args.cfr_plus,
        algo: args.algo,
        epoch: args.epoch,
        force_compensate: false,
        record_step: record_step(args.epoch),
        delta: [DELTA_RP, DELTA_REGRET],
    };

    let logger = logger::Logger::new(
        "./experiments_13700",
        &exp_name,
        &(purpose + "\ndir:" + &exp_name + "\nenv:" + &args.env + "\n" + &cfg.to_string()),
        Some(&notes),
    )
    .unwrap();
    match cfg.algo {
        Algo::Cfr => cfr::cfr(&args.env, logger, cfg),
        Algo::SP => subgame_prune_check_free_only::subgame_prune_cfr(&args.env, logger, cfg),
        Algo::SPGeneral => subgame_prune_general::subgame_prune_cfr(&args.env, logger, cfg),
        Algo::SPC => subgame_prune_check_free_only_c::subgame_prune_cfr(&args.env, logger, cfg),
        Algo::SPGeneralC => subgame_prune_general_c::subgame_prune_cfr(&args.env, logger, cfg),
    }
}

#[test]
fn test_subgame_prune() {
    // cargo run -- <exp_name> <purpose...>
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let exp_name = args.get(0).cloned().unwrap_or_else(|| "default_exp".into());
    let purpose = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "no purpose provided".into()
    };
    let notes = format!("Command line: {:?}", std::env::args().collect::<Vec<_>>());
    let cfg = Config {
        cfr_plus: false,
        algo: Algo::SP,
        epoch: 1000,
        force_compensate: false, // 在计算exploit需要把force_compensate置为true
        record_step: record_step(1000),
        delta: [DELTA_RP, DELTA_REGRET],
    };

    let logger = logger::Logger::new(
        "./experiments",
        &exp_name,
        &(purpose + "\n" + &cfg.to_string()),
        Some(&notes),
    )
    .unwrap();
    subgame_prune_check_free_only::subgame_prune_cfr("my_leduc5-2", logger, cfg);
}

#[test]
fn test_subgame_prune_general() {
    // cargo run -- <exp_name> <purpose...>
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let exp_name = args.get(0).cloned().unwrap_or_else(|| "default_exp".into());
    let purpose = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "no purpose provided".into()
    };
    let notes = format!("Command line: {:?}", std::env::args().collect::<Vec<_>>());
    let cfg = Config {
        cfr_plus: true,
        algo: Algo::SP,
        epoch: 1000,
        force_compensate: false, // 在计算exploit需要把force_compensate置为true
        record_step: record_step(1000),
        delta: [DELTA_RP, DELTA_REGRET],
    };

    let logger = logger::Logger::new(
        "./experiments",
        &exp_name,
        &(purpose + "\n" + &cfg.to_string()),
        Some(&notes),
    )
    .unwrap();
    subgame_prune_general::subgame_prune_cfr("kuhn_poker", logger, cfg);
}
