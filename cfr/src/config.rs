pub struct Config {
    pub game_name: String,
    pub algo_name: String,
    pub subgame_prune: (bool, bool, f64, f64),
    current_iteration: usize,
    pub count_leaves: bool, // whether take leaves nodes into account when calculate pruning prop.
    dynamic_thre: bool,
    pub dir: String,
}

impl Config {
    pub fn step(&mut self) -> f64 {
        let (b, k) = self.dynamic_thershold();
        self.current_iteration += 1;
        if self.dynamic_thre {
            self.subgame_prune.2 =
                0.1 * (10.0 as f64).powf(b) * (self.current_iteration as f64).powf(-k);
            self.subgame_prune.3 = (10.0 as f64).powf(b) * (self.current_iteration as f64).powf(-k);
        }
        self.subgame_prune.2
    }
    pub fn get_current_iteration(&self) -> usize {
        self.current_iteration
    }
    fn dynamic_thershold(&self) -> (f64, f64) {
        /* dynamic check-free thresholds */
        let (b, k): (f64, f64);
        if self.game_name == "liars_dice".to_string() {
            if self.algo_name == "CFR+".to_string() {
                k = 0.9146;
                b = 0.387;
            } else {
                k = 0.6586;
                b = 0.2632;
            }
        } else if self.game_name == "leduc_poker".to_string() {
            if self.algo_name == "CFR+".to_string() {
                k = 0.9124;
                b = 1.0;
            } else {
                k = 0.6;
                b = 1.5;
            }
        } else if self.game_name == "my_leduc5".to_string() {
            if self.algo_name == "CFR+".to_string() {
                k = 1.5;
                b = 0.0;
            } else {
                k = 1.5;
                b = 0.0;
            }
        } else if self.game_name == "tiny_bridge_2p".to_string() {
            k = 1.5;
            b = 1.7;
        } else if self.game_name == "tiny_hanabi".to_string() {
            k = 1.0;
            b = 1.0;
        } else {
            k = 1.0;
            b = 1.0;
        }
        (b, k)
    }

    pub fn new(
        game_name: String,
        algo_name: String,
        count_leaves: bool,
        dynamic_thre: bool,
        dir: String,
    ) -> Self {
        Config {
            game_name: game_name,
            algo_name: algo_name,
            subgame_prune: (false, false, 0.0, 0.0),
            current_iteration: 0,
            count_leaves,
            dynamic_thre,
            dir,
        }
    }
    pub fn new_subgame_prune_wo_check_free(
        game_name: String,
        algo_name: String,
        count_leaves: bool,
        dynamic_thre: bool,
        dir: String,
    ) -> Self {
        Config {
            game_name: game_name,
            algo_name: algo_name,
            subgame_prune: (true, false, 0.0, 0.0),
            current_iteration: 0,
            count_leaves,
            dynamic_thre,
            dir,
        }
    }
    pub fn new_subgame_prune_with_check_free(
        game_name: String,
        algo_name: String,
        count_leaves: bool,
        dynamic_thre: bool,
        dir: String,
    ) -> Self {
        Config {
            game_name: game_name,
            algo_name: algo_name,
            subgame_prune: (true, true, 0.0, 0.0),
            current_iteration: 0,
            count_leaves,
            dynamic_thre,
            dir,
        }
    }
    pub fn to_string(&self) -> String {
        format!(
            "game_env{}-algo-{}-subgame_pruning_cfg-{:.8?}-count_leaves-{}-dynamic_thresholds-{}",
            self.game_name,
            self.algo_name,
            self.subgame_prune,
            self.count_leaves,
            self.dynamic_thre,
        )
    }
}
