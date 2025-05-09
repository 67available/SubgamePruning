# CFR Solver with Subgame Pruning
## CLI Usage
```
cargo run --bin cfr main
```
| Option                 | Description                             | Default            |
| ---------------------- | --------------------------------------- | ------------------ |
| `-a`, `--algo <Algo>`  | Algorithm to use: `CFR` or `CFR+`       | `CFR`              |
| `-g`, `--game <int>`   | Game environment ID (0 \~ 5)            | `0`                |
| `-i`, `--iter <int>`   | Number of iterations                    | `100`              |
| `-s`, `--sp`           | Enable Subgame Pruning                  | *(off by default)* |
| `-d`, `--dir <path>`   | Output directory                        | `./results`        |
| `-t`, `--thread <int>` | Number of threads (0 = auto) | `0`                |
| `-h`, `--help`         | Show help                               | -                  |

**Example**   
```
cargo run --bin cfr main -a CFR+ -g 1 -i 500 -s -d ./exp1 -t 4
```


## 📁 Output
Each run generates a .csv file under the specified output directory, with the following columns:
| Column                     | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| `Itera`                    | Current iteration number                               |
| `Exploitablity`            | Exploitability value (lower is better)                 |
| `Expected_return-0`        | Expected return for player 0 under average strategy    |
| `Expected_return-1`        | Expected return for player 1 under average strategy    |
| `BR-0`                     | Best response value for player 0 against average strategy |
| `BR-1`                     | Best response value for player 1 against average strategy |
| `PrunedProp-pubnode`       | Proportion of pruned public nodes                      |
| `CheckFreePRrop-pubnode`   | Proportion of public nodes not requiring pruning check |
| `PrunedProp-gamestate`     | Proportion of pruned game states                       |
| `CheckFreePRrop-gamestate` | Proportion of game states not requiring pruning check  |


