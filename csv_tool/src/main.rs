use csv::Writer;
use std::error::Error;
pub mod tool;
use tool::write_csv;

fn main() -> Result<(), Box<dyn Error>> {
    // demo
    let headers = vec!["Col1", "Col2", "Col3"]; // `Vec<&str>`
    let col1 = vec![1.1, 2.2, 3.3];
    let col2 = vec![4.4, 5.5, 6.6];
    let col3 = vec![7.7, 8.8, 9.9];

    let data = vec![col1, col2, col3];

    write_csv("results/test/output.csv", &headers, data)?;

    Ok(())
}
