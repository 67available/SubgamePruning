use csv::Writer;
use std::{error::Error, fs};

pub fn write_csv<T: ToString>(
    filename: &str,
    headers: &[&str], // Change to `&[&str]` to support `Vec<&str>`
    columns: Vec<Vec<T>>,
) -> Result<(), Box<dyn Error>> {
    // Get the folder path where the file is located
    let path = std::path::Path::new(filename);
    if let Some(parent) = path.parent() {
        // If the folder doesn't exist, create it
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut wtr = Writer::from_path(filename)?;

    // Write the headers (`&[&str]` can be directly passed to `write_record`)
    wtr.write_record(headers)?;

    // Get the number of rows (assuming all columns have the same length)
    let row_count = columns.first().map_or(0, |col| col.len());

    // Transpose the data (convert columns to rows)
    for i in 0..row_count {
        let row: Vec<String> = columns.iter().map(|col| col[i].to_string()).collect();
        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}
