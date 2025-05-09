use plotters::prelude::*;
use std::collections::HashMap;

fn plot_loss_curves(
    data: HashMap<String, Vec<f64>>,
    filename: &str,
    plot_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure that the data contains "iteration" as the x-axis
    if !data.contains_key("iteration") {
        return Err("Missing key: 'iteration'".into());
    }

    let iteration = &data["iteration"];

    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = *iteration
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let x_max = *iteration
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;
    for (key, values) in &data {
        if key != "iteration" {
            y_min = y_min.min(
                *values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            );
            y_max = y_max.max(
                *values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            );
        }
    }

    if y_min <= 0.0 {
        y_min = 1e-10;
    }

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{}", plot_name), ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((x_min..x_max).log_scale(), (y_min..y_max).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Iteration (Log Scale)")
        .y_desc("Loss (Log Scale)")
        .draw()?;

    let colors = vec![RED, BLUE, GREEN, MAGENTA, CYAN, BLACK];
    let mut color_iter = colors.iter().cycle(); // 颜色循环使用

    for (key, values) in &data {
        if key == "iteration" {
            continue;
        }
        let color = color_iter.next().unwrap().clone();
        chart
            .draw_series(LineSeries::new(
                iteration.iter().zip(values.iter()).map(|(&x, &y)| (x, y)),
                color,
            ))?
            .label(key.clone())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = HashMap::new();
    let iteration: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let loss1: Vec<f64> = iteration.iter().map(|&x| 1.0 / x).collect();
    let loss2: Vec<f64> = iteration.iter().map(|&x| 1.0 / (x * x)).collect();
    let loss3: Vec<f64> = iteration.iter().map(|&x| 1.0 / (x.sqrt())).collect();

    data.insert("iteration".to_string(), iteration);
    data.insert("loss1".to_string(), loss1);
    data.insert("loss2".to_string(), loss2);
    data.insert("loss3".to_string(), loss3);

    plot_loss_curves(data, "results/test/multi_loss_plot.png", "Loss_Curves")
}
