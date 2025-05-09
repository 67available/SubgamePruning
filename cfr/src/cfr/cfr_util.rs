use std::{collections::HashMap, vec};

pub fn transpose<T>(v: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    // 矩阵转置
    let leng = v.len();
    let wid = v[0].len();
    let mut res = vec![vec![]; wid];
    for i in 0..leng {
        for j in 0..wid {
            res[j].push(v[i][j].clone());
        }
    }
    return res;
}

pub fn check_nan(v: &Vec<f64>) {
    // Check for the presence of NaN, usually caused by division by zero
    v.iter().for_each(|x| {
        if x.is_nan() {
            println!("{:?}", v);
            panic!();
        }
    });
}

pub fn pointwise_multiple(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    // Perform element-wise multiplication of the two vectors and then sum the results
    assert_eq!(v1.len(), v2.len());
    let mut res = vec![];
    for i in 0..v1.len() {
        res.push(v1[i] * v2[i]);
    }
    res
}

pub fn multiple(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    // Dot product
    assert_eq!(v1.len(), v2.len());
    let mut res = 0.0;
    for i in 0..v1.len() {
        res += v1[i] * v2[i];
    }
    res
}

pub fn multiple_(v1: &Vec<Vec<f64>>, v2: &Vec<f64>) -> Vec<f64> {
    // matirx @ vec product
    assert_eq!(v1[0].len(), v2.len());
    let mut res = vec![];
    for i in 0..v1.len() {
        res.push(multiple(v2, &v1[i]));
    }
    res
}

pub fn multiple__(v1: &Vec<Vec<Vec<f64>>>, v2: &Vec<f64>) -> Vec<Vec<f64>> {
    // Perform a linear combination over the outermost dimension of the 3D matrix
    assert_eq!(v1.len(), v2.len());
    let mut res = vec![vec![0.0; v1[0][0].len()]; v1[0].len()];
    for weight_index in 0..v1.len() {
        v1[weight_index].iter().enumerate().for_each(|(i, x)| {
            x.iter()
                .enumerate()
                .for_each(|(j, x)| res[i][j] += x * v2[weight_index])
        });
    }
    res
}

pub fn sum_plus(v: &Vec<f64>, sigmoid: f64) -> f64 {
    // Sum the positive numbers in the list, with the default sigmoid value being 1
    v.iter()
        .map(|x| if x > &0.0 { (x).powf(sigmoid) } else { 0.0 })
        .sum()
}

pub fn linear_combine(v1: &mut Vec<f64>, v2: &Vec<f64>, w1: f64, w2: f64) {
    // linear combination of two vec
    assert_eq!(v1.len(), v2.len());
    assert!(w1 + w2 > 0.0);
    for i in 0..v1.len() {
        v1[i] = (v1[i] * w1 + v2[i] * w2) / (w1 + w2);
    }
    check_prob_distribution(v1);
}

pub fn check_prob_distribution(v: &Vec<f64>) {
    // Check if a vector is a distribution by verifying if the sum equals 1
    assert!(0.999 < v.iter().sum());
    assert!(v.iter().sum::<f64>() < 1.001);
}

pub fn max_index4vec(v: &Vec<f64>) -> (f64, usize) {
    // Get the maximum value from a Vec<f64> and the corresponding index. Note: f64 cannot be directly used with iter().max()
    check_nan(v);
    let mut max_value = v[0];
    let mut arg_max = 0;
    v.iter().enumerate().for_each(|(index, &x)| {
        if x > max_value {
            max_value = x;
            arg_max = index
        }
    });
    return (max_value, arg_max);
}

pub fn check_equal(v1: &Vec<f64>, v2: &Vec<f64>, precise: usize) {
    // Check if two values are equal within a specified precision
    for i in 0..v1.len() {
        assert!((v1[i] - v2[i]).abs() < 0.1_f64.powf(precise as f64));
    }
}

pub fn compute_entropy(prob: &Vec<f64>) -> f64 {
    prob.iter()
        .map(|&x| if x == 0.0 { 0.0 } else { x * x.ln() })
        .sum()
}

pub fn check_ranges_normal(ranges: &Vec<Vec<f64>>) -> (bool, usize) {
    for i in 0..ranges[0].len() {
        if ranges.iter().map(|x| x[i]).sum::<f64>() <= 0.0 {
            return (false, i);
        }
    }
    (true, 0)
}

pub fn broadcast_subtract(v1: &Vec<Vec<f64>>, v2: &Vec<f64>, dim: usize) -> Vec<Vec<f64>> {
    let mut res = vec![vec![]; v1.len()];
    if v2.len() == v1.len() && dim == 0 {
        for i in 0..v1.len() {
            for j in 0..v1[0].len() {
                res[i].push(v1[i][j] - v2[i])
            }
        }
    } else {
        assert!(dim == 1);
        for i in 0..v1.len() {
            for j in 0..v1[0].len() {
                res[i].push(v1[i][j] - v2[j])
            }
        }
    }
    return res;
}

pub fn round(v: &mut Vec<f64>, precise: i32) {
    v.iter_mut()
        .for_each(|x| *x = (f64::powi(10.0, precise) * *x).round() / f64::powi(10.0, precise));
}

pub fn flatten_outer_vec(input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
    // Flatten a 3D vector (Vec<Vec<Vec<f64>>>) into a 2D vector (Vec<Vec<f64>>)
    // Assumes that each Vec<Vec<f64>> has the same number of rows
    let mut result: Vec<Vec<f64>> = Vec::new();
    let num_rows = input[0].len();
    for row_idx in 0..num_rows {
        let mut new_row: Vec<f64> = Vec::new();
        for inner_vec in input.iter() {
            inner_vec[row_idx].iter().for_each(|&x| new_row.push(x));
        }
        result.push(new_row);
    }
    result
}

pub fn compute_prob_r_dist(v1: &Vec<Vec<f64>>, v2: &Vec<Vec<f64>>) -> f64 {
    // Normalize the input vectors first
    // Columns represent players, rows represent game states
    // Flatten the vectors and compute the squared distance between them
    let v1 = v1
        .iter()
        .map(|x| {
            let s: f64 = x.iter().sum();
            x.iter().map(|x| x / s).collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    let v2 = v2
        .iter()
        .map(|x| {
            let s: f64 = x.iter().sum();
            x.iter().map(|x| x / s).collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    let v1 = v1.into_iter().flatten().collect::<Vec<f64>>();
    let v2 = v2.into_iter().flatten().collect::<Vec<f64>>();
    (0..v1.len())
        .into_iter()
        .map(|x| (v1[x] - v2[x]).powi(2))
        .sum()
}

pub fn compute_prob_r_kldist(v1: &Vec<Vec<f64>>, v2: &Vec<Vec<f64>>) -> f64 {
    // Normalize, then compute the KL divergence between the new and old reach probabilities
    let num_player = v1[0].len();
    let s_v1 = (0..num_player)
        .into_iter()
        .map(|p| v1.iter().map(|x| x[p]).sum())
        .collect::<Vec<f64>>();
    let s_v2 = (0..num_player)
        .into_iter()
        .map(|p| v2.iter().map(|x| x[p]).sum())
        .collect::<Vec<f64>>();
    let v1 = (0..num_player)
        .into_iter()
        .map(|p| {
            if s_v1[p] > 0.0 {
                v1.iter().map(|x| x[p] / s_v1[p]).collect::<Vec<f64>>()
            } else {
                v1.iter().map(|x| x[p]).collect::<Vec<f64>>()
            }
        })
        .collect::<Vec<Vec<f64>>>();
    let v2 = (0..num_player)
        .into_iter()
        .map(|p| {
            if s_v2[p] > 0.0 {
                v2.iter().map(|x| x[p] / s_v2[p]).collect::<Vec<f64>>()
            } else {
                v2.iter().map(|x| x[p]).collect::<Vec<f64>>()
            }
        })
        .collect::<Vec<Vec<f64>>>();
    let mut kl_dist = (0..v1.len())
        .into_iter()
        .map(|x| kl_dist(&v1[x], &v2[x]))
        .collect::<Vec<f64>>();

    for i in 0..s_v1.len() {
        if s_v1[i] < 1e-8 && s_v2[i] > 1e-8 {
            // If a column in v1 is completely zero, the KL divergence for that column would be 0, which is incorrect!
            // A patch is applied here: if sv1=0 and sv2>0, set the value to a very large number
            // This is considered as an extremely large KL divergence in such cases
            kl_dist[i] = 1.0e10;
        }
    }
    let (max_kl_dist, _) = max_index4vec(&kl_dist);
    return max_kl_dist;
}

pub fn kl_dist(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    v1.iter()
        .enumerate()
        .map(|(k, v)| {
            if *v > 0.0 {
                v * (v / (v2[k] + 1e-20)).ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

pub fn hashmap_to_sorted_vec<T: Clone>(map: &HashMap<usize, T>) -> Vec<T> {
    let mut sorted_vec: Vec<_> = map.iter().collect();
    sorted_vec.sort_by_key(|&(key, _)| key);
    sorted_vec
        .into_iter()
        .map(|(_, value)| value.clone())
        .collect()
}

#[test]
fn test() {
    let m = vec![vec![2.0, 3.0, 1.0], vec![0.0, 1.0, 0.0]];
    let x = vec![-1.0, 3.0, 1.0];
    let res = multiple_(&m, &x);
    println!("{:?}", res);
    let m = vec![vec![vec![1.0; 3]; 3]; 3];
    let res = multiple__(&m, &x);
    println!("{:?}", res);
}

#[test]
fn test_kl() {
    let v1 = vec![vec![1.0, 1.0]];
    let v2 = vec![vec![1.0, 1.0]];
    println!("{}", compute_prob_r_kldist(&v1, &v2));
}
