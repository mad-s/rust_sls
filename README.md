# Sequential line search in Rust

Rust bindings for [sequential-line-search](https://github.com/yuki-koyama/sequential-line-search).

## Example
```rust
let target = vec![0.1f64, 0.2, 0.3, 0.4, 0.5];
let dims = target.len();
let mut sls = SLSFramework::new(dims);
for it in 0..10 {
    let a = sls.getParametersFromSlider(0.);
    let b = sls.getParametersFromSlider(1.);

    // Get closest point to `target` along slider

    // proj = <target-a, b-a>
    // pl   = <b-a, b-a> = |b-a|^2
    let proj : f64 = (b.iter().zip(&a).map(|(bi, ai)| bi-ai))
         .zip(target.iter().zip(&a).map(|(ti,ai)| ti-ai)).map(|(x, y)| x*y).sum();
    let pl : f64 = b.iter().zip(&a).map(|(bi, ai)| (bi-ai)*(bi-ai)).sum();

    let x = proj / pl;
    let x = x.max(0.).min(1.);

    sls.proceedOptimization(x);
}
println!("target: {:?}\nresult: {:?}", target, sls.getXmax());
```


## License

This project is available under the MIT License
