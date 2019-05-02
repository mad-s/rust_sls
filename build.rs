extern crate cpp_build;

extern crate pkg_config;

fn main() {
    let eigen = pkg_config::probe_library("eigen3").expect("Library eigen3 not found");

    let mut config = cpp_build::Config::new();
    config.include("sequential-line-search/include/");
    for path in &eigen.include_paths {
        config.include(path);
    }

    // Builds the project in the directory located in `libfoo`, installing it
    // into $OUT_DIR
    let dst = cmake::build("sequential-line-search");
    dbg!(&dst);
    //config.object(dst.join("lib/libSequentialLineSearch.a"));

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=SequentialLineSearch");
    println!("cargo:rustc-link-lib=dylib=nlopt");

    config.build("src/lib.rs");
}
