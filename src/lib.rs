//! Rust Bindings for [Sequential Line Search](https://koyama.xyz/project/sequential_line_search/).
//!
//! Sequential Line Search is a generic human-in-the-loop optimization algorithm that optimizes for
//! the user's preference, by repeatedly asking them to select their favorite on a slider.
//!
//! The [SLSFramework] type holds the state of the algorithm, look there for the available methods.
//!
//! # Example:
//! ```
//! let target = vec![0.1f64, 0.2, 0.3, 0.4, 0.5];
//! let dims = target.len();
//! let mut sls = SLSFramework::new(dims);
//! for it in 0..10 {
//!     let a = sls.get_parameters_from_slider(0.);
//!     let b = sls.get_parameters_from_slider(1.);
//!
//!     // Get closest point to `target` along slider
//!
//!     // proj = <target-a, b-a>
//!     // pl   = <b-a, b-a> = |b-a|^2
//!     let proj : f64 = (b.iter().zip(&a).map(|(bi, ai)| bi-ai))
//!          .zip(target.iter().zip(&a).map(|(ti,ai)| ti-ai)).map(|(x, y)| x*y).sum();
//!     let pl : f64 = b.iter().zip(&a).map(|(bi, ai)| (bi-ai)*(bi-ai)).sum();
//!
//!     let x = proj / pl;
//!     let x = x.max(0.).min(1.);
//!
//!     sls.proceed_optimization(x);
//! }
//! println!("target: {:?}\nresult: {:?}", target, sls.get_x_max());
//! ```

#![recursion_limit="512"]

#[macro_use]
extern crate cpp;

cpp! {{
    #include <iostream>
    #include <memory>
    #include <sequential-line-search/sequential-line-search.h>
    using namespace sequential_line_search;
    using namespace Eigen;

    struct SLSFramework {
        std::shared_ptr<sequential_line_search::PreferenceRegressor> regressor;
        std::shared_ptr<sequential_line_search::Slider> slider;

        sequential_line_search::Data data;

        size_t          dimension;
        Eigen::VectorXd x_max;
        double          y_max;

        SLSFramework(size_t d) :
            dimension(d),
            regressor(nullptr), slider(nullptr),
            data(),
            x_max(VectorXd::Zero(0)),
            y_max(NAN)
        {
            computeRegression();
            updateSliderEnds();
        }

        void computeRegression()
        {
            regressor = std::make_shared<PreferenceRegressor>(data.X, data.D);
        }

        void updateSliderEnds()
        {
            // If this is the first time...
            if (x_max.rows() == 0)
            {
                slider = std::make_shared<Slider>(utils::generateRandomVector(dimension), utils::generateRandomVector(dimension), true);
                return;
            }

            const VectorXd x_1 = regressor->find_arg_max();
            const VectorXd x_2 = acquisition_function::FindNextPoint(*regressor);

            slider = std::make_shared<Slider>(x_1, x_2, true);
        }

        const VectorXd computeParametersFromSlider(double value)
        {
            return slider->end_0 * (1.0 - value) + slider->end_1 *  value;
        }

        void proceedOptimization(double slider_position)
        {
            // Add new preference data
            const VectorXd x = computeParametersFromSlider(slider_position);
            data.AddNewPoints(x, { slider->orig_0, slider->orig_1 });

            // Compute regression
            computeRegression();

            // Check the current best
            unsigned index;
            y_max = regressor->y.maxCoeff(&index);
            x_max = regressor->X.col(index);

            // Update slider ends
            updateSliderEnds();
        }

    };
}}

cpp_class!(
    /// State of the sequential line search algorithm
    pub unsafe struct SLSFramework as "SLSFramework"
);


unsafe fn as_rust_vec(ev: *const u8) -> Vec<f64> {
    let dim = cpp!([ev as "const VectorXd*"] -> usize as "size_t" {
            return ev->rows();
        });

    let mut res = vec![0.0f64; dim];
    let ptr = res.as_mut_ptr();
    cpp!([ev as "const VectorXd*", dim as "size_t", ptr as "double*"] {
        for (size_t i = 0; i < dim; ++i) {
            ptr[i] = (*ev)(i);
        }
    });
    res
}

impl SLSFramework {
    /// Initializes the algorithm.
    ///
    /// `dim` is the number of dimensions of the parameter space
    pub fn new(dim: usize) -> Self {
        unsafe {
        cpp!([dim as "size_t"] -> SLSFramework as "SLSFramework" {
            return SLSFramework(dim);
        })
        }
    }

    /// Take one step in the algorithm.
    ///
    /// `pos` (`0 <= pos <= 1`) is the best position along the current slider
    pub fn proceed_optimization(&mut self, pos: f64) {
        unsafe {
            cpp!([self as "SLSFramework*", pos as "double"] {
                self->proceedOptimization(pos);
            });
        }
    }

    /// Get positions along the current slider
    ///
    /// `pos` (`0 <= pos <= 1`) is the position along the slider
    pub fn get_parameters_from_slider(&self, pos: f64) -> Vec<f64> {
        unsafe {
            let eigen_vec = cpp!(
                [self as "SLSFramework*", pos as "double"]
                  -> *const u8 as "const VectorXd *"
            {
                VectorXd x = self->computeParametersFromSlider(pos);
                return new VectorXd(x);
            });
            let rsv = as_rust_vec(eigen_vec);
            cpp!([eigen_vec as "const VectorXd *"] {
                delete eigen_vec;
            });
            rsv
        }
    }

    /// Get the best position to date
    pub fn get_x_max(&self) -> Vec<f64> {
        unsafe {
            let eigen_vec = cpp!(
                [self as "SLSFramework*"]
                  -> *const u8 as "const VectorXd *"
            {
                return &self->x_max;
            });
            as_rust_vec(eigen_vec)
        }
    }
}

#[test]
fn test_point() {

    let target = vec![0.1f64, 0.2, 0.3, 0.4, 0.5];
    dbg!(&target);

    let dims = target.len();

    let mut sls = SLSFramework::new(dims);
    for it in 0..10 {
        let a = sls.get_parameters_from_slider(0.);
        let b = sls.get_parameters_from_slider(1.);
        dbg!(&a);
        dbg!(&b);


        // proj = <target-a, b-a>
        // pl   = <b-a, b-a> = |b-a|^2
        let proj : f64 = (b.iter().zip(&a).map(|(bi, ai)| bi-ai))
             .zip(target.iter().zip(&a).map(|(ti,ai)| ti-ai)).map(|(x, y)| x*y).sum();
        let pl : f64 = b.iter().zip(&a).map(|(bi, ai)| (bi-ai)*(bi-ai)).sum();

        let x = proj / pl;
        let x = x.max(0.).min(1.);
        dbg!(x);

        sls.proceed_optimization(x);
    }

    dbg!(sls.get_x_max());
}

