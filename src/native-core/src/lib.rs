//! Portable ML primitives: tensor storage, autograd, device backends, ops.
//!
//! This crate is deliberately FFI-free. The N-API glue for Node lives in
//! `src/native/` and the C ABI for Swift lives in `src/native-swift/`;
//! both depend on this crate.
//!
//! # Modules
//!
//! - [`allocator`] — caching tensor allocator shared across backends.
//! - [`autograd`] — reverse-mode autograd tape.
//! - [`device`] — CPU / CUDA / WebGPU backends (feature-gated).
//! - [`ops`] — elementwise, matmul, conv, attention, etc.
//! - [`tensor`] — `TensorStore` (id-keyed) and `GpuTensor` (the actual
//!   per-backend buffer). `TensorId = usize` is the public handle type.
//! - [`utils`] — misc helpers.
//!
//! # `Tensor` vs `tensor::TensorStore`
//!
//! The crate also exposes a simple owning [`Tensor`] struct at the top
//! level. It was introduced early to prove the Swift FFI pipeline end-
//! to-end without touching the full ml-framework machinery. The Swift
//! wrapper still uses it today via `native-swift`.
//!
//! The "real" tensor system is [`tensor::TensorStore`], which is what
//! the N-API layer and all ops/* code target. Swift will migrate to it
//! once we expose enough of the store API through the C ABI.

pub mod allocator;
pub mod autograd;
pub mod device;
pub mod io;
pub mod ops;
pub mod tensor;
pub mod utils;

// ---------------------------------------------------------------------------
// Simple owning tensor — used by the current Swift FFI surface.
// ---------------------------------------------------------------------------

/// A CPU-resident f32 tensor, laid out in row-major order.
///
/// Kept small and self-contained so the Swift bridge can exercise the
/// FFI pipeline independently of the full `TensorStore` machinery.
#[derive(Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Error returned from fallible `Tensor` constructors.
#[derive(Debug, PartialEq)]
pub enum TensorError {
    /// `data.len()` did not match the product of `shape`.
    ShapeMismatch { expected: usize, actual: usize },
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![0.0; size] }
    }

    pub fn iota(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        Tensor { shape: shape.to_vec(), data }
    }

    pub fn from_data(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::ShapeMismatch { expected, actual: data.len() });
        }
        Ok(Tensor { shape: shape.to_vec(), data })
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn as_slice(&self) -> &[f32] { &self.data }
    pub fn sum(&self) -> f32 { self.data.iter().sum() }
    pub fn numel(&self) -> usize { self.data.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_sums_to_zero() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.sum(), 0.0);
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn iota_sum_matches_gauss() {
        let t = Tensor::iota(&[10]);
        assert_eq!(t.sum(), 45.0);
    }

    #[test]
    fn from_data_round_trips() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_data(&[2, 3], data.clone()).unwrap();
        assert_eq!(t.as_slice(), data.as_slice());
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.sum(), 21.0);
    }

    #[test]
    fn from_data_rejects_wrong_length() {
        let err = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(err, TensorError::ShapeMismatch { expected: 6, actual: 3 });
    }
}
