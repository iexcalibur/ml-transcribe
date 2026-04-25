use crate::allocator::CachingAllocator;
use half::f16;
use rand::Rng;
use rand_distr::StandardNormal;

pub type TensorId = usize;

/// On-device storage dtype for a tensor.
///
/// Today only `F32` and `F16` are first-class — F16 is for storing
/// weights compactly (half the memory of F32). Ops still operate on
/// F32; F16 tensors are materialized to F32 transiently inside
/// `to_host`, so the F16 storage halves the *resident* memory of
/// loaded weights without any per-op API changes.
///
/// BF16 is decoded to F32 at load time (via `io::safetensors`); we
/// don't currently keep it in storage. Add when needed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
}

impl Dtype {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn shape_size(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

// =========================================================================
// CPU tensor
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
pub struct GpuTensor {
    /// F32 payload. Populated for `Dtype::F32` tensors; left empty for
    /// `Dtype::F16` tensors (which store their values in `data_f16`).
    pub data: Vec<f32>,
    /// F16 payload. Populated for `Dtype::F16` tensors; empty otherwise.
    /// Two bytes per element instead of four — half the resident memory
    /// for loaded weight matrices.
    pub data_f16: Vec<f16>,
    /// Which payload is the source of truth.
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub adam_m: Option<Vec<f32>>,
    pub adam_v: Option<Vec<f32>>,
}

#[cfg(any(feature = "cpu", feature = "webgpu"))]
impl GpuTensor {
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected *= self.shape[i];
        }
        true
    }

    /// Number of elements in the live payload (regardless of dtype).
    pub fn numel_storage(&self) -> usize {
        match self.dtype {
            Dtype::F32 => self.data.len(),
            Dtype::F16 => self.data_f16.len(),
        }
    }
}

// =========================================================================
// CUDA tensor
// =========================================================================

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
pub struct GpuTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub size: usize,
    pub requires_grad: bool,
    pub grad: Option<TensorId>,
    pub adam_m: Option<CudaSlice<f32>>,
    pub adam_v: Option<CudaSlice<f32>>,
}

#[cfg(feature = "cuda")]
impl GpuTensor {
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected *= self.shape[i];
        }
        true
    }
}

// =========================================================================
// TensorStore
// =========================================================================

pub struct TensorStore {
    pub(crate) tensors: Vec<Option<GpuTensor>>,
    pub(crate) free_ids: Vec<TensorId>,
    alloc: CachingAllocator,
}

// Shared logic (works for both CPU and CUDA)
impl TensorStore {
    fn insert(&mut self, t: GpuTensor) -> TensorId {
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(t);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(t));
            id
        }
    }

    pub fn get(&self, id: TensorId) -> &GpuTensor {
        self.tensors[id].as_ref().expect("tensor already freed")
    }

    pub fn get_mut(&mut self, id: TensorId) -> &mut GpuTensor {
        self.tensors[id].as_mut().expect("tensor already freed")
    }

    pub fn shape(&self, id: TensorId) -> &[usize] {
        &self.get(id).shape
    }

    pub fn size(&self, id: TensorId) -> usize {
        self.get(id).size
    }

    pub fn set_requires_grad(&mut self, id: TensorId, requires: bool) {
        self.get_mut(id).requires_grad = requires;
    }

    pub fn get_grad(&self, id: TensorId) -> Option<TensorId> {
        self.get(id).grad
    }

    pub fn ones_like(&mut self, id: TensorId) -> TensorId {
        let shape = self.get(id).shape.clone();
        self.ones(&shape)
    }

    pub fn ensure_grad(&mut self, id: TensorId) -> TensorId {
        if let Some(g) = self.get(id).grad {
            return g;
        }
        let shape = self.get(id).shape.clone();
        let grad_id = self.zeros(&shape);
        self.get_mut(id).grad = Some(grad_id);
        grad_id
    }
}

// =========================================================================
// CPU-specific TensorStore methods
// =========================================================================

#[cfg(any(feature = "cpu", feature = "webgpu"))]
impl TensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
            alloc: CachingAllocator::new(),
        }
    }

    pub fn free(&mut self, id: TensorId) {
        if let Some(t) = self.tensors[id].take() {
            self.alloc.dealloc(t.data);
            self.free_ids.push(id);
        }
    }

    /// Garbage-collect every tensor not listed in `keep`. Tensors held by
    /// the caller must have their IDs passed in to survive. Used by the
    /// N-API `gc_tensors` entry point.
    pub fn gc<I: IntoIterator<Item = TensorId>>(&mut self, keep: I) {
        let keep: std::collections::HashSet<TensorId> = keep.into_iter().collect();
        let len = self.tensors.len();
        for id in 0..len {
            if !keep.contains(&id) && self.tensors[id].is_some() {
                self.free(id);
            }
        }
    }

    /// Direct F32 borrow. Panics on F16 tensors — use `to_host` if you
    /// need a dtype-agnostic copy. Only valid for ops that mutate
    /// (training-time): forward-path ops should use `to_host` instead.
    pub fn data(&self, id: TensorId) -> &[f32] {
        let t = self.get(id);
        debug_assert_eq!(
            t.dtype, Dtype::F32,
            "TensorStore::data() called on a non-F32 tensor; use to_host()"
        );
        &t.data
    }

    pub fn data_mut(&mut self, id: TensorId) -> &mut [f32] {
        let t = self.get_mut(id);
        debug_assert_eq!(
            t.dtype, Dtype::F32,
            "TensorStore::data_mut() called on a non-F32 tensor"
        );
        &mut t.data
    }

    /// Read the tensor's elements as a contiguous `Vec<f32>`,
    /// regardless of the underlying storage dtype. F16 tensors are
    /// up-converted on the fly; the caller's Vec is dropped at end of
    /// scope, so memory peaks transiently but the F16 storage stays
    /// resident for the next call.
    pub fn to_host(&self, id: TensorId) -> Vec<f32> {
        let t = self.get(id);
        match t.dtype {
            Dtype::F32 => {
                if t.is_contiguous() { t.data.clone() }
                else { self.make_contiguous_data(id) }
            }
            Dtype::F16 => {
                // F16 is always stored contiguous (we never permute it
                // without converting back to F32 first), so a simple
                // map-collect suffices.
                t.data_f16.iter().map(|&h| h.to_f32()).collect()
            }
        }
    }

    pub fn get_scalar(&self, id: TensorId) -> f32 {
        let t = self.get(id);
        match t.dtype {
            Dtype::F32 => t.data[0],
            Dtype::F16 => t.data_f16[0].to_f32(),
        }
    }

    pub fn zero_grad(&mut self, id: TensorId) {
        if let Some(grad_id) = self.get(id).grad {
            let size = self.get(grad_id).size;
            let data = self.data_mut(grad_id);
            for i in 0..size {
                data[i] = 0.0;
            }
        }
    }

    pub fn zeros(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let data = self.alloc.alloc(size);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn ones(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        data.iter_mut().for_each(|x| *x = 1.0);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn rand(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        let mut rng = rand::thread_rng();
        data.iter_mut().for_each(|x| *x = rng.gen::<f32>());
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn randn(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        let mut rng = rand::thread_rng();
        data.iter_mut().for_each(|x| *x = rng.sample::<f32, _>(StandardNormal));
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn from_slice(&mut self, src: &[f32], shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut data = self.alloc.alloc(size);
        data[..size].copy_from_slice(&src[..size]);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn from_vec(&mut self, src: Vec<f32>, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data: src, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    /// Build an F16-storage tensor from a caller-owned Vec. The bytes
    /// stay as F16; ops up-convert transiently via `to_host`. This is
    /// the load path for keeping fp16 model weights at half the F32
    /// memory cost.
    pub fn from_vec_f16(&mut self, src: Vec<f16>, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data: Vec::new(),
            data_f16: src,
            dtype: Dtype::F16,
            shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        })
    }

    /// Return the storage dtype of the tensor at `id`.
    pub fn dtype(&self, id: TensorId) -> Dtype {
        self.get(id).dtype
    }

    /// Produce a new tensor whose storage matches `target`. F32 → F16
    /// halves the resident memory (with rounding). F16 → F32 doubles
    /// it but lets every op consume the tensor without per-call
    /// up-conversion.
    ///
    /// No-op (returns `id` unchanged) when the tensor is already in
    /// the requested dtype.
    pub fn cast(&mut self, id: TensorId, target: Dtype) -> TensorId {
        let src = self.get(id);
        if src.dtype == target {
            return id;
        }
        let shape = src.shape.clone();
        match target {
            Dtype::F32 => {
                // F16 → F32: lift bytes into F32 storage.
                let data: Vec<f32> = src.data_f16.iter().map(|&h| h.to_f32()).collect();
                self.from_vec(data, &shape)
            }
            Dtype::F16 => {
                // F32 → F16: round each element. We materialize a
                // contiguous source first to keep strides simple.
                let f32_data = if src.is_contiguous() {
                    src.data.clone()
                } else {
                    self.make_contiguous_data(id)
                };
                let f16_data: Vec<f16> = f32_data.iter()
                    .map(|&x| f16::from_f32(x))
                    .collect();
                self.from_vec_f16(f16_data, &shape)
            }
        }
    }

    pub fn accumulate_grad(&mut self, dst: TensorId, src: TensorId) {
        let grad_id = self.ensure_grad(dst);
        let size = self.get(src).size;
        let src_data = self.get(src).data.clone();
        let grad_data = self.data_mut(grad_id);
        for i in 0..size {
            grad_data[i] += src_data[i];
        }
    }

    /// In-place addition: dst[i] += src[i]
    pub fn add_inplace(&mut self, dst: TensorId, src: TensorId) {
        let size = self.get(src).size;
        let src_data = self.get(src).data.clone();
        let dst_data = self.data_mut(dst);
        for i in 0..size {
            dst_data[i] += src_data[i];
        }
    }

    fn make_contiguous_data(&self, id: TensorId) -> Vec<f32> {
        let t = self.get(id);
        let size = t.size;
        let ndim = t.shape.len();
        let mut out = vec![0.0f32; size];
        let out_strides = compute_strides(&t.shape);
        match t.dtype {
            Dtype::F32 => {
                for i in 0..size {
                    let mut src_idx = 0;
                    let mut rem = i;
                    for d in 0..ndim {
                        let coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        src_idx += coord * t.strides[d];
                    }
                    out[i] = t.data[src_idx];
                }
            }
            Dtype::F16 => {
                // F16 storage: gather + up-convert per element.
                for i in 0..size {
                    let mut src_idx = 0;
                    let mut rem = i;
                    for d in 0..ndim {
                        let coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        src_idx += coord * t.strides[d];
                    }
                    out[i] = t.data_f16[src_idx].to_f32();
                }
            }
        }
        out
    }

    pub fn ensure_contiguous(&mut self, id: TensorId) -> TensorId {
        if self.get(id).is_contiguous() {
            return id;
        }
        let data = self.make_contiguous_data(id);
        let shape = self.get(id).shape.clone();
        self.from_vec(data, &shape)
    }

    pub fn clear_alloc_cache(&mut self) {
        self.alloc.clear_cache();
    }
}

// =========================================================================
// CUDA-specific TensorStore methods
// =========================================================================

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
impl TensorStore {
    pub fn new() -> Self {
        let _ = crate::device::GpuDevice::instance();
        Self {
            tensors: Vec::new(),
            free_ids: Vec::new(),
            alloc: CachingAllocator::new(),
        }
    }

    pub fn free(&mut self, id: TensorId) {
        if let Some(t) = self.tensors[id].take() {
            self.alloc.dealloc(t.data);
            self.free_ids.push(id);
        }
    }

    pub fn dev_ptr(&self, id: TensorId) -> u64 {
        let dev = crate::device::GpuDevice::instance();
        dev.ptr(&self.get(id).data)
    }

    pub fn to_host(&self, id: TensorId) -> Vec<f32> {
        let dev = crate::device::GpuDevice::instance();
        let t = self.get(id);
        if t.is_contiguous() {
            dev.stream.memcpy_dtov(&t.data).unwrap()
        } else {
            let host = dev.stream.memcpy_dtov(&t.data).unwrap();
            let size = t.size;
            let ndim = t.shape.len();
            let out_strides = compute_strides(&t.shape);
            let mut out = vec![0.0f32; size];
            for i in 0..size {
                let mut src_idx = 0;
                let mut rem = i;
                for d in 0..ndim {
                    let coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    src_idx += coord * t.strides[d];
                }
                out[i] = host[src_idx];
            }
            out
        }
    }

    pub fn get_scalar(&self, id: TensorId) -> f32 {
        let dev = crate::device::GpuDevice::instance();
        let v = dev.stream.memcpy_dtov(&self.get(id).data).unwrap();
        v[0]
    }

    pub fn zero_grad(&mut self, id: TensorId) {
        if let Some(grad_id) = self.get(id).grad {
            let size = self.get(grad_id).size;
            let dev = crate::device::GpuDevice::instance();
            let out_ptr = dev.ptr(&self.get(grad_id).data);
            let func = dev.get_func("fill_f32");
            unsafe {
                dev.stream.launch_builder(func)
                    .arg(&out_ptr)
                    .arg(&0.0f32)
                    .arg(&(size as i32))
                    .launch(launch_cfg(size as u32))
                    .unwrap();
            }
        }
    }

    pub fn zeros(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let data = self.alloc.alloc(size);
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn ones(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let data = self.alloc.alloc(size);
        let strides = compute_strides(shape);
        let id = self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        });
        let dev = crate::device::GpuDevice::instance();
        let ptr = dev.ptr(&self.get(id).data);
        let func = dev.get_func("fill_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&ptr)
                .arg(&1.0f32)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
        id
    }

    pub fn rand(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut host = vec![0.0f32; size];
        let mut rng = rand::thread_rng();
        host.iter_mut().for_each(|x| *x = rng.gen::<f32>());
        self.from_slice(&host, shape)
    }

    pub fn randn(&mut self, shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let mut host = vec![0.0f32; size];
        let mut rng = rand::thread_rng();
        host.iter_mut().for_each(|x| *x = rng.sample::<f32, _>(StandardNormal));
        self.from_slice(&host, shape)
    }

    pub fn from_slice(&mut self, src: &[f32], shape: &[usize]) -> TensorId {
        let size = shape_size(shape);
        let dev = crate::device::GpuDevice::instance();
        let data = dev.stream.memcpy_stod(&src[..size]).unwrap();
        let strides = compute_strides(shape);
        self.insert(GpuTensor {
            data, shape: shape.to_vec(), strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
            data_f16: Vec::new(), dtype: Dtype::F32,
        })
    }

    pub fn from_vec(&mut self, src: Vec<f32>, shape: &[usize]) -> TensorId {
        self.from_slice(&src, shape)
    }

    pub fn accumulate_grad(&mut self, dst: TensorId, src: TensorId) {
        let grad_id = self.ensure_grad(dst);
        let size = self.get(src).size;
        let dev = crate::device::GpuDevice::instance();
        let src_ptr = dev.ptr(&self.get(src).data);
        let grad_ptr = dev.ptr(&self.get(grad_id).data);
        let func = dev.get_func("add_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&grad_ptr)
                .arg(&grad_ptr)
                .arg(&src_ptr)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
    }

    pub fn add_inplace(&mut self, dst: TensorId, src: TensorId) {
        let size = self.get(src).size;
        let dev = crate::device::GpuDevice::instance();
        let src_ptr = dev.ptr(&self.get(src).data);
        let dst_ptr = dev.ptr(&self.get(dst).data);
        let func = dev.get_func("add_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&dst_ptr)
                .arg(&dst_ptr)
                .arg(&src_ptr)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
    }

    pub fn ensure_contiguous(&mut self, id: TensorId) -> TensorId {
        if self.get(id).is_contiguous() {
            return id;
        }
        let data = self.to_host(id);
        let shape = self.get(id).shape.clone();
        self.from_vec(data, &shape)
    }

    /// Copy tensor data on device with a new shape (no CPU roundtrip).
    pub fn clone_device(&mut self, src: TensorId, new_shape: &[usize]) -> TensorId {
        let size = self.size(src);
        let out = self.zeros(new_shape);
        let dev = crate::device::GpuDevice::instance();
        let src_ptr = dev.ptr(&self.get(src).data);
        let out_ptr = dev.ptr(&self.get(out).data);
        let func = dev.get_func("copy_f32");
        unsafe {
            dev.stream.launch_builder(func)
                .arg(&out_ptr)
                .arg(&src_ptr)
                .arg(&(size as i32))
                .launch(launch_cfg(size as u32))
                .unwrap();
        }
        out
    }

    pub fn clear_alloc_cache(&mut self) {
        self.alloc.clear_cache();
    }
}
