//! Load and save `.safetensors` files as collections of CPU `TensorId`s.
//!
//! The internal `TensorStore` only holds `f32` tensors. For load, we
//! accept `F32`, `F16`, and `BF16` and up-convert the lower-precision
//! variants into `f32` on the way in. For save, F32 and F16 are
//! supported; BF16 save can be added the same way if needed.

use crate::tensor::{Dtype as StoreDtype, TensorId, TensorStore};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Error surface for both load and save. Mapped to a `u32` error code
/// by the FFI layer.
#[derive(Debug)]
pub enum SafetensorsError {
    Io(std::io::Error),
    Parse(safetensors::SafeTensorError),
    /// The file contained a dtype we don't currently up-convert.
    UnsupportedDtype(Dtype),
    UnknownTensorId(TensorId),
    ShapeDataMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
}

impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<safetensors::SafeTensorError> for SafetensorsError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Parse(e)
    }
}

pub struct LoadedWeights {
    pub map: HashMap<String, TensorId>,
    pub sorted_names: Vec<String>,
}

impl LoadedWeights {
    pub fn len(&self) -> usize {
        self.sorted_names.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sorted_names.is_empty()
    }
    pub fn name_at(&self, idx: usize) -> Option<&str> {
        self.sorted_names.get(idx).map(String::as_str)
    }
    pub fn get(&self, name: &str) -> Option<TensorId> {
        self.map.get(name).copied()
    }
}

// ---------------------------------------------------------------------------
// Dtype -> f32 up-conversion
// ---------------------------------------------------------------------------

/// Decode the raw bytes of a tensor into a fresh `Vec<f32>`, regardless
/// of the source dtype. Returns an error for dtypes we can't handle.
///
/// All three paths produce the same row-major ordering; only the
/// per-element byte decoding differs.
fn decode_to_f32(
    dtype: Dtype,
    bytes: &[u8],
    numel: usize,
) -> Result<Vec<f32>, SafetensorsError> {
    match dtype {
        Dtype::F32 => {
            if bytes.len() != numel * 4 {
                return Err(SafetensorsError::ShapeDataMismatch {
                    expected_bytes: numel * 4,
                    actual_bytes: bytes.len(),
                });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Dtype::F16 => {
            if bytes.len() != numel * 2 {
                return Err(SafetensorsError::ShapeDataMismatch {
                    expected_bytes: numel * 2,
                    actual_bytes: bytes.len(),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            if bytes.len() != numel * 2 {
                return Err(SafetensorsError::ShapeDataMismatch {
                    expected_bytes: numel * 2,
                    actual_bytes: bytes.len(),
                });
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => Err(SafetensorsError::UnsupportedDtype(other)),
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

pub fn load_into(
    path: &Path,
    store: &mut TensorStore,
) -> Result<LoadedWeights, SafetensorsError> {
    load_into_with_dtype(path, store, /*keep_f16=*/ false)
}

/// Like `load_into`, but if `keep_f16` is true, F16 AND BF16 source
/// tensors are stored as F16 in the engine (saving half the memory)
/// instead of being up-converted to F32 on entry. F32 source dtypes
/// always produce F32 storage.
///
/// Cohere Transcribe-2B specifically: its 4.13 GiB checkpoint is
/// almost entirely BF16. With `keep_f16=true` the in-memory footprint
/// is ~4 GiB instead of the ~8 GiB you'd get with the F32 path —
/// crucial when the WeightMap then materializes transposed copies for
/// every PyTorch Linear weight on top.
pub fn load_into_with_dtype(
    path: &Path,
    store: &mut TensorStore,
    keep_f16: bool,
) -> Result<LoadedWeights, SafetensorsError> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let mut map = HashMap::new();
    let mut sorted_names: Vec<String> = Vec::new();

    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let numel: usize = shape.iter().product();
        let id = if keep_f16 && view.dtype() == Dtype::F16 {
            // Decode the raw bytes into Vec<f16> without an F32
            // intermediate, then store as F16.
            let raw = view.data();
            if raw.len() != numel * 2 {
                return Err(SafetensorsError::ShapeDataMismatch {
                    expected_bytes: numel * 2,
                    actual_bytes: raw.len(),
                });
            }
            let data_f16: Vec<f16> = raw
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect();
            store.from_vec_f16(data_f16, &shape)
        } else if keep_f16 && view.dtype() == Dtype::BF16 {
            // BF16 → F16 via F32 element-by-element. We never
            // materialize a full F32 buffer, so peak memory stays
            // at the 2-bytes-per-element BF16 input plus the
            // 2-bytes-per-element F16 output.
            let raw = view.data();
            if raw.len() != numel * 2 {
                return Err(SafetensorsError::ShapeDataMismatch {
                    expected_bytes: numel * 2,
                    actual_bytes: raw.len(),
                });
            }
            let data_f16: Vec<f16> = raw
                .chunks_exact(2)
                .map(|c| {
                    let bf = bf16::from_le_bytes([c[0], c[1]]);
                    f16::from_f32(bf.to_f32())
                })
                .collect();
            store.from_vec_f16(data_f16, &shape)
        } else {
            // Default path: everything ends up as F32 in the store.
            // Skip dtypes we can't represent in our F32/F16 store
            // (notably I64 — used by BatchNorm's `num_batches_tracked`,
            // an inference-irrelevant counter). The skipped tensor's
            // name doesn't enter `map`, so any consumer that asks
            // for it gets a clean "missing weight" error.
            match decode_to_f32(view.dtype(), view.data(), numel) {
                Ok(data) => store.from_slice(&data, &shape),
                Err(SafetensorsError::UnsupportedDtype(d)) => {
                    eprintln!(
                        "[safetensors] skipping '{}' with unsupported dtype {:?}",
                        name, d
                    );
                    continue;
                }
                Err(e) => return Err(e),
            }
        };
        map.insert(name.to_string(), id);
        sorted_names.push(name.to_string());
    }

    sorted_names.sort();
    Ok(LoadedWeights { map, sorted_names })
}

// `StoreDtype` is our crate's storage enum (F32 / F16) imported as
// an alias at the top of the file; `Dtype` here refers to
// safetensors' on-disk dtype enum (F32 / F16 / BF16 / I8 / etc).
// We could phrase this more explicitly with a fully qualified path,
// but the alias keeps the call sites readable.
#[allow(dead_code)]
fn _enforce_dtype_alias_use(_d: StoreDtype) {}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

/// Serialize tensors as F32.
pub fn save(
    path: &Path,
    names_and_ids: &[(String, TensorId)],
    store: &TensorStore,
) -> Result<(), SafetensorsError> {
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(names_and_ids.len());
    for (name, id) in names_and_ids {
        let shape = store.shape(*id).to_vec();
        let data = store.data(*id);
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        buffers.push((name.clone(), shape, bytes));
    }
    serialize_buffers(path, &buffers, Dtype::F32)
}

/// Serialize tensors as F16: `f32 → f16::from_f32 → little-endian bytes`.
/// Loses precision but halves the file size; matches how most HF
/// transformer weights are distributed.
pub fn save_as_f16(
    path: &Path,
    names_and_ids: &[(String, TensorId)],
    store: &TensorStore,
) -> Result<(), SafetensorsError> {
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(names_and_ids.len());
    for (name, id) in names_and_ids {
        let shape = store.shape(*id).to_vec();
        let data = store.data(*id);
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &f in data {
            bytes.extend_from_slice(&f16::from_f32(f).to_le_bytes());
        }
        buffers.push((name.clone(), shape, bytes));
    }
    serialize_buffers(path, &buffers, Dtype::F16)
}

fn serialize_buffers(
    path: &Path,
    buffers: &[(String, Vec<usize>, Vec<u8>)],
    dtype: Dtype,
) -> Result<(), SafetensorsError> {
    let mut map: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in buffers {
        let view = TensorView::new(dtype, shape.clone(), bytes)?;
        map.insert(name.clone(), view);
    }
    safetensors::serialize_to_file(&map, &None, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_f32() {
        let tmp = std::env::temp_dir().join("ml_transcribe_st_f32.safetensors");
        let mut store_w = TensorStore::new();
        let id_a = store_w.from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let id_b = store_w.from_slice(&[10.0, 20.0, 30.0], &[3]);
        save(
            &tmp,
            &[("weight".into(), id_a), ("bias".into(), id_b)],
            &store_w,
        )
        .unwrap();

        let mut store_r = TensorStore::new();
        let loaded = load_into(&tmp, &mut store_r).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(store_r.data(loaded.get("weight").unwrap()), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store_r.data(loaded.get("bias").unwrap()), &[10.0, 20.0, 30.0]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn round_trip_f16_exact_for_integers() {
        // Small positive integers are exactly representable in f16.
        let tmp = std::env::temp_dir().join("ml_transcribe_st_f16.safetensors");
        let mut store_w = TensorStore::new();
        let id = store_w.from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, 6.0], &[2, 3]);
        save_as_f16(&tmp, &[("w".into(), id)], &store_w).unwrap();

        let mut store_r = TensorStore::new();
        let loaded = load_into(&tmp, &mut store_r).unwrap();
        assert_eq!(
            store_r.data(loaded.get("w").unwrap()),
            &[1.0, -2.0, 3.0, -4.0, 5.0, 6.0]
        );
        assert_eq!(store_r.shape(loaded.get("w").unwrap()), &[2, 3]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn round_trip_f16_lossy_for_irrational() {
        // pi is not exactly representable in f16; expect ~3 decimal digits.
        let tmp = std::env::temp_dir().join("ml_transcribe_st_f16_lossy.safetensors");
        let mut store_w = TensorStore::new();
        let id = store_w.from_slice(&[std::f32::consts::PI], &[1]);
        save_as_f16(&tmp, &[("pi".into(), id)], &store_w).unwrap();

        let mut store_r = TensorStore::new();
        let loaded = load_into(&tmp, &mut store_r).unwrap();
        let round_tripped = store_r.data(loaded.get("pi").unwrap())[0];
        assert!((round_tripped - std::f32::consts::PI).abs() < 1e-3);

        std::fs::remove_file(&tmp).ok();
    }
}
