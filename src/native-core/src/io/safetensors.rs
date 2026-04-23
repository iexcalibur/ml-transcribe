//! Load and save `.safetensors` files as collections of CPU `TensorId`s.
//!
//! Only F32 tensors are supported for now. Other dtypes (F16, BF16, I8…)
//! will be added once the model we actually care about (Cohere
//! Transcribe is fp16) needs them — at which point the loader will
//! up-convert on the way in.

use crate::tensor::{TensorId, TensorStore};
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
    UnsupportedDtype(Dtype),
    /// Saver was asked for a tensor id that the store does not have.
    UnknownTensorId(TensorId),
    /// The data buffer had a size inconsistent with its declared shape
    /// (safetensors crate usually catches this, but we double-check).
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

/// The result of `load_into`: a name-keyed map of tensor ids now
/// resident in the caller's `TensorStore`, plus a deterministic
/// iteration order.
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

/// Parse a safetensors file at `path` and copy each F32 tensor into the
/// store, returning a map from tensor name to the newly-assigned
/// `TensorId`. The caller is responsible for eventually freeing every
/// id in the returned map (e.g. via the FFI's `safetensors_free`,
/// which loops through and calls `TensorStore::free`).
pub fn load_into(
    path: &Path,
    store: &mut TensorStore,
) -> Result<LoadedWeights, SafetensorsError> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let mut map = HashMap::new();
    let mut sorted_names: Vec<String> = Vec::new();

    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F32 {
            return Err(SafetensorsError::UnsupportedDtype(view.dtype()));
        }
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let expected_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let data_bytes = view.data();
        if data_bytes.len() != expected_bytes {
            return Err(SafetensorsError::ShapeDataMismatch {
                expected_bytes,
                actual_bytes: data_bytes.len(),
            });
        }
        // bytemuck::cast_slice requires alignment; f32 in a safetensors
        // file is stored little-endian packed, which on aarch64/x86_64
        // matches the CPU's native representation. The data() pointer
        // is aligned to 1, not 4, so cast_slice can fail — copy via a
        // chunked iterator instead.
        let data: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let id = store.from_slice(&data, &shape);
        map.insert(name.to_string(), id);
        sorted_names.push(name.to_string());
    }

    sorted_names.sort();
    Ok(LoadedWeights { map, sorted_names })
}

/// Serialize a subset of the store (selected by `(name, id)` pairs) to
/// a safetensors file at `path`.
pub fn save(
    path: &Path,
    names_and_ids: &[(String, TensorId)],
    store: &TensorStore,
) -> Result<(), SafetensorsError> {
    // The safetensors crate wants `HashMap<String, TensorView>` where
    // TensorView borrows a `&[u8]` slice of the raw tensor bytes.
    //
    // We need to keep those byte buffers alive until serialize() runs,
    // so collect them into a Vec<Vec<u8>> first and borrow into it.
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

    let mut map: HashMap<String, TensorView> = HashMap::new();
    for (name, shape, bytes) in &buffers {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)?;
        map.insert(name.clone(), view);
    }
    safetensors::serialize_to_file(&map, &None, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_two_tensors() {
        let tmp = std::env::temp_dir().join("ml_transcribe_st_test.safetensors");
        // Round trip: write from one store, load into another, verify
        // shape and data match exactly.
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
        assert_eq!(loaded.sorted_names, vec!["bias", "weight"]);

        let weight_id = loaded.get("weight").unwrap();
        assert_eq!(store_r.shape(weight_id), &[2, 2]);
        assert_eq!(store_r.data(weight_id), &[1.0, 2.0, 3.0, 4.0]);

        let bias_id = loaded.get("bias").unwrap();
        assert_eq!(store_r.shape(bias_id), &[3]);
        assert_eq!(store_r.data(bias_id), &[10.0, 20.0, 30.0]);

        std::fs::remove_file(&tmp).ok();
    }
}
