//! A `VarMap` is a store that holds named variables.
//!
use candle_core::{safetensors, DType, Device, NdArray, Result, Shape, Tensor, Var};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use candle_nn::Init;
use candle_nn::var_builder::SimpleBackend;

#[derive(Clone)]
pub enum VarOrTensor {
    Var(Var),
    Tensor(Tensor),
}

impl VarOrTensor {
    pub(crate) fn as_tensor(&self) -> &Tensor {
        match self {
            VarOrTensor::Var(var) => {var.as_tensor()}
            VarOrTensor::Tensor(tensor) => tensor
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            VarOrTensor::Var(var) => {var.shape()}
            VarOrTensor::Tensor(tensor) => tensor.shape()
        }
    }
}

/// A `VarMap` is a store that holds named variables. Variables can be retrieved from the stores
/// and new variables can be added by providing some initialization config in case they are
/// missing.
/// `VarMap` structures can be serialized in the safetensors format.
#[derive(Clone)]
pub struct HybridVarMap {
    data: Arc<Mutex<HashMap<String, VarOrTensor>>>,
}

impl HybridVarMap {
    /// Create a new empty `VarMap`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let data = Arc::new(Mutex::new(HashMap::new()));
        Self { data }
    }

    /// Retrieve all the variables currently stored in the map.
    pub fn all_vars(&self) -> Vec<Var> {
        let tensor_data = self.data.lock().unwrap();
        let mut output = Vec::new();
        for val in tensor_data.values() {
            if let VarOrTensor::Var(var) = val {
                output.push(var.clone());
            }
        }
        output
    }

    /// Retrieve or add a new variable.
    pub fn get<S: Into<Shape>>(
        &self,
        shape: S,
        path: &str,
        init: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<VarOrTensor> {
        let shape = shape.into();
        let mut tensor_data = self.data.lock().unwrap();
        if let Some(var_or_tensor) = tensor_data.get(path) {
            let tensor_shape = var_or_tensor.shape();
            if &shape != tensor_shape {
                candle_core::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return Ok(var_or_tensor.clone());
        }
        let var = init.var(shape, dtype, device)?;
        let var_or_tensor = VarOrTensor::Var(var);
        tensor_data.insert(path.to_string(), var_or_tensor.clone());
        Ok(var_or_tensor)
    }

    pub fn data(&self) -> &Mutex<HashMap<String, VarOrTensor>> {
        &self.data
    }
}

impl SimpleBackend for HybridVarMap {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        Ok(HybridVarMap::get(self, s, name, h, dtype, dev)?.as_tensor().clone())
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.data().lock().unwrap().contains_key(name)
    }
}