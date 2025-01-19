//! The LLama2 model.

use candle_core::{DType, Device, IndexOp, Shape, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder, Linear, init};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::sync::{Arc, Mutex};
use anyhow::Context;
use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_transformers::quantized_nn::Linear as QLinear;

pub const MAX_SEQ_LEN: usize = 4096;

fn dump_tensor(x: &Tensor) {
    let mut file = File::create("/ceph-fuse/public/k8s/mavis/data/debug/dump-1.bin").unwrap();
    let x = x.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
    println!("Dumping tensor of shape {:?}", x.shape());
    x.write_bytes(&mut file).unwrap();
}

#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
        }
    }
}

pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

pub struct GGUFReader<'a, R: std::io::Seek + std::io::Read> {
    ct: gguf_file::Content,
    reader: &'a mut R
}

impl<'a, R: std::io::Seek + std::io::Read> GGUFReader<'a, R> {
    pub fn new(    ct: gguf_file::Content,
                   reader: &'a mut R) -> Self {
        Self { ct, reader}
    }

    pub fn get_qtensor(&mut self, device: &Device, prefix: &str, name: &str) -> anyhow::Result<QTensor> {
        let path = format!("{}.{}", prefix, name);
        let res = self.ct.tensor(self.reader, path.as_str(), device)?;
        Ok(res)
    }

    pub fn get_metadata(&mut self, path: &str) -> anyhow::Result<&gguf_file::Value> {
        self.ct.metadata.get(path).context("did not find metadata entry")
    }

    pub fn print_metadata(&self) {
        println!("Metadata keys:");
        for key in self.ct.metadata.keys() {
            println!("{key}");
        }
        println!("Layer keys:");
        for layer in self.ct.tensor_infos.keys() {
            println!("{}", layer);
        }
    }
}

pub struct LoraConfig {
    rank: usize,
    alpha: f64,
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self { rank, alpha }
    }
}

pub struct LoraLayer {
    ff_a: Linear,
    ff_b: Linear,
    scale: Option<f64>,
}

impl LoraLayer {
    pub fn new(in_dim: usize, out_dim: usize, config: &LoraConfig, vb: VarBuilder, dtype: DType) -> candle_core::Result<Self> {
        let ff_a = Linear::new(vb.pp("a").get_with_hints_dtype(
            (config.rank, in_dim),
            "weight",
            init::DEFAULT_KAIMING_NORMAL,
            dtype
        )?, None);
        let ff_b = Linear::new(vb.pp("b").get_with_hints_dtype(
            (out_dim, config.rank),
            "weight",
            init::ZERO,
            dtype
        )?, None);
        let scale = if config.rank > 0 {
            Some(config.alpha / config.rank as f64)
        } else {
            None
        };
        Ok(LoraLayer {
                ff_a,
                ff_b,
                scale
        })
    }

    pub fn forward(&self, input: &Tensor, frozen_output: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.ff_a.forward(input)?;
        let x = self.ff_b.forward(&x)?;
        frozen_output + x*self.scale.unwrap_or(1.0)
    }
}

pub struct LlamaLinear {
    inner: QMatMul,
    lora_layer: Option<LoraLayer>,
    in_out_dtype: DType,
    span: tracing::Span,
}

impl LlamaLinear {
    pub fn new_from_varbuilder(in_dim: usize, out_dim: usize, vb: VarBuilder, in_out_dtype: DType) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        let ws = vb.get((out_dim, in_dim), "weight")?;
        Ok(Self{
            inner: QMatMul::Tensor(ws),
            lora_layer: None,
            in_out_dtype,
            span,
        })
    }

    pub fn new_from_gguf<R: std::io::Seek + std::io::Read>(device: &Device, gguf: &mut GGUFReader<R>, prefix: &str, in_out_dtype: DType) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        let ws = gguf.get_qtensor(device, prefix, "weight")?;
        Ok(Self{
            inner: QMatMul::from_qtensor(ws)?,
            lora_layer: None,
            span,
            in_out_dtype,
        })
    }

    pub fn add_lora(&mut self, lora_config: &LoraConfig, vb: VarBuilder) -> candle_core::Result<()> {
        let shape = match &self.inner {
            QMatMul::QTensor(x) => {x.shape()}
            QMatMul::Tensor(x) => {x.shape()}
            QMatMul::TensorF16(x) => {x.shape()}
        };
        self.lora_layer = Some(LoraLayer::new(shape.dims()[1], shape.dims()[0], lora_config, vb, self.in_out_dtype)?);
        Ok(())
    }
}

impl Module for LlamaLinear {

    fn forward(&self, x: &Tensor) ->  candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        let base_out = self.inner.forward(x)?;

        //let mut val = self.inner.dequantize_f16();
        if let Some(lora_layer) = &self.lora_layer {
            lora_layer.forward(x, &base_out)
        }
        else {
            Ok(base_out)
        }
    }
}

#[derive(Clone)]
pub struct Cache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    num_hidden_layers: usize,
}

impl Cache {

    fn mask(&self, t: usize) -> anyhow::Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn reset(&mut self) {
        self.masks = Arc::new(Mutex::new(HashMap::new()));
        self.kvs = Arc::new(Mutex::new(vec![None; self.num_hidden_layers]));
    }

    pub fn get_cached_token_count(&self) -> usize {
        let w = self.kvs.lock().unwrap();
        match &w[0] {
            Some((k, _)) => {k.dims()[2]}
            None => 0,
        }
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    fn new_from_varbuilder(size: usize, eps: f64, vb: VarBuilder) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?;

        Ok(Self {
            span,
            weight,
            eps
        })
    }

    fn new_from_gguf<R: std::io::Seek + std::io::Read>(device: &Device, reader: &mut GGUFReader<R>, prefix: &str) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let rms_norm_eps = reader.get_metadata("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let weight = reader.get_qtensor(device, prefix, "weight")?;
        let weight = weight.dequantize(device)?;
        Ok(Self {
            weight,
            eps: rms_norm_eps,
            span
        })
    }
}


impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        Ok(x)
    }
}

struct CausalSelfAttention {
    q_proj: LlamaLinear,
    k_proj: LlamaLinear,
    v_proj: LlamaLinear,
    o_proj: LlamaLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    span: tracing::Span,
    span_rot: tracing::Span,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> anyhow::Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> anyhow::Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, n_head, seq_len, hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;

        let cos = cos
            .narrow(0, 0, seq_len)?
            .reshape((seq_len, hidden_size / 2, 1))?;
        let sin = sin
            .narrow(0, 0, seq_len)?
            .reshape((seq_len, hidden_size / 2, 1))?;

        let cos = cos.broadcast_as((b_sz, 1, seq_len, hidden_size/2, 1))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, hidden_size/2, 1))?;
        let x = x.reshape((b_sz, n_head, seq_len, hidden_size / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
        let rope = rope.flatten_from(D::Minus2)?;
        Ok(rope)
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &Cache, detach_grads: bool, use_flash_attn: bool) -> anyhow::Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        //println!("k shape (a): {:?}", k.shape());
        //println!("v shape (a): {:?}", v.shape());

        if cache.use_kv_cache {
            let mut cache = cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            if detach_grads {
                cache[block_idx] = Some((k.detach(), v.detach()))
            }
            else {
                cache[block_idx] = Some((k.clone(), v.clone()))
            }
        }
        //println!("k shape (b): {:?}", k.shape());
        //println!("v shape (b): {:?}", v.shape());

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            //println!("q shape (c): {:?}", q.shape());
            //println!("k shape (c): {:?}", k.shape());
            //println!("v shape (c): {:?}", v.shape());
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            //println!("att shape: {:?}", att.shape());


            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len)?;
                //println!("mask shape: {:?}", mask.shape());
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        let y = if detach_grads {
            y.detach()
        }
        else {
            y
        };
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> anyhow::Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    fn new_from_varbuilder(
        vb: VarBuilder,
        cfg: &Config,
        in_out_dtype: DType
    ) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = LlamaLinear::new_from_varbuilder(size_in, size_q, vb.pp("q_proj"), in_out_dtype)?;
        let k_proj = LlamaLinear::new_from_varbuilder(size_in, size_kv, vb.pp("k_proj"), in_out_dtype)?;
        let v_proj = LlamaLinear::new_from_varbuilder(size_in, size_kv, vb.pp("v_proj"), in_out_dtype)?;
        let o_proj = LlamaLinear::new_from_varbuilder(size_q, size_in, vb.pp("o_proj"), in_out_dtype)?;

        let mut this = Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            span,
            span_rot,
        };

        /*
        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_llama_csa"),
                Some(linear_config),
                None,
                None,
                None,
            )
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("lora_llama_csa"),
                Some(linear_config),
                None,
                None,
                None,
            )
        }
*/
        Ok(this)
    }

    fn new_from_gguf<R: std::io::Seek + std::io::Read>(device: &Device, reader: &mut GGUFReader<R>, prefix: &str, in_out_dtype: DType) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");

        let q_proj = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.attn_q"), in_out_dtype)?;
        let k_proj = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.attn_k"), in_out_dtype)?;
        let v_proj = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.attn_v"), in_out_dtype)?;
        let o_proj = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.attn_output"), in_out_dtype)?;

        let num_attention_heads =  reader.get_metadata("llama.attention.head_count")?.to_u32()? as usize;
        let num_key_value_heads =  reader.get_metadata("llama.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = reader.get_metadata("llama.embedding_length")?.to_u32()? as usize;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads,
            head_dim: embedding_length / num_attention_heads,
            span,
            span_rot,
        })
    }

    fn add_lora(&mut self, lora_config: &LoraConfig, vb: VarBuilder) -> candle_core::Result<()> {
        self.k_proj.add_lora(lora_config, vb.pp("attn"))?;
        self.v_proj.add_lora(lora_config, vb.pp("attn"))?;
        Ok(())
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> anyhow::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

struct Mlp {
    c_fc1: LlamaLinear,
    c_fc2: LlamaLinear,
    c_proj: LlamaLinear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn new_from_varbuilder(vb: VarBuilder, cfg: &Config, in_out_dtype: DType) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = LlamaLinear::new_from_varbuilder(h_size, i_size, vb.pp("gate_proj"), in_out_dtype)?;
        let c_proj = LlamaLinear::new_from_varbuilder(i_size, h_size, vb.pp("down_proj"), in_out_dtype)?;
        let c_fc2 = LlamaLinear::new_from_varbuilder(h_size, i_size, vb.pp("up_proj"), in_out_dtype)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }

    fn new_from_gguf<R: std::io::Seek + std::io::Read>(device: &Device, reader: &mut GGUFReader<R>, prefix: &str, in_out_dtype: DType) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let c_fc1 = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.ffn_gate"), in_out_dtype)?;
        let c_fc2 = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.ffn_up"), in_out_dtype)?;
        let c_proj = LlamaLinear::new_from_gguf(device, reader, &format!("{prefix}.ffn_down"), in_out_dtype)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &Cache, detach_grads: bool, use_flash_attn: bool) -> anyhow::Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;

        let x = (self.attn.forward(&x, index_pos, block_idx, cache, detach_grads, use_flash_attn)? + residual)?;

        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        let x = if detach_grads {
            x.detach()
        }
        else {
            x
        };
        Ok(x)
    }

    fn new_from_varbuilder(
        vb: VarBuilder,
        cfg: &Config,
        in_out_dtype: DType
    ) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::new_from_varbuilder(
            vb.pp("self_attn"),
            cfg,
            in_out_dtype
        )?;
        let mlp = Mlp::new_from_varbuilder(vb.pp("mlp"), cfg, in_out_dtype)?;
        let rms_1 = RmsNorm::new_from_varbuilder(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new_from_varbuilder(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }

    fn new_from_gguf<R: std::io::Seek + std::io::Read>(device: &Device, reader: &mut GGUFReader<R>, prefix: &str, in_out_dtype: DType) -> anyhow::Result<Block> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let rms_1 = RmsNorm::new_from_gguf(device, reader, &format!("{prefix}.attn_norm"))?;
        let attn = CausalSelfAttention::new_from_gguf(device, reader, prefix, in_out_dtype)?;
        let rms_2 = RmsNorm::new_from_gguf(device, reader, &format!("{prefix}.ffn_norm"))?;
        let mlp = Mlp::new_from_gguf(device, reader, prefix, in_out_dtype)?;

        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span
        })
    }

    fn add_lora(&mut self, lora_config: &LoraConfig, vb: VarBuilder) -> candle_core::Result<()> {
        self.attn.add_lora(lora_config, vb.pp("attn"))?;
        Ok(())
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: LlamaLinear,
    hidden_size: usize,
    num_attention_heads: usize,
    rope_theta: f32,
    main_dtype: DType
}

impl Llama {
    pub fn new_from_gguf<'a, R: std::io::Seek + std::io::Read>(device: &Device, ct: gguf_file::Content, reader: & mut R, in_out_dtype: DType) -> anyhow::Result<Self>
    {
        let mut gguf_reader = GGUFReader::new(ct, reader);
        gguf_reader.print_metadata();

        let embedding_length = gguf_reader.get_metadata("llama.embedding_length")?.to_u32()? as usize;
        let block_count = gguf_reader.get_metadata("llama.block_count")?.to_u32()? as usize;

        let tok_embeddings_q = gguf_reader.get_qtensor(device, "token_embd", "weight")?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let wte = Embedding::new(tok_embeddings, embedding_length);

        let mut blocks = Vec::new();
        for i in 0..block_count {
            blocks.push(Block::new_from_gguf(device, &mut gguf_reader, &format!("blk.{i}"), in_out_dtype)?);
        }

        let ln_f = RmsNorm::new_from_gguf(device, &mut gguf_reader, "output_norm")?;

        let lm_head = LlamaLinear::new_from_gguf(device, &mut gguf_reader, "output", in_out_dtype)?;

        let num_attention_heads =  gguf_reader.get_metadata("llama.attention.head_count")?.to_u32()? as usize;

        let rope_theta = gguf_reader.get_metadata("llama.rope.freq_base")
            .and_then(|m| m.to_f32().map_err(|e| e.into()))
            .unwrap_or(10000f32);

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            hidden_size: embedding_length,
            num_attention_heads,
            rope_theta,
            main_dtype: in_out_dtype
        })
    }

    pub fn new_from_varbuilder(
        vb: VarBuilder,
        config: Config,
        in_out_dtype: DType
    ) -> anyhow::Result<Self> {
        let embeddings = vb.pp("model.embed_tokens").get((config.vocab_size, config.hidden_size), "weight")?;
        let wte = Embedding::new(embeddings, config.hidden_size);

        let lm_head = LlamaLinear::new_from_varbuilder(config.hidden_size, config.vocab_size, vb.pp("lm_head"), in_out_dtype)?;
        let ln_f = RmsNorm::new_from_varbuilder(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..config.num_hidden_layers)
            .map(|i| {
                Block::new_from_varbuilder(
                    vb.pp(&format!("model.layers.{i}")),
                    &config,
                    in_out_dtype
                )
                    .unwrap()
            })
            .collect();

        Ok( Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            rope_theta: config.rope_theta,
            main_dtype: in_out_dtype
        })
    }

    pub fn add_lora(&mut self, lora_config: &LoraConfig, vb: VarBuilder) -> candle_core::Result<()> {
        for i in 0..self.blocks.len() {
            self.blocks[i].add_lora(lora_config, vb.pp(format!("block{i}")))?;
        }
        Ok(())
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &Cache, use_flash_attn: bool) -> anyhow::Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, true, use_flash_attn)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits.detach().to_dtype(DType::F32).map_err(|e| e)?)
    }

    pub fn forward_for_training(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> anyhow::Result<Tensor> {
        let (_b_sz, _seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        //println!("after wte: {}", x.backward().unwrap().get_ids().collect::<Vec<_>>().len());
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache, false, false)?;
            //println!("after block {}: {}", block_idx, x.backward().unwrap().get_ids().collect::<Vec<_>>().len());
        }
        //println!("after blocks: {}", x.backward().unwrap().get_ids().collect::<Vec<_>>().len());
        let x = self.ln_f.forward(&x)?;
        //println!("after ln_f: {}", x.backward().unwrap().get_ids().collect::<Vec<_>>().len());
        let x = self.lm_head.forward(&x)?;
        //println!("after lm_head: {}", x.backward().unwrap().get_ids().collect::<Vec<_>>().len());
        Ok(x)
    }

    pub fn new_cache(&self, device: &Device, use_kv_cache: bool) -> anyhow::Result<Cache> {

        // precompute freqs_cis
        let n_elem = self.hidden_size / self.num_attention_heads;
        //println!("n_elem: {}", n_elem);
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / self.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        //let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;

        let cos = idx_theta.cos()?.to_dtype(self.main_dtype)?;
        let sin = idx_theta.sin()?.to_dtype(self.main_dtype)?;

        //dump_tensor(&cos);
        //panic!();

        let num_hidden_layers = self.blocks.len();
        Ok(Cache {
            masks: Arc::new(Mutex::new(HashMap::new())),
            use_kv_cache,
            kvs: Arc::new(Mutex::new(vec![None; num_hidden_layers])),
            device: device.clone(),
            cos,
            sin,
            num_hidden_layers
        })
    }
}
