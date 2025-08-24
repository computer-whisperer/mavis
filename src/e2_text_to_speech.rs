use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder, RmsNorm, linear, linear_no_bias, Dropout, Sequential, Activation, seq, Embedding, embedding, Conv1d, Conv1dConfig};

struct E2RotaryEmbedding {
    inv_freq: Tensor
}

impl E2RotaryEmbedding {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let base = 10000;
        let inv_freq = vb.get((dim/2), "inv_freq")?;
        Ok(Self{inv_freq})
    }

    fn forward_from_seq_len(&self, seq_len: usize) -> candle_core::Result<Tensor> {
        let device = self.inv_freq.device();
        let t = Tensor::arange(0, seq_len as u32, device)?;
        self.forward(&t)
    }

    fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        //let max_pos = t.max()? + 1

        let t = if t.dims().len() == 1 {
            t.unsqueeze(0)?
        } else {
            t.clone()
        };

        let freqs = Tensor::ein
    }
}

struct E2Mish {

}

impl E2Mish {
    fn new() -> Self {
        Self
    }
}

impl Module for E2Mish {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let softplus = (Tensor::full(1.0, xs.shape(), xs.device()) + xs.exp()?).ln()?;
        Ok(xs*softplus.tanh())
    }
}

struct E2ConvPositionEmbedding {
    conv1d: Sequential
}

impl E2ConvPositionEmbedding {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let kernel_size = 31;
        let groups = 16;
        let conv_0 = Conv1d::new(
            vb.pp(0).get((dim, dim, kernel_size), "weight")?,
            Some(vb.pp(0).get((dim, dim, kernel_size), "bias")?),
            Conv1dConfig {
                groups,
                padding: kernel_size / 2,
                ..Default::default()
            }
        );
        let conv_2 = Conv1d::new(
            vb.pp(2).get((dim, dim, kernel_size), "weight")?,
            Some(vb.pp(2).get((dim, dim, kernel_size), "bias")?),
            Conv1dConfig {
                groups,
                padding: kernel_size / 2,
                ..Default::default()
            }
        );
        let conv1d = seq()
            .add(conv_0)
            .add(E2Mish::new())
            .add(conv_2)
            .add(E2Mish::new());
        Ok(Self {
            conv1d
        })
    }
}

impl candle_nn::Module for E2ConvPositionEmbedding{
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.permute((0, 2, 1))?;
        let x = self.conv1d.forward(&x)?;
        let out = x.permute((0, 2, 1))?;
        Ok(out)
    }
}

struct E2InputEmbedding {
    proj: Linear,
    conv_pos_embed: E2ConvPositionEmbedding,
}

impl E2InputEmbedding {
    fn new(mel_dim: usize, text_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let proj = linear(mel_dim*2 + text_dim, out_dim, vb.pp("proj"))?;
        let conv_pos_embed = E2ConvPositionEmbedding::new(out_dim, vb.pp("conv_pos_embed"))?;
        Ok(Self{
            proj,
            conv_pos_embed
        })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor, text_embed: &Tensor, drop_audio_cond: bool) -> candle_core::Result<Tensor> {
        let cond = if drop_audio_cond {
            Tensor::full(0.0, cond.shape(), cond.device())?
        } else {
            cond.clone()
        };
        let cat = Tensor::cat(&[x, &cond, text_embed], D::Minus1)?;
        let x = self.proj.forward(&cat)?;
        let x = (self.conv_pos_embed.forward(&x)? + x)?;
        Ok(x)
    }
}


struct E2TextEmbedding {
    text_embed: Embedding
}

impl E2TextEmbedding {
    fn new(text_num_embeds: usize, text_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self{
            text_embed: embedding(text_num_embeds+1, text_dim, vb.pp("text_embed"))?
        })
    }

    fn forward(&self, text: &Tensor, seq_len: usize, drop_text: bool) -> candle_core::Result<Tensor> {
        let text = (text+Tensor::full(1.0, text.shape(), text.device()))?;
        //let text = (1 + text)?;
        let text = text.i((.., ..seq_len))?;
        let (_batch, _text_len) = text.dims2()?;

        let text = if drop_text {
            Tensor::full(0.0, text.shape(), text.device())?
        } else {
            text
        };

        let text = self.text_embed.forward(&text)?;

        Ok(text)
    }
}

struct E2SinusPositionEmbedding {
    dim: usize
}

impl E2SinusPositionEmbedding {
    fn new(dim: usize) -> candle_core::Result<Self> {
        Ok(Self{
            dim
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let scale = 1000;
        let device = x.device();
        let half_dim = self.dim/2;
        let emb = 1000.0.ln()/((half_dim - 1) as f32);
        let emb = Tensor::exp(Tensor::arange(0, half_dim, device) * -emb)?;
        let emb = scale * x.unsqueeze(1) * emb.unsqueeze(0);
        let emb = Tensor::cat(&[emb.sin(), emb.cos()], candle_core::D::Minus1);
        Ok(emb)
    }
}

struct E2TimestepEmbedding {
    time_embed: E2SinusPositionEmbedding,
    time_mlp: Sequential
}

impl E2TimestepEmbedding {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let freq_embed_dim = 256;
        let time_embed = E2SinusPositionEmbedding::new(freq_embed_dim)?;
        let time_mlp = seq()
            .add(linear(freq_embed_dim, dim, vb.pp("time_mlp").pp(0)))
            .add(Activation::Silu)
            .add(linear(dim, dim, vb.pp("time_mlp").pp(1)));

        Ok(Self{
            time_embed,
            time_mlp
        })
    }

    fn forward(&self, time: &Tensor) -> candle_core::Result<Tensor> {
        let freq_embed = self.time_embed.forward(time)?;
        let freq_embed = self.time_mlp.forward(&freq_embed)?;
        Ok(freq_embed)
    }
}

fn e5_rmsnorm(dim: usize, vb: VarBuilder) -> candle_core::Result<RmsNorm> {
    let inner = RmsNorm::new(vb.get((dim), "g")?, 1e-05);
    Ok(inner)
}
struct E5FeedForward {
    ff: Sequential
}

impl E5FeedForward {
    fn new(dim: usize, mult: usize, dropout: f32, vb: VarBuilder) -> candle_core::Result<Self> {
        let inner_dim = dim*mult;
        let dim_out = dim;
        let ff_prefix = vb.pp("ff");
        let project_in = seq()
            .add(linear(dim, inner_dim, ff_prefix.pp(0).pp(0)))
            .add(Activation::Gelu);
        let ff = seq()
            .add(project_in)
            .add(Dropout::new(dropout))
            .add(linear(inner_dim, dim_out, ff_prefix.pp(2)));
        Ok(Self {
            ff
        })
    }
}

struct E5Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out_linear: Linear,
    to_out_dropout: Dropout,
}

impl E5Attention {
    fn new(dim: usize, heads: usize, dim_head: usize, dropout: f32, vb: VarBuilder) -> candle_core::Result<Self> {
        let inner_dim = dim_head*heads;

        let to_q = linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = linear(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = linear(dim, inner_dim, vb.pp("to_v"))?;

        let to_out_linear = linear(inner_dim, dim, vb.pp("to_out").pp("0"))?;
        let to_out_dropout = Dropout::new(dropout);

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out_linear,
            to_out_dropout
        })
    }
}

struct E2Layer {
    skip_proj: Linear,
    attn_norm: RmsNorm,
    attn: E5Attention,
    ff_norm: RmsNorm,
    ff: E5FeedForward
}

impl E2Layer {
    pub fn load(dim: usize, ff_mult: usize, heads: usize, dim_head: usize, dropout: f32, vb: VarBuilder) -> candle_core::Result<Self> {
        let skip_proj = linear_no_bias(dim*2, dim, vb.pp("0"))?;
        let attn_norm = e5_rmsnorm(dim, vb.pp("1"))?;
        let attn = E5Attention::new(dim, heads, dim_head, dropout, vb.pp("2"))?;
        let ff_norm = e5_rmsnorm(dim, vb.pp("3"))?;
        let ff = E5FeedForward::new(dim, ff_mult, dropout, vb.pp("4"))?;

        Ok(Self {
            skip_proj,
            attn_norm,
            attn,
            ff_norm,
            ff
        })
    }
}

pub struct E2TextToSpeech {
    time_embed: E2TimestepEmbedding,
    text_embed: E2TextEmbedding,
    input_embed: E2InputEmbedding,
    rotary_embed: E2RotaryEmbedding,
    layers: Vec<E2Layer>,
    norm_out: RmsNorm,
    proj_out: Linear,
    device: Device
}

impl E2TextToSpeech {
    pub fn new(device: &Device) -> candle_core::Result<Self> {

        let model_path = std::path::Path::new("/ceph-fuse/public/neural_models/text_to_speech/E2-TTS/E2TTS_Base/model_1200000.safetensors");

        let dtype = if device.supports_bf16() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb =  unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };

        let prefix = vb.pp("ema_model").pp("transformer");

        let dim = 1024;
        let depth = 24;
        let heads = 16;
        let ff_mult = 4;
        let mel_dim = 100;
        let text_dim = mel_dim;
        let text_num_embeds = 256;
        let dim_head = 64;

        let mut layers = vec![];
        let time_embed = E2TimestepEmbedding::new(dim, prefix.pp("time_embed"))?;
        let text_embed = E2TextEmbedding::new(text_num_embeds, text_dim, prefix.pp("text_embed"))?;
        let input_embed = E2InputEmbedding::new(mel_dim, text_dim, dim, prefix.pp("input_embed"))?;

        let rotary_embed = E2RotaryEmbedding::new(dim_head, prefix.pp("rotary_embed"))?;

        for i in 0..depth {
            layers.push(E2Layer::load(dim, ff_mult, heads, 64, 0.1, prefix.pp("layers").pp(i))?)
        }
        let norm_out = e5_rmsnorm(dim, vb.pp("norm_out"))?;
        let proj_out = linear(dim, mel_dim, prefix.pp("proj_out"))?;

        // Implementation goes here
        Ok(Self {
            time_embed,
            text_embed,
            input_embed,
            rotary_embed,
            layers,
            norm_out,
            proj_out,
            device: device.clone()
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor, text: &Tensor, time: &Tensor, drop_audio_cond: bool, drop_text: bool, mask: Option<&Tensor>) -> candle_core::Result<&Tensor> {
        let (batch, seq_len) = x.dims2()?;

        let time = if time.ndims() == 0 {
            time.repeat(batch)?
        } else {
            time
        };

        let t = self.time_embed.forward(time)?;
        let text_embed = self.text_embed.forward(text, seq_len, drop_text)?;
        let x = self.input_embed.forward(x, cond, text_embed, drop_audio_cond)?;

        let x = Tensor::cat(&[t.unsqueeze(1), x], 1)?;
        let mask = if let Some(mask) = mask {
            unimplemented!()
        } else {
            None
        };

        let rope = self.rotary_embed.forward_from_seq_len(seq_len + 1)?;

    }

    pub fn run(&self, text: &str) {

    }
}