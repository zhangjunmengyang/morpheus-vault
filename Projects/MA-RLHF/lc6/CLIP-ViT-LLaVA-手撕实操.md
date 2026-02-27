---
title: "CLIP / ViT / LLaVA 多模态手撕实操"
brief: "多模态核心架构完整实现：ViT（图像patch→token序列）、CLIP（双塔对比学习+InfoNCE loss）、LLaVA（视觉投影层MLP→LLM多模态融合），图像理解→文本生成完整流水线，来源 MA-RLHF 教学项目。"
date: 2026-02-25
type: code-practice
source: "MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)"
tags: [code-practice, clip, vit, llava, multimodal, contrastive-learning, pytorch]
related:
  - "[[Projects/MA-RLHF/lc2/lc2-01-Transformer-手撕实操|Transformer-手撕实操]]"
  - "[[Projects/MA-RLHF/lc1/lc1-02-基础数学组件手撕|基础数学组件手撕]]"
---

# CLIP / ViT / LLaVA 多模态手撕实操

> 来源：MA-RLHF (https://github.com/dhcode-cpp/MA-RLHF)
> 入库日期：2026-02-25

---

## 目录

- [1. ViT — Vision Transformer](#1-vit--vision-transformer)
- [2. CLIP — 双塔对比学习](#2-clip--双塔对比学习)
- [3. CLIP Loss — InfoNCE 对比学习损失](#3-clip-loss--infonce-对比学习损失)
- [4. LLaVA — 多模态大语言模型](#4-llava--多模态大语言模型)

---

## 1. ViT — Vision Transformer

**核心思想**：将图像切分为固定大小的 patch，每个 patch 经线性投影后作为 token 送入标准 Transformer Encoder。

### 架构关键组件

```
Image [B, 3, 224, 224]
  → Patch Embedding (Conv2d: kernel=patch_size, stride=patch_size)
  → [B, num_patches, hidden_size]
  → Prepend [CLS] token
  → Add Positional Embedding
  → Transformer Encoder × N layers
  → LayerNorm
  → [CLS] token → Classification Head
```

### 1.1 Config

```python
class ViTConfig():
    model_type = "vit"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
```

### 1.2 Patch Embedding

> **关键洞察**：用 `Conv2d(kernel_size=patch_size, stride=patch_size)` 一步完成 patch 切分 + 线性投影，输入 `[B, C, H, W]` → 输出 `[B, num_patches, hidden_size]`。本质是把每个 16×16 patch 当作一次卷积运算。

```python
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        projection = self.projection(pixel_values)  # [B, hidden, H/p, W/p]
        flatten = projection.flatten(2)              # [B, hidden, num_patches]
        embeddings = flatten.transpose(1, 2)         # [B, num_patches, hidden]
        return embeddings
```

### 1.3 ViT Embeddings（CLS Token + Position Embedding）

> **关键洞察**：`cls_token` 是可学习的 `[1, 1, D]` 参数，`expand` 到 batch 维度后 cat 到 patch 序列前面。位置编码同样是可学习参数 `[1, num_patches+1, D]`，直接加到嵌入上。

```python
class ViTEmbeddings(nn.Module):
    def __init__(self, config, use_mask_token=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
```

### 1.4 Self-Attention

```python
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, seq_len, head_dim = k.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.attention = ScaleDotProductAttention()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        out, score = self.attention(q, k, v, head_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        outputs = (out, score) if output_attentions else (out,)
        return outputs
```

### 1.5 Transformer Layer（Pre-Norm 结构）

> **关键洞察**：ViT 使用 **Pre-Norm**（先 LayerNorm 再 Attention/FFN），与原始 Transformer 的 Post-Norm 不同。残差连接在 norm 之后。

```python
class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)  # up-project + GELU
        self.output = ViTOutput(config)               # down-project + residual
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # Pre-Norm + Self-Attention + Residual
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), head_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        hidden_states = attention_output + hidden_states
        # Pre-Norm + FFN + Residual
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)  # 内含 residual
        return (layer_output,) + self_attention_outputs[1:]
```

### 1.6 Classification Head

> **关键洞察**：分类时只取 `[CLS]` token（序列第 0 号位置）的输出过线性层。

```python
class ViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, labels=None, **kwargs):
        outputs = self.vit(pixel_values, **kwargs)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])  # CLS token
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits, outputs.hidden_states, outputs.attentions)
```

### 1.7 Position Interpolation（不同分辨率适配）

当输入分辨率与训练不同时，ViT 通过对位置编码做 **bicubic 插值** 来适配新的 patch 数量：

```python
# 核心逻辑：将 position_embeddings reshape 成 2D grid，用 nn.functional.interpolate 缩放
patch_pos_embed = patch_pos_embed.reshape(1, sqrt(N), sqrt(N), dim).permute(0, 3, 1, 2)
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed,
    scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
    mode="bicubic", align_corners=False,
)
patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
```

---

## 2. CLIP — 双塔对比学习

**核心思想**：图像编码器和文本编码器分别提取特征，投影到共享嵌入空间，通过对比学习训练对齐图文表示。

### 架构概览

```
Image → CLIPVisionEncoder → visual_projection → image_embeds (normalized)
Text  → CLIPTextEncoder   → text_projection   → text_embeds  (normalized)

similarity = image_embeds @ text_embeds.T * exp(temperature)
loss = (cross_entropy(sim, labels) + cross_entropy(sim.T, labels)) / 2
```

### 2.1 Config

```python
class CLIPTextConfig():
    def __init__(self, vocab_size=49408, hidden_size=512, intermediate_size=2048,
                 projection_dim=512, num_hidden_layers=12, num_attention_heads=8,
                 max_position_embeddings=77, hidden_act="gelu", layer_norm_eps=1e-5,
                 attention_dropout=0.0, ...):
        ...

class CLIPVisionConfig():
    def __init__(self, hidden_size=768, intermediate_size=3072, projection_dim=512,
                 num_hidden_layers=12, num_attention_heads=12, image_size=224,
                 patch_size=32, ...):
        ...

class CLIPConfig():
    def __init__(self, text_config=None, vision_config=None,
                 projection_dim=512, logit_scale_init_value=2.6592):
        ...
```

### 2.2 通用 Transformer 层（文本/视觉共享结构）

```python
class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        bsz, tgt_len, embed_dim = hidden_states.size()
        self.scale = 1.0 / math.sqrt(embed_dim)
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        # ... scaled dot product attention with optional causal mask
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask):
        _x = x
        x = self.layer_norm1(x)       # Pre-Norm
        x, _ = self.self_attn(x, attention_mask)
        x = _x + x                     # Residual
        _x = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = _x + x
        return x
```

### 2.3 文本编码器（Causal Mask → 取 EOS token）

> **关键洞察**：CLIP 文本编码器使用 **因果注意力掩码**（上三角 mask），取序列最后一个 token（EOS）的表示作为 pooled output。这与 BERT 取 `[CLS]` 不同。

```python
class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(config)  # token_embedding + position_embedding
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.eos_token_id = config.eos_token_id

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        # 因果注意力掩码
        causal_attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -torch.inf
        x = self.encoder(x, attention_mask)
        pooler_output = self.final_layer_norm(x[:, -1, :])  # 取最后一个 token
        return {'last_hidden_state': x, 'pooler_output': pooler_output}
```

### 2.4 视觉编码器（CLS token → Pooler）

> **关键洞察**：CLIP 视觉编码器与 ViT 类似，但用 `post_layernorm` 处理 `[CLS]` token 输出作为图像表示。

```python
class CLIPVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)  # patch_embedding + cls + position
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(hidden_states)
        pooled_output = self.post_layernorm(encoder_outputs[:, 0, :])  # CLS token
        return {'last_hidden_state': encoder_outputs, 'pooler_output': pooled_output}
```

### 2.5 CLIP 完整模型

```python
class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = CLIPTextTransformer(config.text_config)
        self.vision_model = CLIPVisionTransformer(config.vision_config)
        self.visual_projection = nn.Linear(vision_hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_hidden_size, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

    def forward(self, input_ids, pixel_values, attention_mask=None):
        image_embeds = self.visual_projection(self.vision_model(pixel_values)['pooler_output'])
        text_embeds = self.text_projection(self.text_model(input_ids)['pooler_output'])
        # L2 归一化
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # 余弦相似度 × 温度
        logit_scale = self.logit_scale.exp()
        logits_per_text = text_embeds @ image_embeds.t() * logit_scale
        loss = clip_loss(logits_per_text)
        return {'loss': loss, 'logits_per_image': logits_per_text.t(), ...}
```

### 2.6 Zero-Shot 推理

> **关键洞察**：Zero-Shot 时，将每个类别名构造成文本 prompt，分别编码后与图像表示做余弦相似度，`argmax` 即为预测类别。

```python
# 1 张图像 vs N 个类别 prompt
image_embeds = clip_model.visual_projection(vision_model(one_img))
text_embeds = clip_model.text_projection(text_model(prompt))  # [N, D]
# 归一化
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
logits = image_embeds @ text_embeds.t()
prediction = torch.argmax(logits, dim=-1)
```

---

## 3. CLIP Loss — InfoNCE 对比学习损失

### 伪代码（来自论文）

```python
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
I_f = image_encoder(I)                         # [n, d_i]
T_f = text_encoder(T)                          # [n, d_t]
I_e = l2_normalize(I_f @ W_i)                  # [n, d_e]
T_e = l2_normalize(T_f @ W_t)                  # [n, d_e]
logits = I_e @ T_e.T * exp(t)                  # [n, n]
labels = arange(n)
loss = (cross_entropy(logits, labels, axis=0)   # image → text
      + cross_entropy(logits, labels, axis=1))  # text → image
loss /= 2
```

### 完整 PyTorch 实现

> **关键洞察**：CLIP Loss 是 **对称的 InfoNCE**——对角线上是正样本对，每行/每列做一次交叉熵。温度参数 `t` 是可学习的。

```python
class CLIP(nn.Module):
    def __init__(self, d_i, d_t, d_e):
        super().__init__()
        self.W_i_e = nn.Linear(d_i, d_e, bias=False)
        self.W_t_e = nn.Linear(d_t, d_e, bias=False)
        self.temparture = nn.Parameter(torch.ones(1))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, I_f, T_f):
        n, _ = I_f.size()
        I_e = self.W_i_e(I_f)
        T_e = self.W_t_e(T_f)
        # L2 归一化
        I_e = I_e / I_e.norm(p=2, dim=-1, keepdim=True)
        T_e = T_e / T_e.norm(p=2, dim=-1, keepdim=True)
        # 相似度矩阵
        logits = I_e @ T_e.t() * torch.exp(self.temparture)
        labels = torch.arange(n)
        loss_i = self.loss_fn(logits, labels)          # image→text
        loss_t = self.loss_fn(logits.t(), labels)      # text→image
        loss = loss_i + loss_t
        return {'loss': loss, 'image_embedding': I_e, 'text_embedding': T_e, 'logits': logits}
```

### 训练 Pipeline

```python
# step 1: 准备图文对
I = torch.randn(batch_size, C, H, W)
T = torch.randint(0, vocab_size, (batch_size, seq_len))

# step 2: 编码
I_f = image_encoder(I)
T_f = text_encoder(T)

# step 3: 计算 CLIP Loss
output = clip(I_f, T_f)

# step 4: 反向传播
output['loss'].backward()
```

---

## 4. LLaVA — 多模态大语言模型

**核心思想**：冻结预训练的视觉编码器（CLIP ViT），通过一个可学习的 **Projection Layer** 将视觉特征对齐到 LLM 的 token embedding 空间，然后与文本 token 拼接送入 LLM decoder。

### 架构

```
Image → CLIP Vision Encoder → hidden_states[vision_feature_layer][:, 1:]  (去 CLS)
      → Multi-Modal Projector (MLP: Linear → GELU → Linear)
      → image_features [B, num_patches, text_hidden_size]

Text  → LLM Embedding → text_embeds [B, seq_len, text_hidden_size]

Merge: 在 <image> token 位置插入 image_features
      → [B, seq_len + num_patches - 1, hidden_size]
      → LLM Decoder → logits
```

### 4.1 LLaVA Config

```python
class LlavaConfig():
    def __init__(self, vision_config=None, text_config=None,
                 ignore_index=-100, image_token_index=32000,
                 projector_hidden_act="gelu",
                 vision_feature_select_strategy="default",  # "default" 去掉 CLS, "full" 保留
                 vision_feature_layer=-2):                   # 取倒数第二层
        ...
```

### 4.2 Multi-Modal Projector

> **关键洞察**：Projector 只有两个线性层 + GELU，参数量很小。它是连接视觉和语言的桥梁——阶段 1 只训练这个模块。

```python
class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.act = nn.functional.gelu
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
```

### 4.3 LLaVA Forward（核心逻辑）

> **关键洞察**：最复杂的部分是 `_merge_input_ids_with_image_features`——在文本序列中找到 `<image>` token 的位置，替换为视觉 patch embeddings，同时正确处理 attention_mask、position_ids 和 labels 的对齐。

```python
class LlavaForConditionalGeneration(nn.Module):
    def __init__(self, config, vision_model, language_model):
        super().__init__()
        self.vision_tower = vision_model
        self.language_model = language_model
        self.multi_modal_projector = LlavaMultiModalProjector(config)

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # 1. 提取视觉特征（取倒数第二层，去掉 CLS）
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_outputs.hidden_states[-2][:, 1:]

            # 2. 投影到文本空间
            image_features = self.multi_modal_projector(selected_image_feature)

            # 3. 在 <image> token 位置插入视觉 embeddings
            inputs_embeds, attention_mask, labels, position_ids = \
                self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels)

        # 4. 送入 LLM
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds)

        logits = outputs[0]
        # 计算 next-token prediction loss（shift 一位）
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        return (loss, logits)
```

### 4.4 训练策略

| 阶段 | 冻结模块 | 训练模块 | 数据 |
|------|---------|---------|------|
| 阶段 1：投影层预训练 | Vision Encoder + LLM | Projector only | CC3M 595K |
| 阶段 2：指令微调 | Vision Encoder | Projector + LLM | 150K 指令数据 |

### 4.5 LLaVA-Next 改进

- **更大图像输入**：支持 AnyRes（选择最佳分辨率 → 切割为多个子图 → 分别编码 → 拼接）
- **image_newline token**：在每行 patch 后插入换行 token，保留空间结构信息
- **MLP Projector**：与 LLaVA v1 相同的双层 MLP

```python
# AnyRes 核心：选择最佳预设分辨率
def select_best_resolution(original_size, possible_resolutions):
    """选择有效面积最大、浪费最少的分辨率"""
    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        effective_resolution = min(downscaled_w * downscaled_h, original_w * original_h)
        wasted_resolution = (width * height) - effective_resolution
        # 取 effective 最大、wasted 最小的
    return best_fit
```

### 4.6 Prompt 格式

```
USER: <image>\nWhat is shown in this image?\nASSISTANT:
```

`<image>` 是特殊 token，forward 时会被替换为 `num_patches` 个视觉 embedding。

---

## 关键洞察总结

| 模型 | 核心创新 | 一句话 |
|------|---------|--------|
| **ViT** | Patch → Token → Transformer | 图像就是 16×16 词的句子 |
| **CLIP** | 双塔 + InfoNCE 对比学习 | 图文在同一空间，余弦相似度即关联度 |
| **CLIP Loss** | 对称 CrossEntropy on similarity matrix | 对角线是正样本，batch 越大负样本越多 |
| **LLaVA** | Vision Encoder → Projector → LLM | 两层 MLP 搭桥，视觉 patch 变成 LLM 能理解的 token |
| **LLaVA-Next** | AnyRes + image_newline | 动态分辨率 + 保留空间结构 |
