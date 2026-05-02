# 🐍 Kairu

![Language](https://img.shields.io/badge/language-python-yellow) ![License](https://img.shields.io/badge/license-busl--1.1-green) ![Status](https://img.shields.io/badge/status-active-brightgreen)

> ⚡ Real-time inference optimizer for LLMs — faster generation, smarter decoding, and live observability 📊✨

---

## 🌊 Meaning

**Kairu (流れる)** — *to flow, to stream.*

Inference should be fluid — not blocked by latency, inefficiency, or opaque performance.

---

## 🚀 What it is

Kairu wraps any HuggingFace model and adds:

* 🦅 Speculative decoding (EAGLE-style)
* ⏩ Dynamic early exit
* 💸 Token budget enforcement
* 📊 Live dashboard:

  * tokens/sec
  * latency
  * quality tradeoffs

---

## ❗ The problem

Speculative decoding works — but:

* locked inside heavy frameworks (vLLM, etc.)
* hard to experiment with
* no lightweight tooling
* no built-in observability

---

## 🧠 What you learn

* Speculative decoding internals (EAGLE, Medusa)
* KV cache management
* Streaming inference
* Performance optimization

---

## 🚀 Quick Start

```bash
pip install kairu
```

```python
from kairu import wrap_model

model = wrap_model("your-model")
model.generate("Hello world")
```

---

## 🌊 Streaming server (v0.4.0)

```bash
pip install "kairu[server]"
```

```python
import uvicorn
from kairu import create_app, ServerConfig

app = create_app(config=ServerConfig(model_name="kairu-mock"))
uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
curl -N -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "hello world", "max_tokens": 16}'
```

Each frame is OpenAI `chat.completion.chunk`-compatible with a `kairu` extension carrying per-token `latency_ms` and `tokens_per_s`. Stream terminates with `data: [DONE]`. Per-IP rate limit + per-request timeout are enforced at the boundary.

---

## 🧬 Model-aware optimization (v0.5.0)

```python
from kairu import (
    AutoProfile, CachedModel, DynamicGammaScheduler,
    LayerwiseEarlyExitDecoder, MockLayeredModel, MockModel, ModelWrapper,
)

# Auto-pick a decoder strategy + cache size for any model
profile = AutoProfile.recommend(MockModel(), name_hint="llama-3-8b", has_draft=True)
print(profile.strategy, profile.gamma, profile.rationale)

# Layerwise early exit on architectures that expose intermediate logits
decoder = LayerwiseEarlyExitDecoder(MockLayeredModel(num_layers=24), confidence_threshold=0.85)
tokens, stats = decoder.generate([1, 2, 3], max_new_tokens=16)
print(f"saved {stats['compute_saved']:.1%} of layer compute")

# Wrap any model with logits memoization + adaptive γ
wrapper = ModelWrapper(
    MockModel(), draft_model=MockModel(),
    cache_capacity=256, adaptive_gamma=True,
)
```

* **Layerwise early exit** — stops at the first transformer layer whose top-prob ≥ threshold
* **Logits cache (`CachedModel`)** — recycles target-model calls across speculative verification
* **Adaptive γ (`DynamicGammaScheduler`)** — AIMD control loop over speculative lookahead
* **`AutoProfile`** — picks `vanilla` / `early_exit` / `layered_early_exit` / `speculative` from model metadata

---

## 🎯 Vision

> Make LLM inference fast, transparent, and controllable.
