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

## 🎯 Vision

> Make LLM inference fast, transparent, and controllable.
