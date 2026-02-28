# Model Setup

Foxtral uses two GGUF models that run locally via llama.cpp and voxtral.cpp.
Model files are not checked into the repository — download them with the commands below.

## Prerequisites

Install the Hugging Face CLI:

```bash
brew install pipx
pipx install 'huggingface_hub[cli]'
```

## 1. Voxtral Realtime 4B (Speech-to-Text)

Download the [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF) Q4_0 quantized model:

```bash
hf download andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF \
  --include "Q4_0.gguf" \
  --local-dir ./voxtral.cpp/models/voxtral/
```

| | |
|---|---|
| **Expected path** | `voxtral.cpp/models/voxtral/Q4_0.gguf` |
| **Size** | ~2.5 GB |

## 2. Ministral 8B (LLM)

Download the Ministral-3-8B-Instruct-2512 Q4_K_M quantized model:

```bash
hf download bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF \
  --include "mistralai_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf" \
  --local-dir ./llama.cpp/models/
```

| | |
|---|---|
| **Expected path** | `llama.cpp/models/mistralai_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` |
| **Size** | ~4.8 GB |

## Memory Requirements

Both models run simultaneously in unified memory on macOS (Apple Silicon).

| Component | Memory |
|-----------|--------|
| Voxtral Realtime 4B Q4_0 | ~2.5 GB |
| Ministral 8B Q4_K_M | ~5 GB |
| Context / overhead | ~1 GB |
| **Total** | **~8.5 GB** |

A machine with 16 GB of RAM can run both models comfortably.
