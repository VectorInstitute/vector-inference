# Model Weights Tracking

This document tracks all model weights available in the `/model-weights` directory on Killarney cluster and indicates which ones have existing configurations in the cached model config (`/model-weights/vec-inf-shared/models.yaml`). By default, `vec-inf` would use the cached model config. To request new model weights to be downloaded or model configuration to be added, please open an issue for "Model request".

**NOTE**: The [`models.yaml`](./vec_inf/config/models.yaml) file in the package is not always up to date with the latest cached model config on Killarney cluster, new model config would be added to the cached model config. `models.yaml` would be updated to reflect the cached model config when a new version of the package is released.

## Legend
- ✅ **Configured**: Model has a complete configuration in `models.yaml`
- ❌ **Not Configured**: Model exists in `/model-weights` but lacks configuration

---

## Text Generation Models (LLM)

### Cohere for AI: Command R
| Model | Configuration |
|:------|:-------------|
| `c4ai-command-r-plus-08-2024` | ✅ |
| `c4ai-command-r-08-2024` | ✅ |

### Code Llama
| Model | Configuration |
|:------|:-------------|
| `CodeLlama-7b-hf` | ✅ |
| `CodeLlama-7b-Instruct-hf` | ✅ |
| `CodeLlama-13b-hf` | ✅ |
| `CodeLlama-13b-Instruct-hf` | ✅ |
| `CodeLlama-34b-hf` | ✅ |
| `CodeLlama-34b-Instruct-hf` | ✅ |
| `CodeLlama-70b-hf` | ✅ |
| `CodeLlama-70b-Instruct-hf` | ✅ |
| `CodeLlama-7b-Python-hf` | ❌ |
| `CodeLlama-13b-Python-hf` | ❌ |
| `CodeLlama-70b-Python-hf` | ❌ |

### Google: Gemma
| Model | Configuration |
|:------|:-------------|
| `gemma-2b` | ❌ |
| `gemma-2b-it` | ❌ |
| `gemma-7b` | ❌ |
| `gemma-7b-it` | ❌ |
| `gemma-2-2b-it` | ✅ |
| `gemma-2-9b` | ✅ |
| `gemma-2-9b-it` | ✅ |
| `gemma-2-27b` | ✅ |
| `gemma-2-27b-it` | ✅ |
| `gemma-3-1b-it` | ❌ |
| `gemma-3-4b-it` | ❌ |
| `gemma-3-12b-it` | ❌ |
| `gemma-3-27b-it` | ❌ |

### Meta: Llama 2
| Model | Configuration |
|:------|:-------------|
| `Llama-2-7b-hf` | ✅ |
| `Llama-2-7b-chat-hf` | ✅ |
| `Llama-2-13b-hf` | ✅ |
| `Llama-2-13b-chat-hf` | ✅ |
| `Llama-2-70b-hf` | ✅ |
| `Llama-2-70b-chat-hf` | ✅ |

### Meta: Llama 3
| Model | Configuration |
|:------|:-------------|
| `Meta-Llama-3-8B` | ✅ |
| `Meta-Llama-3-8B-Instruct` | ✅ |
| `Meta-Llama-3-70B` | ✅ |
| `Meta-Llama-3-70B-Instruct` | ✅ |

### Meta: Llama 3.1
| Model | Configuration |
|:------|:-------------|
| `Meta-Llama-3.1-8B` | ✅ |
| `Meta-Llama-3.1-8B-Instruct` | ✅ |
| `Meta-Llama-3.1-70B` | ✅ |
| `Meta-Llama-3.1-70B-Instruct` | ✅ |
| `Meta-Llama-3.1-405B-Instruct` | ✅ |

### Meta: Llama 3.2
| Model | Configuration |
|:------|:-------------|
| `Llama-3.2-1B` | ✅ |
| `Llama-3.2-1B-Instruct` | ✅ |
| `Llama-3.2-3B` | ✅ |
| `Llama-3.2-3B-Instruct` | ✅ |

### Meta: Llama 3.3
| Model | Configuration |
|:------|:-------------|
| `Llama-3.3-70B-Instruct` | ✅ |

### Meta: Llama 4
| Model | Configuration |
|:------|:-------------|
| `Llama-4-Scout-17B-16E-Instruct` | ❌ |
| `Llama-4-Maverick-17B-128E-Instruct` | ❌ |

### Mistral AI: Mistral
| Model | Configuration |
|:------|:-------------|
| `Mistral-7B-v0.3` | ✅ |
| `Mistral-7B-Instruct-v0.1` | ✅ |
| `Mistral-7B-Instruct-v0.2` | ✅ |
| `Mistral-7B-Instruct-v0.3` | ✅ |
| `Mistral-Large-Instruct-2407` | ✅ |
| `Mistral-Large-Instruct-2411` | ✅ |

### Mistral AI: Mixtral
| Model | Configuration |
|:------|:-------------|
| `Mixtral-8x7B-Instruct-v0.1` | ✅ |
| `Mixtral-8x22B-v0.1` | ✅ |
| `Mixtral-8x22B-Instruct-v0.1` | ✅ |

### Microsoft: Phi
| Model | Configuration |
|:------|:-------------|
| `Phi-3-medium-128k-instruct` | ✅ |
| `phi-4` | ❌ |

### Nvidia: Llama-3.1-Nemotron
| Model | Configuration |
|:------|:-------------|
| `Llama-3.1-Nemotron-70B-Instruct-HF` | ✅ |

### Qwen: Qwen2.5
| Model | Configuration |
|:------|:-------------|
| `Qwen2.5-0.5B-Instruct` | ✅ |
| `Qwen2.5-1.5B-Instruct` | ✅ |
| `Qwen2.5-3B` | ❌ |
| `Qwen2.5-3B-Instruct` | ✅ |
| `Qwen2.5-7B-Instruct` | ✅ |
| `Qwen2.5-14B-Instruct` | ✅ |
| `Qwen2.5-32B-Instruct` | ✅ |
| `Qwen2.5-72B-Instruct` | ✅ |

### Qwen: Qwen2.5-Math
| Model | Configuration |
|:------|:-------------|
| `Qwen2.5-Math-1.5B-Instruct` | ✅ |
| `Qwen2.5-Math-7B` | ❌ |
| `Qwen2.5-Math-7B-Instruct` | ✅ |
| `Qwen2.5-Math-72B-Instruct` | ✅ |

### Qwen: Qwen2.5-Coder
| Model | Configuration |
|:------|:-------------|
| `Qwen2.5-Coder-3B-Instruct` | ✅ |
| `Qwen2.5-Coder-7B-Instruct` | ✅ |

### Qwen: QwQ
| Model | Configuration |
|:------|:-------------|
| `QwQ-32B` | ✅ |

### Qwen: Qwen2
| Model | Configuration |
|:------|:-------------|
| `Qwen2-1.5B-Instruct` | ❌ |
| `Qwen2-7B-Instruct` | ❌ |
| `Qwen2-Math-1.5B-Instruct` | ❌ |
| `Qwen2-Math-7B-Instruct` | ❌ |
| `Qwen2-Math-72B` | ❌ |
| `Qwen2-Math-72B-Instruct` | ❌ |
| `Qwen2-VL-7B-Instruct` | ❌ |

### Qwen: Qwen2.5-VL
| Model | Configuration |
|:------|:-------------|
| `Qwen2.5-VL-3B-Instruct` | ❌ |
| `Qwen2.5-VL-7B-Instruct` | ✅ |

### Qwen: Qwen3
| Model | Configuration |
|:------|:-------------|
| `Qwen3-14B` | ✅ |
| `Qwen3-8B` | ✅ |
| `Qwen3-32B` | ✅ |
| `Qwen3-235B-A22B` | ❌ |
| `Qwen3-Embedding-8B` | ❌ |

### DeepSeek: DeepSeek-R1
| Model | Configuration |
|:------|:-------------|
| `DeepSeek-R1-Distill-Llama-8B` | ✅ |
| `DeepSeek-R1-Distill-Llama-70B` | ✅ |
| `DeepSeek-R1-Distill-Qwen-1.5B` | ✅ |
| `DeepSeek-R1-Distill-Qwen-7B` | ✅ |
| `DeepSeek-R1-Distill-Qwen-14B` | ✅ |
| `DeepSeek-R1-Distill-Qwen-32B` | ✅ |

### DeepSeek: Other Models
| Model | Configuration |
|:------|:-------------|
| `DeepSeek-Coder-V2-Lite-Instruct` | ❌ |
| `deepseek-math-7b-instruct` | ❌ |

### OpenAI: GPT-OSS
| Model | Configuration |
|:------|:-------------|
| `gpt-oss-120b` | ✅ |
| `gpt-oss-20b` | ✅ |


#### AI21: Jamba
| Model | Configuration |
|:------|:-------------|
| `AI21-Jamba-1.5-Mini` | ❌ |

#### Cohere for AI: Aya
| Model | Configuration |
|:------|:-------------|
| `aya-expanse-32b` | ✅ |

#### OpenAI: GPT-2
| Model | Configuration |
|:------|:-------------|
| `gpt2-large` | ❌ |
| `gpt2-xl` | ❌ |

#### InternLM: InternLM2
| Model | Configuration |
|:------|:-------------|
| `internlm2-math-plus-7b` | ❌ |

#### Janus
| Model | Configuration |
|:------|:-------------|
| `Janus-Pro-7B` | ❌ |

#### Moonshot AI: Kimi
| Model | Configuration |
|:------|:-------------|
| `Kimi-K2-Instruct` | ❌ |

#### Mistral AI: Ministral
| Model | Configuration |
|:------|:-------------|
| `Ministral-8B-Instruct-2410` | ❌ |

#### AI2: OLMo
| Model | Configuration |
|:------|:-------------|
| `OLMo-1B-hf` | ❌ |
| `OLMo-7B-hf` | ❌ |
| `OLMo-7B-SFT` | ❌ |

#### EleutherAI: Pythia
| Model | Configuration |
|:------|:-------------|
| `pythia` | ❌ |

#### Qwen: Qwen1.5
| Model | Configuration |
|:------|:-------------|
| `Qwen1.5-72B-Chat` | ❌ |

#### ReasonFlux
| Model | Configuration |
|:------|:-------------|
| `ReasonFlux-PRM-7B` | ❌ |

#### LMSYS: Vicuna
| Model | Configuration |
|:------|:-------------|
| `vicuna-13b-v1.5` | ❌ |

#### Google: T5 (Encoder-Decoder Models)
**Note**: These are encoder-decoder (T5) models, not decoder-only LLMs.
| Model | Configuration |
|:------|:-------------|
| `t5-large-lm-adapt` | ❌ |
| `t5-xl-lm-adapt` | ❌ |
| `mt5-xl-lm-adapt` | ❌ |

---

## Vision Language Models (VLM)

### LLaVa
| Model | Configuration |
|:------|:-------------|
| `llava-1.5-7b-hf` | ✅ |
| `llava-1.5-13b-hf` | ✅ |
| `llava-v1.6-mistral-7b-hf` | ✅ |
| `llava-v1.6-34b-hf` | ✅ |
| `llava-med-v1.5-mistral-7b` | ❌ |

### Microsoft: Phi 3 Vision
| Model | Configuration |
|:------|:-------------|
| `Phi-3-vision-128k-instruct` | ✅ |
| `Phi-3.5-vision-instruct` | ✅ |

### Meta: Llama 3.2 Vision
| Model | Configuration |
|:------|:-------------|
| `Llama-3.2-11B-Vision` | ❌ |
| `Llama-3.2-11B-Vision-Instruct` | ✅ | (SGLang only)
| `Llama-3.2-90B-Vision` | ❌ |
| `Llama-3.2-90B-Vision-Instruct` | ✅ | (SGLang only)

### Mistral: Pixtral
| Model | Configuration |
|:------|:-------------|
| `Pixtral-12B-2409` | ✅ |

### OpenGVLab: InternVL2.5
| Model | Configuration |
|:------|:-------------|
| `InternVL2_5-8B` | ✅ |
| `InternVL2_5-26B` | ✅ |
| `InternVL2_5-38B` | ✅ |

### THUDM: GLM-4
| Model | Configuration |
|:------|:-------------|
| `glm-4v-9b` | ✅ |

### DeepSeek: DeepSeek-VL2
| Model | Configuration |
|:------|:-------------|
| `deepseek-vl2` | ✅ |
| `deepseek-vl2-small` | ✅ |

### Google: MedGemma
| Model | Configuration |
|:------|:-------------|
| `medgemma-4b-it` | ✅ |
| `medgemma-27b-it` | ✅ |
| `medgemma-27b-text-it` | ❌ |

### Other VLM Models
| Model | Configuration |
|:------|:-------------|
| `instructblip-vicuna-7b` | ❌ |
| `MiniCPM-Llama3-V-2_5` | ❌ |
| `Molmo-7B-D-0924` | ✅ |

---

## Text Embedding Models

### Liang Wang: e5
| Model | Configuration |
|:------|:-------------|
| `e5-mistral-7b-instruct` | ✅ |

### BAAI: bge
| Model | Configuration |
|:------|:-------------|
| `bge-base-en-v1.5` | ✅ |
| `bge-m3` | ❌ |
| `bge-multilingual-gemma2` | ❌ |

### Sentence Transformers: MiniLM
| Model | Configuration |
|:------|:-------------|
| `all-MiniLM-L6-v2` | ✅ |

### Other Embedding Models
| Model | Configuration |
|:------|:-------------|
| `data2vec` | ❌ |
| `gte-modernbert-base` | ❌ |
| `gte-Qwen2-7B-instruct` | ❌ |
| `KaLM-Embedding-Gemma3-12B-2511` | ❌ |
| `llama-embed-nemotron-8b` | ❌ |
| `m2-bert-80M-32k-retrieval` | ❌ |
| `m2-bert-80M-8k-retrieval` | ❌ |

---

## Reward Modeling Models

### Qwen: Qwen2.5-Math
| Model | Configuration |
|:------|:-------------|
| `Qwen2.5-Math-RM-72B` | ✅ |
| `Qwen2.5-Math-PRM-7B` | ✅ |

---

## Vision Models

### CLIP
| Model | Configuration |
|:------|:-------------|
| `clip-vit-base-patch16` | ❌ |
| `clip-vit-large-patch14-336` | ❌ |

### Stable Diffusion
| Model | Configuration |
|:------|:-------------|
| `sd-v1-4-full-ema` | ❌ |
| `stable-diffusion-v1-4` | ❌ |

---
