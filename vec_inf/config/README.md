# Available Models
More profiling metrics coming soon!

## Text Generation Models

### [Cohere for AI: Command R](https://huggingface.co/collections/CohereForAI/c4ai-command-r-plus-660ec4c34f7a69c50ce7f7b9)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`c4ai-command-r-plus`](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | 8x a40 (2 nodes, 4 a40/node) | 412 tokens/s | 541 tokens/s |
| [`c4ai-command-r-plus-08-2024`](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) | 8x a40 (2 nodes, 4 a40/node) | - tokens/s | - tokens/s |
| [`c4ai-command-r-08-2024`](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) | 8x a40 (2 nodes, 4 a40/node) | - tokens/s | - tokens/s |

### [Code Llama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`CodeLlama-7b-hf`](https://huggingface.co/meta-llama/CodeLlama-7b-hf) | 1x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-7b-Instruct-hf`](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf) | 1x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-13b-hf`](https://huggingface.co/meta-llama/CodeLlama-13b-hf) | 1x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-13b-Instruct-hf`](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf) | 1x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-34b-hf`](https://huggingface.co/meta-llama/CodeLlama-34b-hf) | 2x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-34b-Instruct-hf`](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf) | 2x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-70b-hf`](https://huggingface.co/meta-llama/CodeLlama-70b-hf) | 4x a40 | - tokens/s | - tokens/s |
| [`CodeLlama-70b-Instruct-hf`](https://huggingface.co/meta-llama/CodeLlama-70b-Instruct-hf) | 4x a40 | - tokens/s | - tokens/s |

### [Databricks: DBRX](https://huggingface.co/collections/databricks/dbrx-6601c0852a0cdd3c59f71962)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`dbrx-instruct`](https://huggingface.co/databricks/dbrx-instruct) | 8x a40 (2 nodes, 4 a40/node) | 107 tokens/s | 904 tokens/s |

### [Google: Gemma 2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`gemma-2-9b`](https://huggingface.co/google/gemma-2-9b) | 1x a40 | - tokens/s | - tokens/s |
| [`gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | 1x a40 | - tokens/s | - tokens/s |
| [`gemma-2-27b`](https://huggingface.co/google/gemma-2-27b) | 2x a40 | - tokens/s | - tokens/s |
| [`gemma-2-27b-it`](https://huggingface.co/google/gemma-2-27b-it) | 2x a40 | - tokens/s | - tokens/s |

### [Meta: Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b)

| Variant | Suggested resource allocation |
|:----------:|:----------:|
| [`Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1x a40 |
| [`Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 1x a40 |
| [`Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 1x a40 |
| [`Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 1x a40 |
| [`Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf) | 4x a40 |
| [`Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 4x a40 |

### [Meta: Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 1x a40 | 222 tokens/s | 1811 tokens/s |
| [`Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 1x a40 | 371 tokens/s | 1990 tokens/s |
| [`Meta-Llama-3-70B`](https://huggingface.co/meta-llama/Meta-Llama-3-70B) | 4x a40 | 81 tokens/s | 618 tokens/s |
| [`Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) | 4x a40 | 301 tokens/s | 660 tokens/s |

### [Meta: Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) | 1x a40 | - tokens/s | - tokens/s |
| [`Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Meta-Llama-3.1-70B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B) | 4x a40 | - tokens/s | - tokens/s |
| [`Meta-Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) | 4x a40 | - tokens/s | - tokens/s |
| [`Meta-Llama-3.1-405B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) | 32x a40 (8 nodes, 4 a40/node) | - tokens/s | - tokens/s |

### [Meta: Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | 1x a40 | - tokens/s | - tokens/s |
| [`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) | 1x a40 | - tokens/s | - tokens/s |
| [`Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 1x a40 | - tokens/s | - tokens/s |

### [Mistral AI: Mistral](https://huggingface.co/mistralai)

| Variant (Mistral) | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 1x a40 | - tokens/s | - tokens/s|
| [`Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | 1x a40 | - tokens/s | - tokens/s|
| [`Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-v0.2) | 1x a40 | - tokens/s | - tokens/s|
| [`Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3) | 1x a40 | - tokens/s | - tokens/s |
| [`Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 1x a40 | - tokens/s | - tokens/s|
| [`Mistral-Large-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | 8x a40 (2 nodes, 4 a40/node) | - tokens/s | - tokens/s|
| [`Mistral-Large-Instruct-2411`](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411) | 8x a40 (2 nodes, 4 a40/node) | - tokens/s | - tokens/s|

### [Mistral AI: Mixtral](https://huggingface.co/mistralai)

| Variant (Mixtral) | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 4x a40 | 222 tokens/s | 1543 tokens/s |
| [`Mixtral-8x22B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) | 8x a40 (2 nodes, 4 a40/node) | 145 tokens/s | 827 tokens/s|
| [`Mixtral-8x22B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | 8x a40 (2 nodes, 4 a40/node) | 95 tokens/s | 803 tokens/s|

### [Microsoft: Phi 3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Phi-3-medium-128k-instruct`](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) | 2x a40 | - tokens/s | - tokens/s |

### [Aaditya Ura: Llama3-OpenBioLLM](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Llama3-OpenBioLLM-70B`](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B) | 4x a40 | - tokens/s | - tokens/s |

### [Nvidia: Llama-3.1-Nemotron](https://huggingface.co/collections/nvidia/llama-31-nemotron-70b-670e93cd366feea16abc13d8)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Llama-3.1-Nemotron-70B-Instruct-HF`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) | 4x a40 | - tokens/s | - tokens/s |

### [Qwen: Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 2x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | 4x a40 | - tokens/s | - tokens/s |

### [Qwen: Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) | 1x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct) | 4x a40 | - tokens/s | - tokens/s |

### [Qwen: Qwen2.5-Coder](https://huggingface.co/collections/Qwen/qwen25-coder-66eaa22e6f99801bf65b0c2f)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 1x a40 | - tokens/s | - tokens/s |

### [Qwen: QwQ](https://huggingface.co/collections/Qwen/qwq-674762b79b75eac01735070a)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`QwQ-32B-Preview`](https://huggingface.co/Qwen/QwQ-32B-Preview) | 2x a40 | - tokens/s | - tokens/s |

### [DeepSeek-R1: Distilled Models](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 1x a40 | - tokens/s | - tokens/s |
| [`DeepSeek-R1-Distill-Llama-70B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | 4x a40 | - tokens/s | - tokens/s |
| [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1x a40 | - tokens/s | - tokens/s |
| [`DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 1x a40 | - tokens/s | - tokens/s |
| [`DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | 2x a40 | - tokens/s | - tokens/s |
| [`DeepSeek-R1-Distill-Qwen-32B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 4x a40 | - tokens/s | - tokens/s |


## Vision Language Models

### [allenai: Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Molmo-7B-D-0924`](https://huggingface.co/allenai/Molmo-7B-D-0924) | 1x a40 | - tokens/s | - tokens/s |


### [LLaVa-1.5](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | 1x a40 | - tokens/s | - tokens/s |
| [`llava-1.5-13b-hf`](https://huggingface.co/llava-hf/llava-1.5-13b-hf) | 1x a40 | - tokens/s | - tokens/s |

### [LLaVa-NeXT](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`llava-v1.6-mistral-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | 1x a40 | - tokens/s | - tokens/s |
| [`llava-v1.6-34b-hf`](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) | 2x a40 | - tokens/s | - tokens/s |

### [Microsoft: Phi 3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Phi-3-vision-128k-instruct`](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | 2x a40 | - tokens/s | - tokens/s |
| [`Phi-3.5-vision-instruct`](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | 2x a40 | - tokens/s | - tokens/s |

### [Meta: Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Llama-3.2-11B-Vision`](https://huggingface.co/meta-llama/Llama-3.2-1B) | 2x a40 | - tokens/s | - tokens/s |
| [`Llama-3.2-11B-Vision-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 2x a40 | - tokens/s | - tokens/s |
| [`Llama-3.2-90B-Vision`](https://huggingface.co/meta-llama/Llama-3.2-3B) | 8x a40 (2 nodes, 4 a40/node) | - tokens/s | - tokens/s |
| [`Llama-3.2-90B-Vision-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 8x a40 (2 nodes, 4 a40/node) | - tokens/s | - tokens/s |

**NOTE**: `MllamaForConditionalGeneration` currently doesn't support pipeline parallelsim, to save memory, maximum number of requests is reduced and enforce eager mode is on.

### [Mistral: Pixtral](https://huggingface.co/mistralai)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Pixtral-12B-2409`](https://huggingface.co/mistralai/Pixtral-12B-2409) | 1x a40 | - tokens/s | - tokens/s |

### [OpenGVLab: InternVL2.5](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`InternVL2_5-8B`](https://huggingface.co/OpenGVLab/InternVL2_5-8B) | 1x a40 | - tokens/s | - tokens/s |
| [`InternVL2_5-26B`](https://huggingface.co/OpenGVLab/InternVL2_5-26B) | 2x a40 | - tokens/s | - tokens/s |
| [`InternVL2_5-38B`](https://huggingface.co/OpenGVLab/InternVL2_5-38B) | 4x a40 | - tokens/s | - tokens/s |

### [THUDM: GLM-4](https://huggingface.co/collections/THUDM/glm-4-665fcf188c414b03c2f7e3b7)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`glm-4v-9b`](https://huggingface.co/THUDM/glm-4v-9b) | 1x a40 | - tokens/s | - tokens/s |

### [DeepSeek: DeepSeek-VL2](https://huggingface.co/collections/deepseek-ai/deepseek-vl2-675c22accc456d3beb4613ab)
| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`deepseek-vl2`](https://huggingface.co/deepseek-ai/deepseek-vl2) | 2x a40 | - tokens/s | - tokens/s |
| [`deepseek-vl2-small`](https://huggingface.co/deepseek-ai/deepseek-vl2-small) | 1x a40 | - tokens/s | - tokens/s |


## Text Embedding Models

### [Liang Wang: e5](https://huggingface.co/intfloat)
| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`e5-mistral-7b-instruct`](https://huggingface.co/intfloat/e5-mistral-7b-instruct) | 1x a40 | - tokens/s | - tokens/s |

### [BAAI: bge](https://huggingface.co/BAAI)
| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | 1x A40 | - tokens/s | - tokens/s |

### [Sentence Transformers: MiniLM](https://huggingface.co/sentence-transformers)
| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 1x A40 | - tokens/s | - tokens/s |



## Reward Modeling Models

### [Qwen: Qwen2.5-Math](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e)

| Variant | Suggested resource allocation | Avg prompt throughput | Avg generation throughput |
|:----------:|:----------:|:----------:|:----------:|
| [`Qwen2.5-Math-RM-72B`](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B) | 4x a40 | - tokens/s | - tokens/s |
| [`Qwen2.5-Math-PRM-7B`](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) | 1x a40 | - tokens/s | - tokens/s |
