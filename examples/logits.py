from vllm import LLM, SamplingParams
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from functools import partial
import time

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


logits = []
def logit_cacher(module, args, output, pointer):
    pointer.append(output)

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096)
llm = LLM(model="/model-weights/Llama-2-7b-chat-hf/")

for n, m in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_modules():
    if isinstance(m, LogitsProcessor):
        forward_hook = m.register_forward_hook(partial(logit_cacher, pointer=logits))

start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()
diff = int(end - start)
total_toks = 0
for output in outputs:
    total_toks += len(output.outputs[0].token_ids)
toks_per_sec = total_toks / diff
print(f"Tokens per second: {toks_per_sec}")

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")