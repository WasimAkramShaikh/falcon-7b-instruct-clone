# How to use me?

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon_tokenizer")


model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-micro-self-instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_auth_token="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",
)

model.bfloat16()
model.cuda()

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda:0")
sequences = pipeline(
    "What is your favourite dad joke?",
    max_length=200,
    do_sample=True,
    top_k=10,
    repetition_penalty=1.2,
    num_return_sequences=2,
    eos_token_id=tokenizer.eos_token_id,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```