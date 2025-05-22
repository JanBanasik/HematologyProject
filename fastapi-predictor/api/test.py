from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "speakleash/Bielik-4.5B-v3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

inputs = tokenizer("Witaj, jak się dziś czujesz?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
