from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('../models/Phi-mini-MoE-instruct', trust_remote_code=True)
# .save_pretrained writes tokenizer.model alongside the other files
tok.save_pretrained('../models/Phi-mini-MoE-instruct')
print('Done')
