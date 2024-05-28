from llama_cpp import Llama

llm = Llama(
    #model_path="../models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
    model_path="/home/pride/work/llm/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
    # temperature=0.75,
    # top_p=1,
    n_ctx=32768,
    n_threads=8,
    n_gpu_layers=-1,
    # CallbackManager=callback_manager,
    verbose=False,
)

system_message = "Answer the question being asked."
prompt = "What is a memcpy?"

output = llm(
  prompt="<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant".format(system_message,prompt), # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
  echo=True        # Whether to echo the prompt
)

print(output)