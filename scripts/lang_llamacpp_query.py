from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import sys

#with open(sys.argv[1], 'r') as my_file:
with open("../datasets/VulDeePecker/CWE-119/source_files/102203/CWE415_Double_Free__no_assignment_op_01_bad.cpp", 'r') as my_file:
    source = ""
    for i, line in enumerate(my_file):
        source = source + ('%04d %s'%(i+1, line))

def get_num_tokens(self, text: str) -> int:
    tokenized_text = self.client.tokenize(text.encode("utf-8"))
    return len(tokenized_text)

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="../models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
    # temperature=0.75,
    # top_p=1,
    n_ctx=32768,
    max_tokens=512,
    n_threads=8,
    n_gpu_layers=-1,
    # CallbackManager=callback_manager,
    verbose=False
)

num_tokens: int = llm.get_num_tokens(source)
print("Num_tokens: {}".format(num_tokens))

cpp_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CPP, chunk_size=300, chunk_overlap=0
)

# prompt = """Answer the query based on the context below. Identify
# security vulnerabilities with line numbers of occurance and a summary
# of said vulnerability. If there no security vulnerabilities present,
# answer with "There are no vulnerabilities present".

# Context: You are an assistant that reads in C and C++ source files
# to determine if there are security vulnerabilities.If you locate a 
# security vulnerability, list the line number where it appears and sumarize
# what the issue is. If there is no security vulnerabilities, only there 
# isn't known security vulnerabilities detected.

# Source:
# {0}

# Answer: """.format(source)

# print(prompt)
# print(llm.invoke(prompt))
