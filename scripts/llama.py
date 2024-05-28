from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import sys
import argparse
import inspect
from os import path, makedirs
import json
import re

#_MODEL_PATH = "../models/dolphin-2.7-mixtral-8x7b.Q3_K_M.gguf"
_MODEL_PATH = "../models/dolphin-2.9-llama3-8b-256k.Q8_0.gguf"


def get_num_tokens(self, text: str) -> int:
    tokenized_text = self.client.tokenize(text.encode("utf-8"))
    return len(tokenized_text)


def inference(source_code, model_path=_MODEL_PATH, temperature=0.25, n_ctx=4096, max_tokens=100, n_threads=8, n_gpu_layers=80, verbose=True):
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        top_p=1,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        callback_manager=callback_manager,
        verbose=verbose
    )

    system_msg = '''
    You are an assistant that reads in C and C++ source files to determine if there are any security vulnerabilities or potential 
    vulnerabilities. These includes vulnerabilities such as buffer overflow, integer overflow, integer underflows, null pointer 
    dereferences, user after-free, pointer initialization, incorrect type conversion, format string, and any other security vulnerabilities.
    You must respond using JSON format. Responses should be organized in 4 name-value pairs. The first name-value pair is called 
    "vulnerability-type" and includes a short description of said vulnerability, short as possible and preferably in one or two words 
    The second name-value pair is "line-number" and should include the line number or range of line numbers of where the vulnerabliity occurs. 
    The third name-value pair is "summary" and include a short description, one hundred words or less, summarizing the vulnerability and what 
    makes it one. The fourth name-value pair is "cwe" or common weakness enumerations and should include the CWE number that the vulnerability 
    mostly closely associates with, otherwise if unable to identify, then label as "undefined". Repeat if multiple potential security 
    vulnerabilities are detected. 
    '''

    prompt = "\nFind any and all potential security vulnerabilities in the following code.\n" + source_code

    inf_msg = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {0}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """.format(system_msg, prompt)
    print(inf_msg)

    output = llm.invoke(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="source_file",
                        help="Source file to scan for vulnerabilities")
    parser.add_argument("-m", "--model", dest="model_path", default=inspect.signature(inference).parameters['model_path'].default,
                        help="Path to the model")
    parser.add_argument("-t", "--temp", dest="temp", default=inspect.signature(inference).parameters['temperature'].default,
                        help="The temperature to use for sampling")
    parser.add_argument("-c", "--context", dest="n_ctx", default=inspect.signature(inference).parameters['n_ctx'].default,
                        help="Text context window size")
    parser.add_argument("-max", "--max_tokens", dest="max_tokens", default=inspect.signature(inference).parameters['max_tokens'].default,
                        help="The maximum number of tokens to generate")
    parser.add_argument("-th", "--threads", dest="threads", default=inspect.signature(inference).parameters['n_threads'].default,
                        help="Number of threads to use for generation")
    parser.add_argument("-g", "--gpu_layers", dest="gpu_layers", default=inspect.signature(inference).parameters['n_gpu_layers'].default,
                        help="Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded")
    parser.add_argument("-v", "--verbose", dest="verbose", default=inspect.signature(inference).parameters['verbose'].default,
                        help="Print verbose output to stderr")
    args = parser.parse_args()

    if not (args.source_file):
        parser.error("Must specify a source file.")

    # Reads file and add line numbers to each line
    source_code = ""
    with open(args.source_file, 'r') as my_file:
        for i, line in enumerate(my_file):
            source_code = source_code + ('%04d %s'%(i+1, line))

    inference(source_code, 
              args.model_path,
              args.temp, 
              args.n_ctx, 
              args.max_tokens, 
              args.threads, 
              args.gpu_layers, 
              args.verbose)
