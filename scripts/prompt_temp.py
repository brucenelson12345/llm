from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
import argparse
import inspect
import math
from pathlib import Path
import os
import json
import re


_PROMPT_PATH = "../prompts/svd_system_04.txt"
#_MODEL_PATH = "../models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"
_MODEL_PATH = "../models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"

_TEMPERATURE = 0.1
_N_CTX = 4000
_N_BATCH = 512
_MAX_TOKENS = 1024
_N_THREADS = 8
_N_GPU_LAYERS = 41
_VERBOSE = True

_NON_VUL = ['UNDEFINED', 'NONE', 'BENIGN', 'HARMLESS']

def fstr(template):
    return eval(f"f'''{template}'''")

class SVDScanner(object):

    def __init__(self, options):
        self.options = options

        self.llm = self.build_llm(
            model_path=options.model_path,
            temperature=options.temp,
            n_ctx=options.n_ctx,
            n_batch=options.n_batch,
            max_tokens=options.max_tokens,
            n_threads=options.threads,
            n_gpu_layers=options.n_gpu_layers,
            grammar_path=options.grammar_path,
            verbose=options.verbose
        )

    def build_llm(self, model_path=_MODEL_PATH, temperature=_TEMPERATURE, n_ctx=_N_CTX, n_batch=_N_BATCH, max_tokens=_MAX_TOKENS, 
                  n_threads=_N_THREADS, n_gpu_layers=_N_GPU_LAYERS, grammar_path=None, verbose=_VERBOSE):
        """
        Builds LlamaCpp llm model
        :return:
        """
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        return LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        top_p=1,
        n_ctx=n_ctx,
        n_batch=_N_BATCH,
        max_tokens=max_tokens,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        callback_manager=callback_manager,
        grammar_path=grammar_path,
        verbose=verbose
        )

    def get_num_tokens(self, text: str) -> int:
        """
        Retrieves estimated number of tokens from text
        :return:
        """
        tokenized_text = self.llm.client.tokenize(text.encode("utf-8"))
        return len(tokenized_text)

    def prompt_splitter(self, inf_msg, system_msg, source):
        """
        Counts number of tokens from source input
        Limits prompt sizes to max context window size
        Splits prompts into chunks if exceeds context window size
        """
        inf_tokens = scanner.get_num_tokens(system_msg)
        prompt_tokens = scanner.get_num_tokens(source)
        # Account for 10% overhead in token limit
        limit_tokens = math.ceil((inf_tokens) * 1.10)
        prompt_limit = _N_CTX - limit_tokens
        token_splitter = TokenTextSplitter(chunk_size=prompt_limit, chunk_overlap=0)
        prompts = token_splitter.split_text(source)
        
        return prompts

    def inference(self, inf_msg):
        """
        Runs inference and outputs detected svds
        :return:
        """
        output = self.llm.invoke(inf_msg)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="source_file",
                        help="Source file to scan for vulnerabilities")
    parser.add_argument("-s", "--system_file", dest="system_file", default=_PROMPT_PATH,
                        help="Template file to match with model")
    parser.add_argument("-m", "--model", dest="model_path", default=inspect.signature(SVDScanner.build_llm).parameters['model_path'].default,
                        help="Path to the model")
    parser.add_argument("-t", "--temp", dest="temp", default=inspect.signature(SVDScanner.build_llm).parameters['temperature'].default,
                        help="The temperature to use for sampling")
    parser.add_argument("-c", "--context", dest="n_ctx", default=inspect.signature(SVDScanner.build_llm).parameters['n_ctx'].default,
                        help="Text context window size")
    parser.add_argument("-b", "--batch", dest="n_batch", default=inspect.signature(SVDScanner.build_llm).parameters['n_batch'].default,
                        help="Prompt processing maximum batch size")
    parser.add_argument("-max", "--max_tokens", dest="max_tokens", default=inspect.signature(SVDScanner.build_llm).parameters['max_tokens'].default,
                        help="The maximum number of tokens to generate")
    parser.add_argument("-th", "--threads", dest="threads", default=inspect.signature(SVDScanner.build_llm).parameters['n_threads'].default,
                        help="Number of threads to use for generation")
    parser.add_argument("-l", "--layers", dest="n_gpu_layers", default=inspect.signature(SVDScanner.build_llm).parameters['n_gpu_layers'].default,
                        help="Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded")
    parser.add_argument("-g", "--grammar_path", dest="grammar_path", default=None,
                        help="Formal grammar for constraining model outputs")
    parser.add_argument("-v", "--verbose", dest="verbose", default=inspect.signature(SVDScanner.build_llm).parameters['verbose'].default,
                        help="Print verbose output to stderr")
    args = parser.parse_args()

    if not (args.source_file):
        parser.error("Must specify a source file.")

    # Reads source file and adds line numbers to each line
    source = ""
    with open(args.source_file, 'r') as my_file:
        for i, line in enumerate(my_file):
            source = source + ('%04d %s'%(i+1, line))
    
    system_f = open(args.system_file, 'r')
    system_msg = system_f.read()
    inf_msg = "<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant".format(system_msg, source)

    scanner = SVDScanner(args)

    prompts = scanner.prompt_splitter(inf_msg, system_msg, source)

    svd_results = []
    for prompt in prompts:
        inf_msg = "<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant".format(system_msg, prompt)
        svd_result = scanner.inference(inf_msg)
        svd_results.append(svd_result)

    with open("../logs/" + Path(args.source_file).stem + ".txt", 'w') as log_f:
        for svd_result in svd_results:
            log_f.write(svd_result)

    #### TODO: LLM Json Parsing unreliable for now ####
    # # removes random characters the llm may add before the brackets
    # svd_results = re.sub(r'^.*?{', '{', svd_results)

    # svd_dict = json.loads(svd_results)
    # json_object = json.dumps(svd_dict, indent=4)

    # json_file = "../logs/" + Path(args.source_file).stem + ".json"
    # with open(json_file, "w") as outfile:
    #     outfile.write(json_object)
