import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", dest="source_file",
                    help="Source file to scan for vulnerabilities")
parser.add_argument("-m", "--model", dest="model_path",
                    help="Path to the model")
parser.add_argument("-t", "--temp", dest="temp",
                    help="The temperature to use for sampling")
parser.add_argument("-c", "--context", dest="n_ctx",
                    help="Text context window size")
parser.add_argument("-max", "--max_tokens", dest="max_tokens",
                    help="The maximum number of tokens to generate")
parser.add_argument("-th", "--threads", dest="threads",
                    help="Number of threads to use for generation")
parser.add_argument("-g", "--gpu_layers", dest="gpu_layers",
                    help="Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded")
parser.add_argument("-v", "--verbose", dest="verbose",
                    help="Print verbose output to stderr")
args = parser.parse_args()
print(args)