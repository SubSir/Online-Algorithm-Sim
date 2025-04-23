import os
import sys
import argparse
from sklearn.model_selection import train_test_split
import src.data_downloader as data_downloader
from src.input_processor import InputProcessor
from src.utils import MyNamespace
from src.scheduler import (
    OPT,
    FIFO,
    LIFO,
    LRU,
    LFU,
    Marking,
    Belady,
    LSTMScheduler,
    LSTM_Cache,
)


def read_config(config_file):
    with open(config_file, "r") as stream:
        try:
            import yaml

            return MyNamespace(**yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./config/config.yaml")
    parser.add_argument("--algorithm", type=str, default="default")
    parser.add_argument("--cache_size", type=int, default=-1)
    parse_args = parser.parse_args()
    args = read_config(parse_args.config_file)

    if parse_args.algorithm != "default":
        args.algorithm = parse_args.algorithm
    if parse_args.cache_size != -1:
        args.cache_size = parse_args.cache_size

    print("Practical Paging Algorithm Simulator")
    print(args)

    traces_list = data_downloader.download_data(args)

    print("\n------------------------------------")

    input_processor = InputProcessor()
    output_file = os.path.join(
        args.output_dir, args.algorithm + f"_{args.cache_size}" + ".txt"
    )

    if not os.path.exists(output_file):
        open(output_file, "w").close()
    else:
        os.remove(output_file)
        open(output_file, "w").close()

    for trace_file in traces_list:
        print(f"Processing {trace_file}")
        requests = input_processor.process_input(trace_file)

        if (args.algorithm == "LSTM"):
            train_r, test_r = train_test_split(requests, test_size=0.25, shuffle=False)
            lstm = LSTM_Cache(args.cache_size, (1 << 20) - 1, N=30)
            lstm.train(train_r)
            predict = lstm.predict(test_r)
            prediction = [False if (i < 0.5) else True for i in predict]

            # belady = Belady(args.cache_size)
            # belady.initial(test_r)
            # belady.resize(len(test_r))
            # prediction = belady.result
            
            scheduler = LSTMScheduler(args.cache_size)
            result = scheduler.run(test_r, prediction)

            opt_result = OPT(args.cache_size).run(test_r)
            lru_result = LRU(args.cache_size).run(test_r)

            print("OPT result ")
            print(f"Total number of requests: {opt_result.total_requests}")
            print(f"Total number of unique pages: {opt_result.unique_pages}")
            print(f"Total number of cache misses: {opt_result.cache_misses}")

            print("LRU result ")
            print(f"Total number of requests: {lru_result.total_requests}")
            print(f"Total number of unique pages: {lru_result.unique_pages}")
            print(f"Total number of cache misses: {lru_result.cache_misses}")

        else:
            scheduler = eval(args.algorithm)(args.cache_size)
            result = scheduler.run(requests)

        with open(output_file, "a") as f:
            print(f"Total number of requests: {result.total_requests}")
            print(f"Total number of unique pages: {result.unique_pages}")
            print(f"Total number of cache misses: {result.cache_misses}")
            f.write(f"Trace file: {trace_file}\n")
            f.write(f"Total number of requests: {result.total_requests}\n")
            f.write(f"Total number of unique pages: {result.unique_pages}\n")
            f.write(f"Total number of cache misses: {result.cache_misses}\n")


if __name__ == "__main__":
    main()
