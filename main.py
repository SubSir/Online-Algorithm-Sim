import os
import sys
import numpy as np
import argparse
from sklearn.model_selection  import train_test_split
import src.data_downloader  as data_downloader 
from src.input_processor  import InputProcessor 
from src.utils  import MyNamespace
import matplotlib.pyplot as plt
from src.scheduler  import (
    OPT, FIFO, LIFO, LRU, LFU, Marking, Belady, LSTMScheduler,
    LSTM_Cache, SVM_Cache, ISVM_Cache, Perceptron_Cache,
    MultiVar_Cache, NaiveBayes_Cache, ISVM_Cache2
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
    parser.add_argument("--config_file",  type=str, default="./config/config.yaml") 
    parser.add_argument("--algorithm",  type=str, default="default")  # 支持多算法对比
    parser.add_argument("--cache_size",  type=int, default=-1)
    parse_args = parser.parse_args() 
 
    args = read_config(parse_args.config_file) 
    if parse_args.algorithm  != "default":
        args.algorithm  = parse_args.algorithm 
    if parse_args.cache_size  != -1:
        args.cache_size  = parse_args.cache_size 
 
    print("Practical Paging Algorithm Simulator")
    print(args)
 
    traces_list = data_downloader.download_data(args) 
    print("\n------------------------------------")
 
    input_processor = InputProcessor()
 
    # 输出文件名
    output_file = os.path.join( 
        args.output_dir, 
        f"{'_'.join(args.algorithm.split(','))}_{args.cache_size}.txt" 
    )
    if not os.path.exists(output_file): 
        open(output_file, "w").close()
    else:
        os.remove(output_file) 
        open(output_file, "w").close()
 
    for trace_file in traces_list:
        print(f"Processing {trace_file}")
        requests = input_processor.process_input(trace_file) 

        # 如果指定了多个算法，则依次运行
        algorithms = [alg.strip() for alg in args.algorithm.split(",")] 
        results = {}
 
        train_r, _ = train_test_split(requests, test_size=0.9, shuffle=False)
        test_r = requests
        for alg in algorithms:
            print(f"\nRunning algorithm: {alg}")

            belady = Belady(args.cache_size) 
            belady.initial(test_r) 
            belady.resize(len(test_r)) 
            if alg == "LSTM":
                # lstm = LSTM_Cache(args.cache_size,  1 << 10, N=30)
                # lstm.train(train_r) 
                # predict = lstm.predict(test_r) 
                # prediction = [False if i < 0.5 else True for i in predict]
                
                prediction = belady.result  
                print(np.mean(prediction))
                scheduler = LSTMScheduler(args.cache_size) 
                result = scheduler.run(test_r,  prediction)
                results[alg] = {
                    "total_requests": result.total_requests, 
                    "unique_pages": result.unique_pages, 
                    "cache_misses": result.cache_misses 
                }
 
            elif alg in ["SVM", "Perceptron", "MultiVar", "NaiveBayes"]:
                model = eval(alg + "_Cache")(args.cache_size) 
                model.train(train_r) 
                predict = model.predict_online(test_r) 
                scheduler = LSTMScheduler(args.cache_size) 
                result = scheduler.run(test_r,  predict)
                results[alg] = {
                    "total_requests": result.total_requests, 
                    "unique_pages": result.unique_pages, 
                    "cache_misses": result.cache_misses 
                }
 
            elif alg == "ISVM":
                isvm = ISVM_Cache(args.cache_size, upper_bound=30, k=30)
                # isvm.predict_online(train_r) 
                predict = isvm.predict_online(test_r) 
                # print(predict)
                scheduler = LSTMScheduler(args.cache_size) 
                result = scheduler.run(test_r,  predict)
                results[alg] = {
                    "total_requests": result.total_requests, 
                    "unique_pages": result.unique_pages, 
                    "cache_misses": result.cache_misses  
                }

            elif alg == "ISVM2":
                isvm = ISVM_Cache2(args.cache_size, upper_bound=30, k=30, N=1024,M=16)
                # isvm.predict_online(train_r) 
                predict = isvm.predict_online(test_r) 
                # print(predict)
                scheduler = LSTMScheduler(args.cache_size) 
                result = scheduler.run(test_r,  predict)
                results[alg] = {
                    "total_requests": result.total_requests, 
                    "unique_pages": result.unique_pages, 
                    "cache_misses": result.cache_misses  
                }
 
            else:
                scheduler = eval(alg)(args.cache_size) 
                result = scheduler.run(test_r) 
                results[alg] = {
                    "total_requests": result.total_requests, 
                    "unique_pages": result.unique_pages, 
                    "cache_misses": result.cache_misses  
                }

        # 任选一个用于展示请求数和唯一页数
        res = results[list(results.keys())[0]]
        print(f"Total  number of requests: {res['total_requests']}")
        print(f"Total  number of unique pages: {res['unique_pages']}")

        # 准备画图数据
        algs = list(results.keys())
        cache_misses = [results[alg]['cache_misses'] for alg in algs]

        # 画柱状图
        plt.figure(figsize=(8,5))
        plt.bar(algs, cache_misses, color='skyblue', width=0.5)
        plt.xlabel('Algorithm')
        plt.ylabel('Number of Cache Misses')
        plt.title(f'Cache Misses Comparison ({res["total_requests"]} total requests, {res["unique_pages"]} unique pages)')
        for i, miss in enumerate(cache_misses):
            plt.text(i, miss+1, str(miss), ha='center', va='bottom')
        plt.ylim(0, max(cache_misses)*1.2)
        plt.tight_layout()
        plt.savefig("cache_misses_bar.png")

        for alg, res in results.items(): 
            print(f"{alg}'s number of cache misses: {res['cache_misses']}")

        with open(output_file, "a") as f:
            f.write(f"Trace  file: {trace_file}\n")
            for alg, res in results.items(): 
                f.write(f"Algorithm:  {alg}\n")
                f.write(f"Total  number of requests: {res['total_requests']}\n")
                f.write(f"Total  number of unique pages: {res['unique_pages']}\n")
                f.write(f"Total  number of cache misses: {res['cache_misses']}\n")
            f.write("\n") 
 
if __name__ == "__main__":
    main()