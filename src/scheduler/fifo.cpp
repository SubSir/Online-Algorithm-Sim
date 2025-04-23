/*
    Implementation of the FIFO scheduler
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <queue>

#include "scheduler.hpp"

class FIFOScheduler : public Scheduler {
    public:
        std :: queue<uint64_t> cache;
        std :: set<uint64_t> cache_set;

        FIFOScheduler(uint64_t cache_size) : Scheduler(cache_size) {
            // We define a queue and a set here to efficiently
            // 1. Check if an object is in the cache
            // 2. Remove the oldest object from the cache
            this -> cache = std :: queue<uint64_t>();
            this -> cache_set = std :: set<uint64_t>();
        }

        Result run(std :: vector<Request>& requests) {
            // Initialize the cache misses
            auto result = Result(requests);
            
            for (auto &request : requests) {
                auto obj = request.obj_id;

                // first we check if the object is in the cache
                if (this -> cache_set.find(obj) == this -> cache_set.end()) {
                    // Not in the cache
                    result.cache_misses++;

                    // If the cache is full, remove the oldest object
                    if (this -> cache.size() == this -> cache_size) {
                        auto oldest_obj = this -> cache.front();
                        this -> cache.pop();
                        this -> cache_set.erase(oldest_obj);
                    }

                    this -> cache.push(obj);
                    this -> cache_set.insert(obj);
                }
            }

            return result;
        }
};

namespace py = pybind11;

PYBIND11_MODULE(fifo, m) {
    // Bind the FIFO scheduler
    py::class_<FIFOScheduler>(m, "FIFO")
        .def(py::init<const uint64_t>())
        .def("run", &FIFOScheduler::run);
}