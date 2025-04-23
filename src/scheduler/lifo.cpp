/*
    Implementation of the LIFO scheduler
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <deque>

#include "scheduler.hpp"

class LIFOScheduler : public Scheduler {
    public:
        std :: deque<uint64_t> cache;
        std :: set<uint64_t> cache_set;

        LIFOScheduler(uint64_t cache_size) : Scheduler(cache_size) {
            // We define a dequeue and a set here to efficiently
            // 1. Check if an object is in the cache
            // 2. Remove the oldest object from the cache
            this -> cache = std :: deque<uint64_t>();
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

                    // If the cache is full, remove the object that was inserted last
                    if (this -> cache.size() == this -> cache_size) {
                        auto oldest_obj = this -> cache.back();
                        this -> cache.pop_back();
                        this -> cache_set.erase(oldest_obj);
                    }

                    this -> cache.push_back(obj);
                    this -> cache_set.insert(obj);
                }
            }

            return result;
        }
};

namespace py = pybind11;

PYBIND11_MODULE(lifo, m) {
    // Bind the LIFO scheduler
    py::class_<LIFOScheduler>(m, "LIFO")
        .def(py::init<const uint64_t>())
        .def("run", &LIFOScheduler::run);
}