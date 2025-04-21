/*
    Implementation of the LFU scheduler
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <queue>

#include "scheduler.hpp"

struct LFUObject {
    uint64_t obj_id;
    int64_t num_of_access;

    LFUObject(const uint64_t obj_id, const int64_t num_of_access) {
        this -> obj_id = obj_id;
        this -> num_of_access = num_of_access;
    }

    bool operator < (const LFUObject& other) const {
        return this -> num_of_access == other.num_of_access ? this -> obj_id < other.obj_id : this -> num_of_access < other.num_of_access;
    }
};

class LFUScheduler : public Scheduler {
    public:
        std :: set<LFUObject> cache;
        std :: set<LFUObject, std :: function<bool(const LFUObject&, const LFUObject&)>> cache_set;

        LFUScheduler(uint64_t cache_size) : Scheduler(cache_size), 
            cache_set([](const LFUObject& a, const LFUObject& b) {
                return a.obj_id < b.obj_id;
            }) {
            // We define two sets for efficiently
            // 1. Check if an object is in the cache
            // 2. Remove the object that has the longest time to be nextly accessed
            this -> cache = std :: set<LFUObject>();
        }

        Result run(std :: vector<Request>& requests) {
            // Initialize the cache misses
            int cnt = 0;
            auto result = Result(requests);
            
            uint32_t counter = 0;
            for (auto &request : requests) {
                if (cnt % 100000 == 0) {
                    std:: cout << "Running " << cnt << "-th request" << std::endl; 
                }
                cnt++;
                auto obj_id = request.obj_id;
                auto num_of_access = 1;

                counter++;

                // if (counter % 10000 == 0) {
                //     std :: cerr << "Obj: " << obj_id << " Timestamp: " << last_access << std :: endl;
                // }
                // unique_set.insert(obj_id);
                // auto print_cache = [&cache = this -> cache]() {
                //     for (auto &obj : cache) {
                //         std :: cerr << obj.obj_id << " ";
                //     }
                //     std :: cerr << std :: endl;
                // };

                // if (counter % 10000 == 0) {
                    // std :: cerr << "Obj: " << obj_id << " Timestamp: " << request.timestamp << std :: endl;
                    // print_cache();
                // }

                // first we check if the object is in the cache
                auto is_in_cache = [&cache_set = this -> cache_set](const uint64_t& obj_id) {
                    auto it = cache_set.lower_bound(LFUObject(obj_id, 0));
                    if (it == cache_set.end() || it -> obj_id != obj_id) {
                        return cache_set.end();
                    }
                    return it;
                };

                std :: set<LFUObject> :: iterator it;
                it = is_in_cache(obj_id);
                // std :: cerr << (it == this -> cache_set.end()) << std :: endl;
                if ((it = is_in_cache(obj_id)) == this -> cache_set.end()) {
                    // Not in the cache
                    result.cache_misses++;

                    // If the cache is full, remove the object that has the longest time to be nextly accessed
                    if (this -> cache_set.size() == this -> cache_size) {
                        auto evicted_obj = *this -> cache.begin();
                        this -> cache.erase(this -> cache.begin());
                        this -> cache_set.erase(evicted_obj);
                    }
                }
                else {
                    // In the cache
                    // We need to update the next access time of the object

                    // Remove the old object from the cache
                    auto obj = *it;
                    num_of_access = obj.num_of_access + 1;
                    this -> cache.erase(obj);
                    this -> cache_set.erase(it);
                }

                // Insert the object into the cache
                auto obj = LFUObject(obj_id, num_of_access);
                this -> cache.insert(obj);
                this -> cache_set.insert(obj);

                // print_cache();
            }

            return result;
        }
};

namespace py = pybind11;

PYBIND11_MODULE(lfu, m) {
    // Bind the LFU scheduler
    py::class_<LFUScheduler>(m, "LFU")
        .def(py::init<const uint64_t>())
        .def("run", &LFUScheduler::run);
}