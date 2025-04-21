/*
    Implementation of the LRU scheduler
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <queue>

#include "scheduler.hpp"

struct LRUObject {
    uint64_t obj_id;
    int64_t last_access;

    LRUObject(const uint64_t obj_id, const int64_t last_access) {
        this -> obj_id = obj_id;
        this -> last_access = last_access;
    }

    bool operator < (const LRUObject& other) const {
        return this -> last_access < other.last_access;
    }
};

class LRUScheduler : public Scheduler {
    public:
        std :: set<LRUObject> cache;
        std::set<LRUObject, std::function<bool(const LRUObject&, const LRUObject&)>> cache_set;

        LRUScheduler(uint64_t cache_size) : Scheduler(cache_size), 
            cache_set([](const LRUObject& a, const LRUObject& b) {
                return a.obj_id < b.obj_id;
            }) {
            // We define two sets for efficiently
            // 1. Check if an object is in the cache
            // 2. Remove the object that has the longest time to be nextly accessed
            this -> cache = std :: set<LRUObject>();
        }

        Result run(std :: vector<Request>& requests) {
            // Initialize the cache misses
            auto result = Result(requests);
            int cnt = 0;

            // auto unique_set = std :: set<uint64_t>();
            
            // uint32_t counter = 0;
            for (auto &request : requests) {
                // counter++;
                if (cnt % 100000 == 0) {
                    std:: cout << "Running " << cnt << "-th request" << std::endl; 
                }
                cnt++;
                
                auto obj_id = request.obj_id;
                auto last_access = request.timestamp;


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

                // print_cache();

                // first we check if the object is in the cache
                auto is_in_cache = [&cache_set = this -> cache_set](const uint64_t& obj_id) {
                    auto it = cache_set.lower_bound(LRUObject(obj_id, 0));
                    if (it == cache_set.end() || it -> obj_id != obj_id) {
                        return cache_set.end();
                    }
                    return it;
                };

                std :: set<LRUObject> :: iterator it;
                if ((it = is_in_cache(obj_id)) == this -> cache_set.end()) {
                    assert (*it.obj_id != obj_id);
                    // Not in the cache
                    result.cache_misses++;

                    // If the cache is full, remove the object that has the longest time to be nextly accessed
                    if (this -> cache.size() == this -> cache_size) {
                        auto evicted_obj = *this -> cache.begin();
                        this -> cache.erase(evicted_obj);
                        this -> cache_set.erase(evicted_obj);
                    }
                }
                else {
                    // In the cache
                    // We need to update the next access time of the object

                    // Remove the old object from the cache
                    auto obj = *it;
                    this -> cache.erase(obj);
                    this -> cache_set.erase(obj);
                }

                // Insert the object into the cache
                auto obj = LRUObject(obj_id, last_access);
                this -> cache.insert(obj);
                this -> cache_set.insert(obj);

                // print_cache();
                // if (counter % 10000 == 0) {
                //     std :: cerr << "Cache misses: " << result.cache_misses << " Unique objects: " << unique_set.size() << std :: endl;
                // }
            }

            return result;
        }
};

namespace py = pybind11;

PYBIND11_MODULE(lru, m) {
    // Bind the LRU scheduler
    py::class_<LRUScheduler>(m, "LRU")
        .def(py::init<const uint64_t>())
        .def("run", &LRUScheduler::run);
}