
/*
    Implementation of the OPT scheduler
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <queue>

#include "scheduler.hpp"

struct OPTObject {
    uint64_t obj_id;
    int64_t next_access;

    OPTObject(const uint64_t obj_id, const int64_t next_access) {
        this -> obj_id = obj_id;
        this -> next_access = next_access;
    }

    bool operator < (const OPTObject& other) const {
        return this -> next_access > other.next_access;
    }
};

class OPTScheduler : public Scheduler {
    public:
        std :: set<OPTObject> cache;
        std::set<OPTObject, std::function<bool(const OPTObject&, const OPTObject&)>> cache_set;

        OPTScheduler(uint64_t cache_size) : Scheduler(cache_size), 
            cache_set([](const OPTObject& a, const OPTObject& b) {
                return a.obj_id < b.obj_id;
            }) {
            // We define two sets for efficiently
            // 1. Check if an object is in the cache
            // 2. Remove the object that has the longest time to be nextly accessed
            this -> cache = std :: set<OPTObject>();
        }

        Result run(std :: vector<Request>& requests) {
            // Initialize the cache misses
            auto result = Result(requests);
            int cnt = 0;
            
            // uint32_t counter = 0;
            for (auto &request : requests) {
                if (cnt % 100000 == 0) {
                    std:: cout << "Running " << cnt << "-th request" << std::endl; 
                }
                cnt++;
                auto obj_id = request.obj_id;
                auto next_access = request.next_access_vtime;

                // std :: cout << "Requesting object " << obj_id << " at " << next_access << std :: endl;
                // counter++;
                // if (counter == 10) {
                //     break;
                // }

                // first we check if the object is in the cache
                auto is_in_cache = [&cache_set = this -> cache_set](const uint64_t& obj_id) {
                    auto it = cache_set.lower_bound(OPTObject(obj_id, 0));
                    if (it == cache_set.end() || it -> obj_id != obj_id) {
                        return cache_set.end();
                    }
                    return it;
                };

                std :: set<OPTObject> :: iterator it;
                if ((it = is_in_cache(obj_id)) == this -> cache_set.end()) {
                    // Not in the cache
                    result.cache_misses++;

                    // If the cache is full, remove the object that has the longest time to be nextly accessed
                    if (this -> cache.size() == this -> cache_size) {
                        auto evicted_obj = *this -> cache.begin();
                        // std :: cerr << "Evicting object with next access time " << evicted_obj.next_access << std :: endl;
                        this -> cache.erase(this -> cache.begin());
                        this -> cache_set.erase(evicted_obj);
                    }
                }
                else {
                    // In the cache
                    // We need to update the next access time of the object

                    // Remove the old object from the cache
                    auto obj = *it;
                    this -> cache.erase(obj);
                    this -> cache_set.erase(it);
                }

                // Insert the object into the cache
                auto obj = OPTObject(obj_id, next_access);
                // if (next_access != -1) {
                //     assert (obj_id == requests[next_access-1].obj_id);
                // }
                this -> cache.insert(obj);
                this -> cache_set.insert(obj);
            }

            return result;
        }
};

namespace py = pybind11;

PYBIND11_MODULE(opt, m) {
    // Bind the OPT scheduler
    py::class_<OPTScheduler>(m, "OPT")
        .def(py::init<const uint64_t>())
        .def("run", &OPTScheduler::run);
}