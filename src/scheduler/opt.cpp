
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
    uint64_t next_access;

    OPTObject(const uint64_t obj_id, const uint64_t next_access) 
        : obj_id(obj_id), next_access(next_access) {}

    bool operator<(const OPTObject& other) const {
        return next_access > other.next_access; // Belady 优先淘汰 future max
    }
};

class OPTScheduler : public Scheduler {
public:
    std::set<OPTObject> cache;
    std::unordered_map<uint64_t, std::set<OPTObject>::iterator> cache_map;

    OPTScheduler(uint64_t cache_size) : Scheduler(cache_size) {}

    Result run(std::vector<Request>& requests) {
        auto result = Result(requests);
        for (auto& request : requests) {
            uint64_t obj_id = request.obj_id;
            uint64_t next_access = request.next_access_vtime;
            bool insert = true;

            if (cache_map.find(obj_id) == cache_map.end()) {
                // Miss
                result.cache_misses++;
                if (cache.size() == this->cache_size) {
                    auto evict_it = cache.begin();
                    if (evict_it->next_access >= next_access) {
                        cache_map.erase(evict_it->obj_id);
                        cache.erase(evict_it);
                    } else {
                        insert = false;
                    }
                }
            } else {
                // Hit, update
                auto hit_it = cache_map[obj_id];
                cache.erase(hit_it);
            }
            // Insert/update
            if (insert) {
                auto ins_it = cache.insert(OPTObject(obj_id, next_access)).first;
                cache_map[obj_id] = ins_it;
            }
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