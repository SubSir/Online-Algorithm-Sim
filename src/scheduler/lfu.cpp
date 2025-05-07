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

    LFUObject(const uint64_t obj_id, const int64_t num_of_access)
        : obj_id(obj_id), num_of_access(num_of_access) {}

    bool operator < (const LFUObject& other) const {
        return (num_of_access == other.num_of_access) 
            ? obj_id < other.obj_id 
            : num_of_access < other.num_of_access;
    }
};

class LFUScheduler : public Scheduler {
public:
    std::set<LFUObject> cache; // 按频率排序的 set
    std::unordered_map<uint64_t, std::set<LFUObject>::iterator> cache_map; // obj_id => set迭代器

    LFUScheduler(uint64_t cache_size) : Scheduler(cache_size) {}

    Result run(std::vector<Request>& requests) {
        auto result = Result(requests);

        for (const auto& request : requests) {
            uint64_t obj_id = request.obj_id;

            auto it = cache_map.find(obj_id);

            int64_t updated_access = 1;
            if (it == cache_map.end()) {
                // 未命中
                result.cache_misses++;
                if (cache.size() == this->cache_size) {
                    // 淘汰频率最小的（set头）
                    auto evict_it = cache.begin();
                    cache_map.erase(evict_it->obj_id);
                    cache.erase(evict_it);
                }
            } else {
                // 命中
                updated_access = (*it->second).num_of_access + 1;
                cache.erase(it->second); // 移除旧对象
            }
            // 插入新对象（或更新访问次数），同步cache_map
            auto ins_it = cache.insert(LFUObject(obj_id, updated_access)).first;
            cache_map[obj_id] = ins_it;
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