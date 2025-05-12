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
    LRUObject(uint64_t obj_id, int64_t last_access) : obj_id(obj_id), last_access(last_access) {}
};

// 用于 LRU 淘汰
struct LRUCompare {
    bool operator()(const LRUObject* a, const LRUObject* b) const {
        // last_access小的排前面（优先淘汰）
        if (a->last_access != b->last_access) return a->last_access < b->last_access;
        return a->obj_id < b->obj_id;   // 排除 last_access 相同的情况
    }
};

class LRUScheduler : public Scheduler {
public:
    std::set<LRUObject*, LRUCompare> cache; // LRU队列
    std::unordered_map<uint64_t, LRUObject*> cache_map; // id 查找表

    LRUScheduler(uint64_t cache_size) : Scheduler(cache_size) {}

    ~LRUScheduler() {
        for (auto p : cache) delete p;
    }

    Result run(std::vector<Request>& requests) {
        auto result = Result(requests);
        uint64_t idx = 0;

        for (auto& request : requests) {
            uint64_t obj_id = request.obj_id;
            int64_t last_access = idx++;

            // 已经在cache中
            auto it = cache_map.find(obj_id);
            if (it != cache_map.end()) {
                // 先移除旧对象
                cache.erase(it->second);
                delete it->second; // 删掉老对象防内存泄漏
                cache_map.erase(it);
            } else {
                // miss
                result.cache_misses++;
                if (cache.size() == this->cache_size) {
                    // 淘汰最旧
                    auto oldest = *cache.begin();
                    cache_map.erase(oldest->obj_id);
                    cache.erase(cache.begin());
                    delete oldest;
                }
            }

            // 插入新对象
            auto obj = new LRUObject(obj_id, last_access);
            cache.insert(obj);
            cache_map[obj_id] = obj;
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