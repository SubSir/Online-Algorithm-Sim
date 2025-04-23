
/*
    Implementation of the belady algorithm
*/
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <set>
#include <vector>
#include <queue>

#include "scheduler.hpp"

struct OPTObject {
    uint64_t obj_id;
    uint64_t next_access;
    uint64_t insert_time;

    OPTObject(const uint64_t obj_id, const uint64_t next_access, const uint64_t insert_time)
        : obj_id(obj_id), next_access(next_access), insert_time(insert_time) {}

    bool operator < (const OPTObject& other) const {
        // next_access 较大优先被淘汰
        if (next_access != other.next_access)
            return next_access > other.next_access;
        return obj_id < other.obj_id; // 保证唯一性
    }
};

struct OPTObject2 {
    uint64_t obj_id;
    bool prediction;
    uint64_t insert_time;

    OPTObject2(const uint64_t obj_id, const bool prediction, uint64_t insert_time)
        : obj_id(obj_id), prediction(prediction), insert_time(insert_time) {}

    bool operator < (const OPTObject2& other) const {
        if (prediction != other.prediction)
            return prediction > other.prediction;
        return insert_time < other.insert_time;
    }
};

class Belady {
public:
    uint64_t pos = 0;
    uint64_t cache_size;
    std::set<OPTObject> cache; // 按 next_access 降序排序
    std::unordered_map<uint64_t, std::set<OPTObject>::iterator> cache_map; // obj_id->迭代器
    std::vector<bool> result;

    Belady(const uint64_t cache_size)
        : cache_size(cache_size), result() {}

    void initial(std::vector<Request>& v) {
        result.resize(v.size());
        for (auto& request : v) {
            process(request);
        }
    }

    void droplet(Request& request) {
        process(request);
    }

    void resize(const uint64_t X_len) {
        result.resize(X_len);
    }

private:
    void process(Request& request) {
        uint64_t obj_id = request.obj_id;
        uint64_t next_access = request.next_access_vtime;

        auto it = cache_map.find(obj_id);
        if (it == cache_map.end()) {
            // 没有命中cache，考虑淘汰
            if (cache.size() == cache_size) {
                auto evicted = *cache.begin();
                cache_map.erase(evicted.obj_id);
                cache.erase(cache.begin());
                // 标记被淘汰位置
                if (evicted.insert_time >= result.size())
                    result.resize(evicted.insert_time + 1);
                result[evicted.insert_time] = true;
            }
        } else {
            // 命中cache，需要先删除旧条目
            cache.erase(it->second);
            cache_map.erase(it);
        }
        // 更新到cache和cache_map
        auto ins = cache.insert(OPTObject(obj_id, next_access, pos)).first;
        cache_map[obj_id] = ins;
        pos++;
    }
};
class LSTMScheduler {
    public:
        uint64_t pos = 0;
        uint64_t cache_size;
        std::set<OPTObject2> cache; // 按 prediction、insert_time 排序
        std::unordered_map<uint64_t, std::set<OPTObject2>::iterator> cache_map;
    
        LSTMScheduler(uint64_t cache_size) : cache_size(cache_size) {}
    
        Result run(std::vector<Request>& requests, std::vector<bool>& predictions) {
            auto result = Result(requests);
    
            for (const auto& request : requests) {
                uint64_t obj_id = request.obj_id;
                bool pred = predictions[pos];
    
                auto it = cache_map.find(obj_id);
    
                if (it == cache_map.end()) {
                    // Not in cache
                    result.cache_misses++;
                    // 缓存满了，淘汰优先策略元素
                    if (cache.size() == cache_size) {
                        auto evict_it = cache.begin();
                        cache_map.erase(evict_it->obj_id);
                        cache.erase(evict_it);
                    }
                } else {
                    // In cache，需更新 prediction, insert_time
                    cache.erase(it->second); // 先移除旧元素
                }
                // 插入新元素，并同步 map
                auto ins = cache.insert(OPTObject2(obj_id, pred, pos)).first;
                cache_map[obj_id] = ins;
                pos++;
            }
    
            return result;
        }
    };


namespace py = pybind11;

PYBIND11_MODULE(belady, m) {
    // Bind the Belady class
    py::class_<Belady>(m, "Belady")
        .def(py::init<const uint64_t>())
        .def("initial", &Belady::initial)
        .def("droplet", &Belady::droplet)
        .def("resize", &Belady::resize)
        .def_readwrite("result", &Belady::result);

    // Bind the LSTMScheduler class
    py::class_<LSTMScheduler>(m, "LSTMScheduler")
        .def(py::init<const uint64_t>())
        .def("run", &LSTMScheduler::run);   
}