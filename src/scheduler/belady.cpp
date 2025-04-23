
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
    int64_t next_access;
    int64_t insert_time;

    OPTObject(const uint64_t obj_id, const int64_t next_access, const int64_t insert_time) {
        this -> obj_id = obj_id;
        this -> next_access = next_access;
        this -> insert_time = insert_time;
    }

    bool operator < (const OPTObject& other) const {
        return this -> next_access > other.next_access;
    }
};

class Belady {
    public:
    std:: uint64_t cache_size;
    std :: set<OPTObject> cache;
    std::set<OPTObject, std::function<bool(const OPTObject&, const OPTObject&)>> cache_set;
    std::vector<bool> result;

    Belady(const uint64_t cache_size) :cache_size(cache_size), result(),
    cache_set([](const OPTObject& a, const OPTObject& b) {
        return a.obj_id < b.obj_id;
    }) {
        this->cache_size = cache_size;
    }

    void initial(std::vector<Request> &v) {
        int64_t v_time = -1;
        this -> result.resize(v.size());
        for (auto &request : v) {
            v_time++;
            auto obj_id = request.obj_id;
            auto next_access = request.next_access_vtime;
            auto is_in_cache = [&cache_set = this -> cache_set](const uint64_t& obj_id) {
                auto it = cache_set.lower_bound(OPTObject(obj_id, 0, 0));
                if (it == cache_set.end() || it -> obj_id != obj_id) {
                    return cache_set.end();
                }
                return it;
            };

            std :: set<OPTObject> :: iterator it;
            if ((it = is_in_cache(obj_id)) == this -> cache_set.end()) {
                if (this -> cache.size() == this -> cache_size) {
                    auto evicted_obj = *this -> cache.begin();
                    this -> cache.erase(this -> cache.begin());
                    this -> cache_set.erase(evicted_obj);
                    if (evicted_obj.insert_time >= this -> result.size()) {
                        this -> result.resize(evicted_obj.insert_time + 1);
                    }
                    this -> result[evicted_obj.insert_time] = true;
                }
            }
            else {
                auto obj = *it;
                this -> cache.erase(obj);
                this -> cache_set.erase(it);
            }

            auto obj = OPTObject(obj_id, next_access, v_time);
            this -> cache.insert(obj);
            this -> cache_set.insert(obj);
        }
    }

    void droplet(Request &request, int64_t v_time) {
        auto obj_id = request.obj_id;
        auto next_access = request.next_access_vtime;
        auto is_in_cache = [&cache_set = this -> cache_set](const uint64_t& obj_id) {
            auto it = cache_set.lower_bound(OPTObject(obj_id, 0, 0));
            if (it == cache_set.end() || it -> obj_id != obj_id) {
                return cache_set.end();
            }
            return it;
        };

        std :: set<OPTObject> :: iterator it;
        if ((it = is_in_cache(obj_id)) == this -> cache_set.end()) {
            if (this -> cache.size() == this -> cache_size) {
                auto evicted_obj = *this -> cache.begin();
                this -> cache.erase(this -> cache.begin());
                this -> cache_set.erase(evicted_obj);
                if (evicted_obj.insert_time >= this -> result.size()) {
                    this -> result.resize(evicted_obj.insert_time + 1);
                }
                this -> result[evicted_obj.insert_time] = true;
            }
        }
        else {
            auto obj = *it;
            this -> cache.erase(obj);
            this -> cache_set.erase(it);
        }

        auto obj = OPTObject(obj_id, next_access, v_time);
        this -> cache.insert(obj);
        this -> cache_set.insert(obj);
    }

    void resize(const uint64_t X_len) {
        this -> result.resize(X_len);
    }
};

class LSTMScheduler {
    public:
        uint64_t cache_size;
        std :: set<uint64_t> cache;

        LSTMScheduler(uint64_t cache_size) : cache_size(cache_size){
            this -> cache = std :: set<uint64_t>();
        }

        Result run(std :: vector<Request>& requests, std:: vector<bool>& predictions) {
            // Initialize the cache misses
            auto result = Result(requests);
            
            // uint32_t counter = 0;
            uint64_t cnt = -1;
            for (auto &request : requests) {
                cnt ++;
                auto obj_id = request.obj_id;
                // first we check if the object is in the cache

                auto it = this -> cache.find(obj_id);
                if (it == this -> cache.end()) {
                    // Not in the cache
                    result.cache_misses++;

                    if (!predictions[cnt]) {
                        // If the cache is full, remove the object that has the longest time to be nextly accessed
                        if (this -> cache.size() == this -> cache_size) {
                            this -> cache.erase(this -> cache.begin());
                        }
                        
                        // Insert the object into the cache
                        this -> cache.insert(obj_id);
                    }
                }
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