
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

struct OPTObject2 {
    uint64_t obj_id;
    bool prediction;
    int64_t insert_time;

    OPTObject2(const uint64_t obj_id, const bool prediction, int64_t insert_time) {
        this -> obj_id = obj_id;
        this -> prediction = prediction;
        this -> insert_time = insert_time;
    }

    bool operator < (const OPTObject2& other) const {
        if (this -> prediction != other . prediction) {
            return this -> prediction;
        }
        return this -> insert_time < other . insert_time;
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
        std :: set<OPTObject2> cache;
        std::set<OPTObject2, std::function<bool(const OPTObject2&, const OPTObject2&)>> cache_set;


        LSTMScheduler(uint64_t cache_size) : cache_size(cache_size), 
        cache_set([](const OPTObject2& a, const OPTObject2& b) {
            return a.obj_id < b.obj_id;
        }) {
            this -> cache = std :: set<OPTObject2>();
        }

        Result run(std :: vector<Request>& requests, std:: vector<bool>& predictions) {
            // Initialize the cache misses
            auto result = Result(requests);
            
            // uint32_t counter = 0;
            int64_t cnt = -1;
            for (auto &request : requests) {
                cnt ++;
                auto obj_id = request.obj_id;
                // first we check if the object is in the cache

                auto is_in_cache = [&cache_set = this -> cache_set](const uint64_t& obj_id) {
                    auto it = cache_set.lower_bound(OPTObject2(obj_id, 0, 0));
                    if (it == cache_set.end() || it -> obj_id != obj_id) {
                        return cache_set.end();
                    }
                    return it;
                };
                std :: set<OPTObject2> :: iterator it;
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
                } else {
                    // In the cache
                    // We need to update the next access time of the object

                    // Remove the old object from the cache
                    auto obj = *it;
                    this -> cache.erase(obj);
                    this -> cache_set.erase(it);
                }
                auto obj = OPTObject2(obj_id, predictions[cnt], cnt);
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