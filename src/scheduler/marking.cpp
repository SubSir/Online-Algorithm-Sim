/*
    Implementation of the Marking algorithm
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <deque>
#include <ctime>

#include "scheduler.hpp"

template < typename T >
class UnmarkedSet {
    /*
        A helper class to store the unmarked objects, which efficiently
        1. randomly evict an object
        2. check if an object is in the unmarked set
    */
    private:
        std :: set<T> unmarked_;
        std :: deque<T> unmarked_pool_;

    public:
        UnmarkedSet() {
            this -> unmarked_ = std :: set<T>();
            this -> unmarked_pool_ = std :: deque<T>();
        }

        auto size() {
            return this -> unmarked_.size();
        }

        void insert(T obj) {
            this -> unmarked_.insert(obj);
            this -> unmarked_pool_.push_back(obj);
        }

        void erase(T obj) {
            // Here we lazily remove the object from the unmarked pool
            this -> unmarked_.erase(obj);
        }

        const auto find(T obj) {
            return this -> unmarked_.find(obj);
        }

        const auto end() {
            return this -> unmarked_.end();
        }

        T randomly_evict() {
            while (true) {
                auto idx = rand() % this -> unmarked_pool_.size();
                auto obj = this -> unmarked_pool_[idx];
                if (this -> unmarked_.find(obj) != this -> unmarked_.end()) {
                    return obj;
                }
                else {
                    this -> unmarked_pool_.erase(this -> unmarked_pool_.begin() + idx);
                }
            }

            throw std :: runtime_error("No object to evict");
        }

        std :: set<T> unmark_all(std :: set<T> marked) {
            std :: swap(this -> unmarked_, marked);
            this -> unmarked_pool_ = std :: deque<T>(this -> unmarked_.begin(), this -> unmarked_.end());
            return marked;
        }
};

class MarkingScheduler : public Scheduler {
    public:
        std :: set<uint64_t> marked;
        UnmarkedSet<uint64_t> unmarked;
        std :: set<uint64_t> cache_set;

        MarkingScheduler(uint64_t cache_size) : Scheduler(cache_size) {
            // We define one two sets and one defined helper set to efficiently
            // 1. Check if an object is in the cache
            // 2. Mark the object that is accessed
            // 3. Randomly evict the object that is unmarked
            this -> marked = std :: set<uint64_t>();
            this -> unmarked = UnmarkedSet<uint64_t>();
            this -> cache_set = std :: set<uint64_t>();
        }

        Result run(std :: vector<Request>& requests) {
            srand(time(0));
            // Initialize the cache misses
            auto result = Result(requests);
            
            for (auto &request : requests) {
                auto obj_id = request.obj_id;

                // first we check if the object is in the cache
                if (this -> cache_set.find(obj_id) == this -> cache_set.end()) {
                    // Not in the cache
                    result.cache_misses++;

                    // If the cache is full, remove the object that has the longest time to be nextly accessed
                    if (this -> cache_set.size() == this -> cache_size) {
                        // Randomly evict the object that is unmarked
                        auto obj_to_evict = this -> unmarked.randomly_evict();
                        this -> cache_set.erase(obj_to_evict);
                        this -> unmarked.erase(obj_to_evict);
                    }

                    // Mark the new object
                    this -> marked.insert(obj_id);
                    this -> cache_set.insert(obj_id);
                }
                else {
                    // If the object is unmarked, then mark it
                    auto it = this -> unmarked.find(obj_id);
                    if (it != this -> unmarked.end()) {
                        // Mark the object, removing it from the unmarked list
                        this -> unmarked.erase(obj_id);

                        // Add the object to the marked list
                        this -> marked.insert(obj_id);
                    }
                }

                // Unmark all the objects if there are no unmarked objects
                if (this -> marked.size() == this -> cache_size) {
                    this -> marked = this -> unmarked.unmark_all(std :: move(this -> marked));
                }
            }

            return result;
        }
};

namespace py = pybind11;

PYBIND11_MODULE(marking, m) {
    // Bind the LRU scheduler
    py::class_<MarkingScheduler>(m, "Marking")
        .def(py::init<const uint64_t>())
        .def("run", &MarkingScheduler::run);
}