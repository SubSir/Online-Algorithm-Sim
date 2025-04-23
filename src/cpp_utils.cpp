/*
    Implementation of the parser functions
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <map>

#include "cpp_utils.hpp"

const int max_requests =  10000;

std :: vector<Request> _parse_trace(std::string trace_file) {
    std :: ifstream trace(trace_file);

    if (!trace.is_open()) {
        std :: cerr << "Error opening file: " << trace_file << std :: endl;
        exit(1);
    }

    std :: vector<Request> requests;

    Request req;

    char buffer[24];

    // Read the requests until the end of the file
    while (trace.read(buffer, sizeof(buffer)) and requests.size() < max_requests) {
        req.timestamp = *reinterpret_cast<uint32_t*>(buffer);
        req.obj_id = *reinterpret_cast<uint64_t*>(buffer + 4);
        req.obj_size = *reinterpret_cast<uint32_t*>(buffer + 12);
        req.next_access_vtime = *reinterpret_cast<int64_t*>(buffer + 16);
        requests.push_back(req);
        if (requests.size() % 1000000 == 0) {
            std:: cout << "Finish parsing " << requests.size() << "requests" << std:: endl;
        }
    }

    if (trace.eof() or requests.size() == max_requests) {
        std:: cout << "Finish parsing " << trace_file << std:: endl;
        return std :: move(requests);
    }
    
    std :: cerr << "Error reading file: " << trace_file << std :: endl;
    exit(1);
}

namespace py = pybind11;

PYBIND11_MODULE(cpp_utils, m) {
    // Bind the Request struct
    py::class_<Request>(m, "Request")
        .def_readwrite("timestamp", &Request::timestamp)
        .def_readwrite("obj_id", &Request::obj_id)
        .def_readwrite("obj_size", &Request::obj_size)
        .def_readwrite("next_access_vtime", &Request::next_access_vtime);

    // Bind the parser function
    m.doc() = "Module for parsing requests from a trace file";
    m.def("_parse_trace", &_parse_trace, "Parse requests from a trace file");
}