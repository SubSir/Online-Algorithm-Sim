/*
    Implementation of the parser functions
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

#include "cpp_utils.hpp"

std :: vector<Request> _parse_trace(std::string trace_file) {
    std :: ifstream trace(trace_file);

    if (!trace.is_open()) {
        std :: cerr << "Error opening file: " << trace_file << std :: endl;
        exit(1);
    }

    std :: vector<Request> requests;

    Request req;

    // Read the requests until the end of the file
    while (trace.read((char*)&req, sizeof(Request))) {
        requests.push_back(req);
    }

    if (trace.eof()) {
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