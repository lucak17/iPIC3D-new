#pragma once

#include <unordered_map>
#include <string>
#include <mutex>
#include <any>
#include <utility>
#include <stdexcept>

class DataConnector {
public:
    // Get the one-and-only instance
    static DataConnector& instance() {
        static DataConnector inst;
        return inst;
    }

    
    template<typename T>
    void registerData(const std::string& name, T&& value) {
        std::lock_guard<std::mutex> lg(mutex_);
        // Forward ‘value’ into the container, preserving its value‐category
        auto it = data_.find(name);
        if (it!=data_.end()){
            throw std::runtime_error("Data '" + name + "' is already registered in DataConnector");
        }
        else{
            data_[name] = std::forward<T>(value);
        }
    }

    // Retrieve a copy or pointer (for non‐aggregate types)
    template<typename T>
    T getDataAs(const std::string& name) {
        std::lock_guard<std::mutex> lg(mutex_);
        auto it = data_.find(name);
        if (it == data_.end())
            throw std::runtime_error("DataConnector: no data named '" + name + "'");
        return std::any_cast<T>(it->second);  // throws bad_any_cast
    }

    // Retrieve by reference (for mutating or large types)
    template<typename T>
    T& getDataAsRef(const std::string& name) {
        std::lock_guard<std::mutex> lg(mutex_);
        auto it = data_.find(name);
        if (it == data_.end())
            throw std::runtime_error("DataConnector: no data named '" + name + "'");
        return std::any_cast<T&>(it->second);  // throws bad_any_cast
    }

    // Remove one entry
    void deregisterData(const std::string& name) {
        std::lock_guard<std::mutex> lg(mutex_);
        auto it = data_.find(name);
        if (it == data_.end()) {
            throw std::runtime_error("DataConnector: no data named '" + name + "' to deregister");
        }
        data_.erase(it);
    }

    // Remove everything
    void deregisterAll() {
        std::lock_guard<std::mutex> lg(mutex_);
        data_.clear();
    }

    // Disallow copying or assignment
    DataConnector(const DataConnector&)            = delete;
    DataConnector& operator=(const DataConnector&) = delete;

private:
    DataConnector()  = default;   // private ctor/dtor
    ~DataConnector() = default;

    std::unordered_map<std::string, std::any> data_;
    std::mutex                          mutex_;
};
