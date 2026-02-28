#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <string>

struct llama_model;
struct llama_context;
struct llama_sampler;

namespace godot {

class LlamaContext : public RefCounted {
    GDCLASS(LlamaContext, RefCounted)

protected:
    static void _bind_methods();

public:
    LlamaContext();
    ~LlamaContext() override;

    bool load_model(const String &path);
    void unload_model();
    void generate_text(const String &prompt);
    void stop_generation();

    void set_model_path(const String &path);
    String get_model_path() const;

    void set_context_size(int size);
    int get_context_size() const;

    void set_temperature(float temp);
    float get_temperature() const;

    void set_max_tokens(int max);
    int get_max_tokens() const;

    void set_top_p(float p);
    float get_top_p() const;

    void set_top_k(int k);
    int get_top_k() const;

    bool get_is_running() const;
    bool get_is_loaded() const;

private:
    void _generation_thread(const std::string &prompt_text);

    struct llama_model *model_ = nullptr;
    struct llama_context *ctx_ = nullptr;

    String model_path_;
    int context_size_ = 2048;
    float temperature_ = 0.8f;
    int max_tokens_ = 512;
    float top_p_ = 0.95f;
    int top_k_ = 40;

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::thread worker_thread_;
    std::mutex model_mutex_;
};

} // namespace godot
