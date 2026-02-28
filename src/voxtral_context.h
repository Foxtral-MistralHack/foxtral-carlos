#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>

struct voxtral_model;
struct voxtral_context;

namespace godot {

class VoxtralContext : public RefCounted {
    GDCLASS(VoxtralContext, RefCounted)

protected:
    static void _bind_methods();

public:
    VoxtralContext();
    ~VoxtralContext() override;

    bool load_model(const String &path);
    void unload_model();

    void transcribe_file(const String &wav_path);
    void transcribe_audio(const PackedFloat32Array &samples);

    void set_model_path(const String &path);
    String get_model_path() const;

    void set_n_threads(int n);
    int get_n_threads() const;

    void set_max_tokens(int n);
    int get_max_tokens() const;

    bool get_is_running() const;
    bool get_is_loaded() const;

private:
    void _transcribe_file_thread(const std::string &wav_path);
    void _transcribe_audio_thread(std::vector<float> audio);

    voxtral_model   *model_ = nullptr;
    voxtral_context *ctx_   = nullptr;

    String model_path_;
    int n_threads_  = 8;
    int max_tokens_ = 256;

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::thread worker_thread_;
    std::mutex model_mutex_;
};

} // namespace godot
