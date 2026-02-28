#include "voxtral_context.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "voxtral.h"

namespace godot {

VoxtralContext::VoxtralContext() {}

VoxtralContext::~VoxtralContext() {
    stop_requested_.store(true);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    unload_model();
}

void VoxtralContext::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "path"), &VoxtralContext::load_model);
    ClassDB::bind_method(D_METHOD("unload_model"), &VoxtralContext::unload_model);
    ClassDB::bind_method(D_METHOD("transcribe_file", "wav_path"), &VoxtralContext::transcribe_file);
    ClassDB::bind_method(D_METHOD("transcribe_audio", "samples"), &VoxtralContext::transcribe_audio);

    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &VoxtralContext::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &VoxtralContext::get_model_path);
    ClassDB::bind_method(D_METHOD("set_n_threads", "n"), &VoxtralContext::set_n_threads);
    ClassDB::bind_method(D_METHOD("get_n_threads"), &VoxtralContext::get_n_threads);
    ClassDB::bind_method(D_METHOD("set_max_tokens", "n"), &VoxtralContext::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"), &VoxtralContext::get_max_tokens);
    ClassDB::bind_method(D_METHOD("get_is_running"), &VoxtralContext::get_is_running);
    ClassDB::bind_method(D_METHOD("get_is_loaded"), &VoxtralContext::get_is_loaded);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path"), "set_model_path", "get_model_path");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads"), "set_n_threads", "get_n_threads");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens"), "set_max_tokens", "get_max_tokens");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_running"), "", "get_is_running");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_loaded"), "", "get_is_loaded");

    ADD_SIGNAL(MethodInfo("transcription_completed", PropertyInfo(Variant::STRING, "text")));
    ADD_SIGNAL(MethodInfo("model_loaded"));
}

bool VoxtralContext::load_model(const String &path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    unload_model();

    std::string path_utf8 = path.utf8().get_data();

    UtilityFunctions::print("VoxtralContext: Loading model from: ", path);

    model_ = voxtral_model_load_from_file(
        path_utf8,
        [](voxtral_log_level level, const std::string &msg) {
            if (level <= voxtral_log_level::info) {
                UtilityFunctions::print("voxtral: ", String::utf8(msg.c_str(), msg.size()));
            }
        },
        voxtral_gpu_backend::metal);

    if (!model_) {
        UtilityFunctions::printerr("VoxtralContext: Failed to load model from: ", path);
        return false;
    }

    voxtral_context_params params;
    params.n_threads = n_threads_;
    params.log_level = voxtral_log_level::info;
    params.gpu = voxtral_gpu_backend::metal;

    ctx_ = voxtral_init_from_model(model_, params);
    if (!ctx_) {
        UtilityFunctions::printerr("VoxtralContext: Failed to create inference context");
        voxtral_model_free(model_);
        model_ = nullptr;
        return false;
    }

    model_path_ = path;
    UtilityFunctions::print("VoxtralContext: Model loaded successfully");
    call_deferred("emit_signal", "model_loaded");
    return true;
}

void VoxtralContext::unload_model() {
    if (running_.load()) {
        stop_requested_.store(true);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    if (ctx_) {
        voxtral_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        voxtral_model_free(model_);
        model_ = nullptr;
    }
}

void VoxtralContext::transcribe_file(const String &wav_path) {
    if (running_.load()) {
        UtilityFunctions::printerr("VoxtralContext: Already transcribing");
        return;
    }
    if (!ctx_) {
        UtilityFunctions::printerr("VoxtralContext: No model loaded");
        return;
    }

    std::string path_utf8 = wav_path.utf8().get_data();
    stop_requested_.store(false);

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    worker_thread_ = std::thread(&VoxtralContext::_transcribe_file_thread, this, path_utf8);
}

void VoxtralContext::_transcribe_file_thread(const std::string &wav_path) {
    running_.store(true);

    UtilityFunctions::print("VoxtralContext: Transcribing: ", String(wav_path.c_str()));

    voxtral_result result;
    bool ok = voxtral_transcribe_file(*ctx_, wav_path, max_tokens_, result);

    if (ok && !result.text.empty()) {
        String text = String::utf8(result.text.c_str(), result.text.size());
        UtilityFunctions::print("VoxtralContext: Transcript: ", text);
        call_deferred("emit_signal", "transcription_completed", text);
    } else {
        UtilityFunctions::printerr("VoxtralContext: Transcription failed for: ", String(wav_path.c_str()));
        call_deferred("emit_signal", "transcription_completed", String(""));
    }

    running_.store(false);
}

void VoxtralContext::transcribe_audio(const PackedFloat32Array &samples) {
    if (running_.load()) {
        UtilityFunctions::printerr("VoxtralContext: Already transcribing");
        return;
    }
    if (!ctx_) {
        UtilityFunctions::printerr("VoxtralContext: No model loaded");
        return;
    }

    const float *ptr = samples.ptr();
    std::vector<float> audio(ptr, ptr + samples.size());
    stop_requested_.store(false);

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    worker_thread_ = std::thread(&VoxtralContext::_transcribe_audio_thread, this, std::move(audio));
}

void VoxtralContext::_transcribe_audio_thread(std::vector<float> audio) {
    running_.store(true);

    UtilityFunctions::print("VoxtralContext: Transcribing ", (int64_t)audio.size(), " samples");

    voxtral_result result;
    bool ok = voxtral_transcribe_audio(*ctx_, audio, max_tokens_, result);

    if (ok && !result.text.empty()) {
        String text = String::utf8(result.text.c_str(), result.text.size());
        UtilityFunctions::print("VoxtralContext: Transcript: ", text);
        call_deferred("emit_signal", "transcription_completed", text);
    } else {
        UtilityFunctions::printerr("VoxtralContext: Transcription failed");
        call_deferred("emit_signal", "transcription_completed", String(""));
    }

    running_.store(false);
}

void VoxtralContext::set_model_path(const String &path) { model_path_ = path; }
String VoxtralContext::get_model_path() const { return model_path_; }

void VoxtralContext::set_n_threads(int n) { n_threads_ = n; }
int VoxtralContext::get_n_threads() const { return n_threads_; }

void VoxtralContext::set_max_tokens(int n) { max_tokens_ = n; }
int VoxtralContext::get_max_tokens() const { return max_tokens_; }

bool VoxtralContext::get_is_running() const { return running_.load(); }
bool VoxtralContext::get_is_loaded() const { return model_ != nullptr && ctx_ != nullptr; }

} // namespace godot
