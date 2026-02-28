#include "llama_context.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/os.hpp>

#include "llama.h"

namespace godot {

LlamaContext::LlamaContext() {
    llama_backend_init();
}

LlamaContext::~LlamaContext() {
    stop_generation();
    unload_model();
    llama_backend_free();
}

void LlamaContext::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_model", "path"), &LlamaContext::load_model);
    ClassDB::bind_method(D_METHOD("unload_model"), &LlamaContext::unload_model);
    ClassDB::bind_method(D_METHOD("generate_text", "prompt"), &LlamaContext::generate_text);
    ClassDB::bind_method(D_METHOD("stop_generation"), &LlamaContext::stop_generation);

    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &LlamaContext::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaContext::get_model_path);
    ClassDB::bind_method(D_METHOD("set_context_size", "size"), &LlamaContext::set_context_size);
    ClassDB::bind_method(D_METHOD("get_context_size"), &LlamaContext::get_context_size);
    ClassDB::bind_method(D_METHOD("set_temperature", "temp"), &LlamaContext::set_temperature);
    ClassDB::bind_method(D_METHOD("get_temperature"), &LlamaContext::get_temperature);
    ClassDB::bind_method(D_METHOD("set_max_tokens", "max"), &LlamaContext::set_max_tokens);
    ClassDB::bind_method(D_METHOD("get_max_tokens"), &LlamaContext::get_max_tokens);
    ClassDB::bind_method(D_METHOD("set_top_p", "p"), &LlamaContext::set_top_p);
    ClassDB::bind_method(D_METHOD("get_top_p"), &LlamaContext::get_top_p);
    ClassDB::bind_method(D_METHOD("set_top_k", "k"), &LlamaContext::set_top_k);
    ClassDB::bind_method(D_METHOD("get_top_k"), &LlamaContext::get_top_k);
    ClassDB::bind_method(D_METHOD("set_system_prompt", "prompt"), &LlamaContext::set_system_prompt);
    ClassDB::bind_method(D_METHOD("get_system_prompt"), &LlamaContext::get_system_prompt);
    ClassDB::bind_method(D_METHOD("get_is_running"), &LlamaContext::get_is_running);
    ClassDB::bind_method(D_METHOD("get_is_loaded"), &LlamaContext::get_is_loaded);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path"), "set_model_path", "get_model_path");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "context_size"), "set_context_size", "get_context_size");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "temperature"), "set_temperature", "get_temperature");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tokens"), "set_max_tokens", "get_max_tokens");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_p"), "set_top_p", "get_top_p");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "top_k"), "set_top_k", "get_top_k");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "system_prompt"), "set_system_prompt", "get_system_prompt");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_running"), "", "get_is_running");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_loaded"), "", "get_is_loaded");

    ADD_SIGNAL(MethodInfo("token_generated", PropertyInfo(Variant::STRING, "token")));
    ADD_SIGNAL(MethodInfo("generation_completed", PropertyInfo(Variant::STRING, "full_text")));
    ADD_SIGNAL(MethodInfo("model_loaded"));
}

bool LlamaContext::load_model(const String &path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (model_) {
        unload_model();
    }

    std::string path_utf8 = path.utf8().get_data();

    llama_model_params model_params = llama_model_default_params();
    model_ = llama_model_load_from_file(path_utf8.c_str(), model_params);

    if (!model_) {
        UtilityFunctions::printerr("LlamaContext: Failed to load model from: ", path);
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size_;
    ctx_params.n_batch = 512;

    ctx_ = llama_init_from_model(model_, ctx_params);

    if (!ctx_) {
        UtilityFunctions::printerr("LlamaContext: Failed to create context");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    model_path_ = path;
    call_deferred("emit_signal", "model_loaded");
    return true;
}

void LlamaContext::unload_model() {
    stop_generation();
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

void LlamaContext::generate_text(const String &prompt) {
    if (running_.load()) {
        UtilityFunctions::printerr("LlamaContext: Generation already in progress");
        return;
    }
    if (!model_ || !ctx_) {
        UtilityFunctions::printerr("LlamaContext: No model loaded");
        return;
    }

    std::string prompt_utf8 = prompt.utf8().get_data();
    stop_requested_.store(false);

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    worker_thread_ = std::thread(&LlamaContext::_generation_thread, this, prompt_utf8);
}

void LlamaContext::stop_generation() {
    stop_requested_.store(true);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

std::string LlamaContext::_apply_chat_template(const std::string &user_message) const {
    std::vector<llama_chat_message> messages;

    if (!system_prompt_.empty()) {
        messages.push_back({"system", system_prompt_.c_str()});
    }
    messages.push_back({"user", user_message.c_str()});

    const char *tmpl = llama_model_chat_template(model_, nullptr);

    int32_t len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (len < 0) {
        UtilityFunctions::printerr("LlamaContext: chat template not supported, using raw prompt");
        return user_message;
    }

    std::string formatted(len, '\0');
    llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size() + 1);
    return formatted;
}

void LlamaContext::_generation_thread(const std::string &prompt_text) {
    running_.store(true);

    llama_memory_clear(llama_get_memory(ctx_), true);

    std::string formatted = _apply_chat_template(prompt_text);

    const llama_vocab *vocab = llama_model_get_vocab(model_);

    int n_prompt_tokens = -llama_tokenize(vocab, formatted.c_str(), formatted.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt_tokens);
    llama_tokenize(vocab, formatted.c_str(), formatted.size(), tokens.data(), tokens.size(), true, true);

    if (tokens.empty()) {
        running_.store(false);
        return;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k_));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p_, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature_));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_, batch) != 0) {
        UtilityFunctions::printerr("LlamaContext: Failed to decode prompt");
        llama_sampler_free(smpl);
        running_.store(false);
        return;
    }

    std::string full_text;
    int n_generated = 0;

    while (n_generated < max_tokens_ && !stop_requested_.load()) {
        llama_token new_token = llama_sampler_sample(smpl, ctx_, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            break;
        }

        std::string piece(buf, n);
        full_text += piece;

        String token_str = String::utf8(piece.c_str(), piece.size());
        call_deferred("emit_signal", "token_generated", token_str);

        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx_, batch) != 0) {
            break;
        }

        n_generated++;
    }

    String result = String::utf8(full_text.c_str(), full_text.size());
    call_deferred("emit_signal", "generation_completed", result);

    llama_sampler_free(smpl);
    running_.store(false);
}

void LlamaContext::set_model_path(const String &path) { model_path_ = path; }
String LlamaContext::get_model_path() const { return model_path_; }

void LlamaContext::set_context_size(int size) { context_size_ = size; }
int LlamaContext::get_context_size() const { return context_size_; }

void LlamaContext::set_temperature(float temp) { temperature_ = temp; }
float LlamaContext::get_temperature() const { return temperature_; }

void LlamaContext::set_max_tokens(int max) { max_tokens_ = max; }
int LlamaContext::get_max_tokens() const { return max_tokens_; }

void LlamaContext::set_top_p(float p) { top_p_ = p; }
float LlamaContext::get_top_p() const { return top_p_; }

void LlamaContext::set_top_k(int k) { top_k_ = k; }
int LlamaContext::get_top_k() const { return top_k_; }

void LlamaContext::set_system_prompt(const String &prompt) { system_prompt_ = prompt.utf8().get_data(); }
String LlamaContext::get_system_prompt() const { return String::utf8(system_prompt_.c_str()); }

bool LlamaContext::get_is_running() const { return running_.load(); }
bool LlamaContext::get_is_loaded() const { return model_ != nullptr && ctx_ != nullptr; }

} // namespace godot
