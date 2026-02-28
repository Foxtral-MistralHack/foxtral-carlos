extends Node

## Push-to-talk speech-to-text → LLM pipeline.
##
## Hold SPACE to record, release to transcribe with Voxtral,
## then feed the transcript into llama.cpp for a response.

const VOXTRAL_MODEL_PATH := "/Users/shinchan/src/hacks/foxtral-assetsv2/foxtral-carlos/voxtral.cpp/models/voxtral/Q4_0.gguf"
const LLAMA_MODEL_PATH := "/Users/shinchan/src/hacks/foxtral-assetsv2/foxtral-carlos/llama.cpp/models/mistralai_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"

const CAPTURE_BUS_NAME := "MicCapture"

var stt: VoxtralContext
var llm: LlamaContext
var mic_player: AudioStreamPlayer
var capture_effect: AudioEffectCapture

var transcript_label: RichTextLabel
var status_label: Label
var mic_level_bar: ProgressBar

var mix_rate: float
var is_recording := false
var recorded_samples := PackedFloat32Array()
var peak_level := 0.0


func _ready() -> void:
	_setup_ui()
	_setup_mic_bus()
	_setup_mic_player()

	var input_enabled = ProjectSettings.get_setting("audio/driver/enable_input", false)
	_log("Audio input enabled: %s  |  Mix rate: %d Hz" % [str(input_enabled), int(AudioServer.get_mix_rate())])
	if not input_enabled:
		_log("[color=red]WARNING: Audio input NOT enabled! Enable in Project > Audio > Driver[/color]")
	_log("")

	_setup_voxtral()
	_setup_llama()


# ── UI ───────────────────────────────────────────────────────────────────────

func _setup_ui() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)

	var margin := MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 24)
	canvas.add_child(margin)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 6)
	margin.add_child(vbox)

	var title := Label.new()
	title.text = "Foxtral — Speech-to-LLM"
	title.add_theme_font_size_override("font_size", 28)
	vbox.add_child(title)

	status_label = Label.new()
	status_label.text = "Initializing..."
	status_label.add_theme_font_size_override("font_size", 16)
	status_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	vbox.add_child(status_label)

	var mic_hbox := HBoxContainer.new()
	mic_hbox.add_theme_constant_override("separation", 8)
	vbox.add_child(mic_hbox)
	var mic_lbl := Label.new()
	mic_lbl.text = "Mic:"
	mic_lbl.add_theme_font_size_override("font_size", 14)
	mic_hbox.add_child(mic_lbl)
	mic_level_bar = ProgressBar.new()
	mic_level_bar.min_value = 0.0
	mic_level_bar.max_value = 1.0
	mic_level_bar.show_percentage = false
	mic_level_bar.custom_minimum_size = Vector2(300, 16)
	mic_hbox.add_child(mic_level_bar)

	vbox.add_child(HSeparator.new())

	var heading := Label.new()
	heading.text = "Conversation:"
	heading.add_theme_font_size_override("font_size", 18)
	vbox.add_child(heading)

	transcript_label = RichTextLabel.new()
	transcript_label.bbcode_enabled = true
	transcript_label.scroll_following = true
	transcript_label.size_flags_vertical = Control.SIZE_EXPAND_FILL
	transcript_label.add_theme_font_size_override("normal_font_size", 16)
	vbox.add_child(transcript_label)

	var hint := Label.new()
	hint.text = "Hold SPACE to speak, release to send"
	hint.add_theme_font_size_override("font_size", 13)
	hint.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(hint)


# ── Audio bus & mic ──────────────────────────────────────────────────────────

func _setup_mic_bus() -> void:
	var bus_idx := AudioServer.bus_count
	AudioServer.add_bus(bus_idx)
	AudioServer.set_bus_name(bus_idx, CAPTURE_BUS_NAME)
	AudioServer.set_bus_mute(bus_idx, true)
	capture_effect = AudioEffectCapture.new()
	capture_effect.buffer_length = 10.0
	AudioServer.add_bus_effect(bus_idx, capture_effect)
	mix_rate = AudioServer.get_mix_rate()


func _setup_mic_player() -> void:
	mic_player = AudioStreamPlayer.new()
	mic_player.stream = AudioStreamMicrophone.new()
	mic_player.bus = CAPTURE_BUS_NAME
	add_child(mic_player)
	mic_player.play()


# ── Voxtral (speech-to-text) ────────────────────────────────────────────────

func _setup_voxtral() -> void:
	if not ClassDB.class_exists(&"VoxtralContext"):
		_log("[color=red]ERROR: VoxtralContext class not found — build the GDExtension first[/color]")
		status_label.text = "ERROR: GDExtension not loaded"
		return

	stt = VoxtralContext.new()
	stt.n_threads = 8
	stt.max_tokens = 256
	stt.transcription_completed.connect(_on_transcription_completed)
	stt.model_loaded.connect(_on_stt_model_loaded)

	_log("Loading Voxtral model from: %s" % VOXTRAL_MODEL_PATH)
	status_label.text = "Loading Voxtral model..."
	var ok := stt.load_model(VOXTRAL_MODEL_PATH)
	if not ok:
		_log("[color=red]FAILED to load Voxtral model[/color]")
		status_label.text = "ERROR: Voxtral model load failed"


func _on_stt_model_loaded() -> void:
	_log("[color=green]Voxtral model loaded[/color]")
	_update_status()


# ── LLaMA (text generation) ─────────────────────────────────────────────────

func _setup_llama() -> void:
	if not ClassDB.class_exists(&"LlamaContext"):
		_log("[color=yellow]WARNING: LlamaContext class not found — LLM disabled[/color]")
		return

	llm = LlamaContext.new()
	llm.context_size = 2048
	llm.temperature = 0.7
	llm.max_tokens = 256
	llm.system_prompt = "You are a helpful voice assistant. Respond concisely."
	llm.token_generated.connect(_on_llm_token)
	llm.generation_completed.connect(_on_llm_done)
	llm.model_loaded.connect(_on_llm_model_loaded)

	_log("Loading LLaMA model from: %s" % LLAMA_MODEL_PATH)
	var ok := llm.load_model(LLAMA_MODEL_PATH)
	if not ok:
		_log("[color=yellow]LLaMA model not found at %s — LLM disabled[/color]" % LLAMA_MODEL_PATH)
		_log("[color=yellow]Download a GGUF model and place it at the path above[/color]")
		llm = null


func _on_llm_model_loaded() -> void:
	_log("[color=green]LLaMA model loaded[/color]")
	_update_status()


func _on_llm_token(token: String) -> void:
	transcript_label.append_text(token)


func _on_llm_done(full_text: String) -> void:
	transcript_label.append_text("\n\n")
	_update_status()


# ── Input handling ───────────────────────────────────────────────────────────

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.keycode == KEY_SPACE:
		if event.pressed and not event.echo and not is_recording:
			_start_recording()
		elif not event.pressed and is_recording:
			_stop_recording()


func _start_recording() -> void:
	if not stt or not stt.is_loaded:
		return
	is_recording = true
	recorded_samples.clear()
	capture_effect.clear_buffer()
	status_label.text = "Recording... (release SPACE to stop)"


func _stop_recording() -> void:
	if not is_recording:
		return
	is_recording = false
	status_label.text = "Transcribing..."

	var n_samples := recorded_samples.size()
	_log("Captured %d samples (%.1fs at %d Hz)" % [n_samples, float(n_samples) / mix_rate, int(mix_rate)])

	if n_samples < int(mix_rate * 0.3):
		_log("[color=yellow]Too short — speak for at least 0.3s[/color]")
		_update_status()
		return

	var samples_16k := _resample_to_mono_16k(recorded_samples)
	var wav_path := OS.get_user_data_dir() + "/recording.wav"
	_save_wav(wav_path, samples_16k, 16000)
	_log("Saved WAV (%d samples @ 16kHz) to: %s" % [samples_16k.size(), wav_path])

	stt.transcribe_file(wav_path)


func _on_transcription_completed(text: String) -> void:
	var clean := text.strip_edges()
	_log("Transcript: [%s]" % clean)

	if clean.is_empty():
		_log("[color=yellow]Empty transcript — try speaking louder or longer[/color]")
		_update_status()
		return

	transcript_label.append_text("[color=cyan]You:[/color] %s\n" % clean)

	if llm and llm.is_loaded:
		transcript_label.append_text("[color=green]AI:[/color] ")
		status_label.text = "Generating response..."
		llm.generate_text(clean)
	else:
		transcript_label.append_text("\n")
		_update_status()


# ── Per-frame: drain mic buffer while recording ─────────────────────────────

func _process(_delta: float) -> void:
	if is_recording:
		_drain_and_record()
	else:
		_drain_for_level()


func _drain_and_record() -> void:
	var available := capture_effect.get_frames_available()
	if available <= 0:
		return
	var frames := capture_effect.get_buffer(available)
	var local_peak := 0.0
	for frame in frames:
		var mono := (frame.x + frame.y) * 0.5
		recorded_samples.append(mono)
		var m := absf(mono)
		if m > local_peak:
			local_peak = m
	_update_mic_bar(local_peak)


func _drain_for_level() -> void:
	var avail := capture_effect.get_frames_available()
	if avail <= 0:
		mic_level_bar.value = lerpf(mic_level_bar.value, 0.0, 0.1)
		return
	var buf := capture_effect.get_buffer(mini(avail, 1024))
	var pk := 0.0
	for frame in buf:
		var m := absf((frame.x + frame.y) * 0.5)
		if m > pk:
			pk = m
	_update_mic_bar(pk)


func _update_mic_bar(local_peak: float) -> void:
	peak_level = maxf(local_peak, lerpf(peak_level, local_peak, 0.3))
	var db := 20.0 * log(maxf(peak_level, 0.00001)) / log(10.0)
	mic_level_bar.value = clampf((db + 60.0) / 60.0, 0.0, 1.0)
	peak_level *= 0.9


# ── Audio helpers ────────────────────────────────────────────────────────────

func _resample_to_mono_16k(mono: PackedFloat32Array) -> PackedFloat32Array:
	var n := mono.size()
	if n == 0:
		return PackedFloat32Array()
	var ratio := 16000.0 / mix_rate
	var out_len := int(float(n) * ratio)
	if out_len <= 0:
		return PackedFloat32Array()
	var resampled := PackedFloat32Array()
	resampled.resize(out_len)
	for i in out_len:
		var src_pos := float(i) / ratio
		var idx := int(src_pos)
		var frac := src_pos - float(idx)
		if idx + 1 < n:
			resampled[i] = mono[idx] * (1.0 - frac) + mono[idx + 1] * frac
		else:
			resampled[i] = mono[mini(idx, n - 1)]
	return resampled


func _save_wav(path: String, samples: PackedFloat32Array, rate: int) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	var n := samples.size()
	var data_size := n * 2

	f.store_buffer("RIFF".to_utf8_buffer())
	f.store_32(36 + data_size)
	f.store_buffer("WAVE".to_utf8_buffer())
	f.store_buffer("fmt ".to_utf8_buffer())
	f.store_32(16)
	f.store_16(1)
	f.store_16(1)
	f.store_32(rate)
	f.store_32(rate * 2)
	f.store_16(2)
	f.store_16(16)
	f.store_buffer("data".to_utf8_buffer())
	f.store_32(data_size)

	for s in samples:
		var v := int(clamp(s * 32767.0, -32768.0, 32767.0))
		f.store_16(v & 0xFFFF)

	f.close()


# ── Utilities ────────────────────────────────────────────────────────────────

func _update_status() -> void:
	var stt_ok := stt and stt.is_loaded
	var llm_ok := llm and llm.is_loaded
	if stt_ok and llm_ok:
		status_label.text = "Ready — hold SPACE to speak"
	elif stt_ok:
		status_label.text = "Ready (STT only, no LLM) — hold SPACE to speak"
	else:
		status_label.text = "Loading models..."


func _log(msg: String) -> void:
	var clean := msg
	for tag in ["[color=red]", "[color=green]", "[color=yellow]", "[color=cyan]", "[color=white]", "[/color]"]:
		clean = clean.replace(tag, "")
	print(clean)
	if transcript_label:
		transcript_label.append_text(msg + "\n")


func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST or what == NOTIFICATION_PREDELETE:
		if llm and llm.is_running:
			llm.stop_generation()
		if stt:
			stt.unload_model()
