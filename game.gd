extends Node2D

## Voice-controlled fox game.
##
## Hold SPACE to speak a command. Voxtral transcribes the audio, Ministral
## interprets it as a Mistral-style function call, and the result is
## dispatched to the appropriate fox(es).

const VOXTRAL_MODEL_PATH := "/Users/carloshurtado/Documents/projects/foxtral-carlos/voxtral.cpp/models/voxtral/Q4_0.gguf"
const LLAMA_MODEL_PATH := "/Users/carloshurtado/Documents/projects/foxtral-carlos/llama.cpp/models/mistralai_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"

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

var foxes: Dictionary = {}
var locations: Dictionary = {}

var _llm_accumulator: String = ""


func _ready() -> void:
	_register_world()
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


# ── World registry ───────────────────────────────────────────────────────────

func _register_world() -> void:
	var world := $World
	for child in world.get_children():
		if child.has_method("move_to"):
			foxes[child.fox_id] = child
		elif child.get("label_name"):
			locations[child.label_name.to_lower()] = child.global_position

	_log("Registered %d foxes: %s" % [foxes.size(), ", ".join(foxes.keys())])
	_log("Registered %d locations: %s" % [locations.size(), ", ".join(locations.keys())])


# ── UI (compact overlay at the bottom) ───────────────────────────────────────

func _setup_ui() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)

	var panel := PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_BOTTOM_WIDE)
	panel.offset_top = -220
	panel.offset_bottom = 0
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.08, 0.08, 0.12, 0.88)
	style.content_margin_left = 16
	style.content_margin_right = 16
	style.content_margin_top = 10
	style.content_margin_bottom = 10
	style.corner_radius_top_left = 8
	style.corner_radius_top_right = 8
	panel.add_theme_stylebox_override("panel", style)
	canvas.add_child(panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	var top_row := HBoxContainer.new()
	top_row.add_theme_constant_override("separation", 12)
	vbox.add_child(top_row)

	status_label = Label.new()
	status_label.text = "Initializing..."
	status_label.add_theme_font_size_override("font_size", 14)
	status_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	status_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top_row.add_child(status_label)

	var mic_lbl := Label.new()
	mic_lbl.text = "Mic:"
	mic_lbl.add_theme_font_size_override("font_size", 13)
	top_row.add_child(mic_lbl)

	mic_level_bar = ProgressBar.new()
	mic_level_bar.min_value = 0.0
	mic_level_bar.max_value = 1.0
	mic_level_bar.show_percentage = false
	mic_level_bar.custom_minimum_size = Vector2(120, 12)
	top_row.add_child(mic_level_bar)

	vbox.add_child(HSeparator.new())

	transcript_label = RichTextLabel.new()
	transcript_label.bbcode_enabled = true
	transcript_label.scroll_following = true
	transcript_label.size_flags_vertical = Control.SIZE_EXPAND_FILL
	transcript_label.add_theme_font_size_override("normal_font_size", 13)
	vbox.add_child(transcript_label)

	var hint := Label.new()
	hint.text = "Hold SPACE to speak a command  |  e.g. \"Fox 1, move to the forest\""
	hint.add_theme_font_size_override("font_size", 12)
	hint.add_theme_color_override("font_color", Color(0.45, 0.45, 0.45))
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

	_log("Loading Voxtral model...")
	status_label.text = "Loading Voxtral model..."
	if not stt.load_model(VOXTRAL_MODEL_PATH):
		_log("[color=red]FAILED to load Voxtral model[/color]")
		status_label.text = "ERROR: Voxtral model load failed"


func _on_stt_model_loaded() -> void:
	_log("[color=green]Voxtral model loaded[/color]")
	_update_status()


# ── LLaMA (function-calling LLM) ────────────────────────────────────────────

func _setup_llama() -> void:
	if not ClassDB.class_exists(&"LlamaContext"):
		_log("[color=yellow]WARNING: LlamaContext class not found — LLM disabled[/color]")
		return

	llm = LlamaContext.new()
	llm.context_size = 4096
	llm.temperature = 0.1
	llm.max_tokens = 256
	llm.top_p = 0.95
	llm.top_k = 40
	llm.system_prompt = _build_system_prompt()
	llm.token_generated.connect(_on_llm_token)
	llm.generation_completed.connect(_on_llm_done)
	llm.model_loaded.connect(_on_llm_model_loaded)

	_log("Loading LLM model...")
	if not llm.load_model(LLAMA_MODEL_PATH):
		_log("[color=yellow]LLM model not found — LLM disabled[/color]")
		llm = null


func _build_system_prompt() -> String:
	var fox_names := ", ".join(foxes.keys())
	var loc_names := ", ".join(locations.keys())

	var fox_actions := '["move_to","sit","jump","eat","hide","show","howl","patrol","stop"]'

	return """You control foxes in a game by outputting JSON function calls.

Foxes: %s
Locations: %s
Actions: %s

"move_to" requires "foxes" (array of fox names or "all") and "target" (location name).
All other actions require only "foxes" (array of fox names or "all").

If the user says "fox" or "foxes" without a number, use ["all"]. "Fox 1" means ["Fox1"], etc.

RESPOND WITH ONLY THIS JSON FORMAT, NO OTHER TEXT:
[{"name":"<action>","arguments":{"foxes":[...],...}}]

Examples:
User: Fox move to the forest -> [{"name":"move_to","arguments":{"foxes":["all"],"target":"forest"}}]
User: Fox 1 sit -> [{"name":"sit","arguments":{"foxes":["Fox1"]}}]
User: All foxes howl -> [{"name":"howl","arguments":{"foxes":["all"]}}]""" % [fox_names, loc_names, fox_actions]


func _on_llm_model_loaded() -> void:
	_log("[color=green]LLM model loaded[/color]")
	_update_status()


func _on_llm_token(token: String) -> void:
	_llm_accumulator += token


func _on_llm_done(full_text: String) -> void:
	_log("[color=white]LLM raw: %s[/color]" % full_text.strip_edges())
	_parse_and_dispatch(full_text)
	_llm_accumulator = ""
	_update_status()


# ── Function-call parsing & dispatch ─────────────────────────────────────────

func _parse_and_dispatch(raw: String) -> void:
	var text := raw.strip_edges()

	if text.begins_with("[TOOL_CALLS]"):
		text = text.substr(len("[TOOL_CALLS]")).strip_edges()

	var start := text.find("[")
	var end := text.rfind("]")
	if start == -1 or end == -1 or end <= start:
		_log("[color=yellow]Could not find JSON array in LLM response[/color]")
		return

	var json_str := text.substr(start, end - start + 1)
	var parsed = JSON.parse_string(json_str)

	if not parsed is Array or parsed.size() == 0:
		_log("[color=yellow]Failed to parse tool call JSON: %s[/color]" % json_str)
		return

	for tool_call in parsed:
		if tool_call is Dictionary:
			_dispatch_tool_call(tool_call)


func _dispatch_tool_call(tc: Dictionary) -> void:
	var fn_name: String = tc.get("name", "")
	var args: Dictionary = tc.get("arguments", {})
	var fox_names: Array = args.get("foxes", ["all"])
	var target_foxes := _resolve_foxes(fox_names)

	if target_foxes.is_empty():
		_log("[color=yellow]No foxes matched: %s[/color]" % str(fox_names))
		return

	var fox_list := ", ".join(target_foxes.map(func(f): return f.fox_id))
	_log("[color=green]>> %s(%s) -> [%s][/color]" % [fn_name, str(args), fox_list])
	transcript_label.append_text("[color=green]Action:[/color] %s -> %s\n" % [fn_name, fox_list])

	match fn_name:
		"move_to":
			var target_key: String = args.get("target", "").to_lower()
			if target_key in locations:
				var loc: Vector2 = locations[target_key]
				for fox in target_foxes:
					fox.move_to(loc + Vector2(randf_range(-30, 30), randf_range(-30, 30)))
			else:
				_log("[color=yellow]Unknown location: %s[/color]" % target_key)
		"sit":
			for fox in target_foxes: fox.sit()
		"jump":
			for fox in target_foxes: fox.jump()
		"eat":
			for fox in target_foxes: fox.eat()
		"hide":
			for fox in target_foxes: fox.hide_fox()
		"show":
			for fox in target_foxes: fox.show_fox()
		"howl":
			for fox in target_foxes: fox.howl()
		"patrol":
			for fox in target_foxes: fox.patrol()
		"stop":
			for fox in target_foxes: fox.stop_action()
		_:
			_log("[color=yellow]Unknown function: %s[/color]" % fn_name)


func _resolve_foxes(names: Array) -> Array:
	var result := []
	for raw_name in names:
		var n: String = str(raw_name).strip_edges()
		if n.to_lower() == "all":
			return foxes.values()
		if n in foxes:
			result.append(foxes[n])
		else:
			for key in foxes:
				if key.to_lower() == n.to_lower():
					result.append(foxes[key])
					break
	return result


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
		status_label.text = "Interpreting command..."
		_llm_accumulator = ""
		llm.generate_text(clean)
	else:
		_update_status()


# ── Per-frame: drain mic buffer ──────────────────────────────────────────────

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
	var stt_ok: bool = stt != null and stt.is_loaded
	var llm_ok: bool = llm != null and llm.is_loaded
	if stt_ok and llm_ok:
		status_label.text = "Ready — hold SPACE to command your foxes"
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
