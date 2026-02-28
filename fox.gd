extends Node2D

## A fox entity controlled via voice commands through the LLM.
## Each fox has a unique id, a color, and a set of actions it can perform.

@export var fox_id: String = "Fox1"
@export var fox_color: Color = Color(1.0, 0.5, 0.0)

enum State { IDLE, MOVING, SITTING, JUMPING, EATING, HIDDEN, HOWLING, PATROLLING }

var state: State = State.IDLE
var carrying_chicken := false
var _tween: Tween
var _patrol_tween: Tween
var _base_scale: Vector2

var _body: Polygon2D
var _label: Label
var _chicken_icon: Node2D


func _ready() -> void:
	_base_scale = scale
	_build_visuals()


func _build_visuals() -> void:
	_body = Polygon2D.new()
	_body.polygon = PackedVector2Array([
		Vector2(0, -20),
		Vector2(14, 10),
		Vector2(0, 4),
		Vector2(-14, 10),
	])
	_body.color = fox_color
	add_child(_body)

	var left_ear := Polygon2D.new()
	left_ear.polygon = PackedVector2Array([
		Vector2(-10, -18),
		Vector2(-6, -32),
		Vector2(-2, -18),
	])
	left_ear.color = fox_color.darkened(0.15)
	add_child(left_ear)

	var right_ear := Polygon2D.new()
	right_ear.polygon = PackedVector2Array([
		Vector2(2, -18),
		Vector2(6, -32),
		Vector2(10, -18),
	])
	right_ear.color = fox_color.darkened(0.15)
	add_child(right_ear)

	var tail := Polygon2D.new()
	tail.polygon = PackedVector2Array([
		Vector2(-4, 8),
		Vector2(-18, 20),
		Vector2(-12, 24),
		Vector2(0, 14),
	])
	tail.color = fox_color.lightened(0.1)
	add_child(tail)

	_label = Label.new()
	_label.text = fox_id
	_label.add_theme_font_size_override("font_size", 13)
	_label.add_theme_color_override("font_color", Color.WHITE)
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_label.position = Vector2(-30, -50)
	_label.size = Vector2(60, 20)
	add_child(_label)


func _kill_tween() -> void:
	if _tween and _tween.is_valid():
		_tween.kill()
	_tween = null


func _kill_patrol() -> void:
	if _patrol_tween and _patrol_tween.is_valid():
		_patrol_tween.kill()
	_patrol_tween = null


func _build_chicken_icon() -> Node2D:
	var icon := Node2D.new()
	icon.position = Vector2(16, -8)

	var body_shape := Polygon2D.new()
	var pts := PackedVector2Array()
	for a in range(10):
		var angle := float(a) / 10.0 * TAU
		pts.append(Vector2(cos(angle) * 5.0, sin(angle) * 5.0))
	body_shape.polygon = pts
	body_shape.color = Color(1.0, 0.95, 0.4)
	icon.add_child(body_shape)

	var beak := Polygon2D.new()
	beak.polygon = PackedVector2Array([
		Vector2(5, -1),
		Vector2(8, 0),
		Vector2(5, 1),
	])
	beak.color = Color(1.0, 0.5, 0.0)
	icon.add_child(beak)

	return icon


# ── Actions ──────────────────────────────────────────────────────────────────

func steal() -> void:
	if carrying_chicken:
		return
	carrying_chicken = true
	_label.text = fox_id + " [chicken]"

	_chicken_icon = _build_chicken_icon()
	add_child(_chicken_icon)

	_kill_tween()
	_tween = create_tween()
	_tween.tween_property(_chicken_icon, "scale", Vector2(1.3, 1.3), 0.15)
	_tween.tween_property(_chicken_icon, "scale", Vector2.ONE, 0.15)


func move_to(target_pos: Vector2) -> void:
	_kill_tween()
	_kill_patrol()
	state = State.MOVING
	_label.text = fox_id + " (moving)"

	var dist := global_position.distance_to(target_pos)
	var duration := clampf(dist / 200.0, 0.3, 4.0)

	_tween = create_tween().set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	_tween.tween_property(self, "global_position", target_pos, duration)
	_tween.tween_callback(_on_move_done)


func _on_move_done() -> void:
	state = State.IDLE
	_label.text = fox_id + (" [chicken]" if carrying_chicken else "")


func sit() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.SITTING
	_label.text = fox_id + " (sitting)"

	_tween = create_tween().set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
	_tween.tween_property(self, "scale", _base_scale * Vector2(1.2, 0.6), 0.3)


func jump() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.JUMPING
	_label.text = fox_id + " (jumping)"

	var origin := position
	_tween = create_tween()
	_tween.tween_property(self, "position", origin + Vector2(0, -60), 0.2) \
		.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
	_tween.tween_property(self, "position", origin, 0.2) \
		.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	_tween.tween_callback(func():
		state = State.IDLE
		_label.text = fox_id
	)


func eat() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.EATING
	_label.text = fox_id + " (eating)"

	_tween = create_tween().set_loops(4)
	_tween.tween_property(self, "scale", _base_scale * 1.15, 0.2) \
		.set_trans(Tween.TRANS_SINE)
	_tween.tween_property(self, "scale", _base_scale, 0.2) \
		.set_trans(Tween.TRANS_SINE)
	_tween.finished.connect(func():
		state = State.IDLE
		_label.text = fox_id
	)


func hide_fox() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.HIDDEN
	_label.text = fox_id + " (hidden)"

	_tween = create_tween().set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	_tween.tween_property(self, "modulate:a", 0.15, 0.4)


func show_fox() -> void:
	_kill_tween()
	state = State.IDLE
	_label.text = fox_id

	_tween = create_tween().set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
	_tween.tween_property(self, "modulate:a", 1.0, 0.3)


func howl() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.HOWLING
	_label.text = fox_id + " (howling)"

	var original_color := fox_color
	_tween = create_tween()
	_tween.tween_property(self, "scale", _base_scale * 1.3, 0.25) \
		.set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	_tween.parallel().tween_property(_body, "color", Color.WHITE, 0.15)
	_tween.tween_property(_body, "color", original_color, 0.3)
	_tween.parallel().tween_property(self, "scale", _base_scale, 0.3) \
		.set_trans(Tween.TRANS_SINE)
	_tween.tween_callback(func():
		state = State.IDLE
		_label.text = fox_id
	)


func patrol() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.PATROLLING
	_label.text = fox_id + " (patrolling)"
	_patrol_step()


func _patrol_step() -> void:
	if state != State.PATROLLING:
		return
	var offset := Vector2(randf_range(-80, 80), randf_range(-80, 80))
	var target := global_position + offset

	_patrol_tween = create_tween().set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	_patrol_tween.tween_property(self, "global_position", target, randf_range(0.8, 1.5))
	_patrol_tween.tween_callback(_patrol_step)


func stop_action() -> void:
	_kill_tween()
	_kill_patrol()
	scale = _base_scale
	modulate.a = 1.0
	_body.color = fox_color
	state = State.IDLE
	_label.text = fox_id + (" [chicken]" if carrying_chicken else "")
