extends Node2D

## A fox entity controlled via voice commands through the LLM.
## Uses the pixel-art sprite sheet for animated visuals.

@export var fox_id: String = "Ruby"
@export var fox_color: Color = Color(1.0, 0.5, 0.0)

enum State { IDLE, MOVING, SITTING, JUMPING, EATING, HIDDEN, HOWLING, PATROLLING }

var state: State = State.IDLE
var carrying_chicken := false
var _tween: Tween
var _patrol_tween: Tween
var _base_scale: Vector2

var _sprite: AnimatedSprite2D
var _label: Label
var _chicken_icon: Node2D

const FRAME_SIZE := Vector2(32, 32)
const SPRITE_SCALE := 2.5

const ANIM_DEF := {
	"idle":  {"row": 0, "frames": 5,  "fps": 6.0,  "loop": true},
	"run":   {"row": 1, "frames": 8,  "fps": 10.0, "loop": true},
	"walk":  {"row": 2, "frames": 8,  "fps": 8.0,  "loop": true},
	"jump":  {"row": 3, "frames": 11, "fps": 12.0, "loop": false},
	"eat":   {"row": 4, "frames": 5,  "fps": 6.0,  "loop": true},
	"sit":   {"row": 5, "frames": 6,  "fps": 4.0,  "loop": true},
	"howl":  {"row": 6, "frames": 7,  "fps": 8.0,  "loop": false},
}


func _ready() -> void:
	_base_scale = scale
	_build_visuals()


func _build_visuals() -> void:
	var sheet := load("res://assets/sprites/fox_sprite_sheet.png") as Texture2D

	var frames := SpriteFrames.new()
	if frames.has_animation("default"):
		frames.remove_animation("default")

	for anim_name in ANIM_DEF:
		var def: Dictionary = ANIM_DEF[anim_name]
		frames.add_animation(anim_name)
		frames.set_animation_speed(anim_name, def["fps"])
		frames.set_animation_loop(anim_name, def["loop"])

		for i in def["frames"]:
			var atlas := AtlasTexture.new()
			atlas.atlas = sheet
			atlas.region = Rect2(
				i * FRAME_SIZE.x,
				def["row"] * FRAME_SIZE.y,
				FRAME_SIZE.x,
				FRAME_SIZE.y
			)
			frames.add_frame(anim_name, atlas)

	_sprite = AnimatedSprite2D.new()
	_sprite.sprite_frames = frames
	_sprite.scale = Vector2(SPRITE_SCALE, SPRITE_SCALE)
	_sprite.self_modulate = Color.WHITE.lerp(fox_color, 0.25)
	_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	_sprite.animation_finished.connect(_on_animation_finished)
	add_child(_sprite)
	_sprite.play("idle")

	_label = Label.new()
	_label.text = fox_id
	_label.add_theme_font_size_override("font_size", 13)
	_label.add_theme_color_override("font_color", Color.WHITE)
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_label.position = Vector2(-30, -25)
	_label.size = Vector2(60, 20)
	add_child(_label)


func _on_animation_finished() -> void:
	if state == State.HOWLING:
		state = State.IDLE
		scale = _base_scale
		_update_label()
		_sprite.play("idle")


func _update_label(suffix: String = "") -> void:
	var chicken_tag := " [chicken]" if carrying_chicken else ""
	_label.text = fox_id + suffix + chicken_tag


func _kill_tween() -> void:
	if _tween and _tween.is_valid():
		_tween.kill()
	_tween = null


func _kill_patrol() -> void:
	if _patrol_tween and _patrol_tween.is_valid():
		_patrol_tween.kill()
	_patrol_tween = null


func show_confusion() -> void:
	var qm := Sprite2D.new()
	qm.texture = load("res://question_mark.png") as Texture2D
	qm.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	qm.position = Vector2(0, -50)
	qm.scale = Vector2(0.04, 0.04)  # Original is 2000x1116; scale to ~80px
	add_child(qm)
	var tween := create_tween()
	tween.tween_interval(2.0)
	tween.tween_callback(func(): qm.queue_free())


func _build_chicken_icon() -> Node2D:
	var icon := Node2D.new()
	icon.position = Vector2(24, -18)

	var body_shape := Polygon2D.new()
	var pts := PackedVector2Array()
	for a in range(10):
		var angle := float(a) / 10.0 * TAU
		pts.append(Vector2(cos(angle) * 6.0, sin(angle) * 6.0))
	body_shape.polygon = pts
	body_shape.color = Color(1.0, 0.95, 0.4)
	icon.add_child(body_shape)

	var beak := Polygon2D.new()
	beak.polygon = PackedVector2Array([
		Vector2(6, -1.5),
		Vector2(10, 0),
		Vector2(6, 1.5),
	])
	beak.color = Color(1.0, 0.5, 0.0)
	icon.add_child(beak)

	return icon


# ── Actions ──────────────────────────────────────────────────────────────────

func steal() -> void:
	if carrying_chicken:
		return
	carrying_chicken = true
	_update_label()

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
	_update_label(" (moving)")

	if target_pos.x < global_position.x:
		_sprite.flip_h = true
	elif target_pos.x > global_position.x:
		_sprite.flip_h = false

	_sprite.play("run")

	var dist := global_position.distance_to(target_pos)
	var duration := clampf(dist / 200.0, 0.3, 4.0)

	_tween = create_tween().set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	_tween.tween_property(self, "global_position", target_pos, duration)
	_tween.tween_callback(_on_move_done)


func _on_move_done() -> void:
	state = State.IDLE
	_update_label()
	_sprite.play("idle")


func sit() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.SITTING
	_update_label(" (sitting)")
	_sprite.play("sit")


func jump() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.JUMPING
	_update_label(" (jumping)")
	_sprite.play("jump")

	var origin := position
	_tween = create_tween()
	_tween.tween_property(self, "position", origin + Vector2(0, -60), 0.25) \
		.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
	_tween.tween_property(self, "position", origin, 0.25) \
		.set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	_tween.tween_callback(func():
		state = State.IDLE
		_update_label()
		_sprite.play("idle")
	)


func eat() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.EATING
	_update_label(" (eating)")
	_sprite.play("eat")


func hide_fox() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.HIDDEN
	_update_label(" (hidden)")

	_tween = create_tween().set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	_tween.tween_property(self, "modulate:a", 0.15, 0.4)


func show_fox() -> void:
	_kill_tween()
	state = State.IDLE
	_update_label()
	_sprite.play("idle")

	_tween = create_tween().set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)
	_tween.tween_property(self, "modulate:a", 1.0, 0.3)


func howl() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.HOWLING
	_update_label(" (howling)")
	_sprite.play("howl")

	_tween = create_tween()
	_tween.tween_property(self, "scale", _base_scale * 1.2, 0.3) \
		.set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	_tween.tween_property(self, "scale", _base_scale, 0.4) \
		.set_trans(Tween.TRANS_SINE)


func patrol() -> void:
	_kill_tween()
	_kill_patrol()
	state = State.PATROLLING
	_update_label(" (patrolling)")
	_sprite.play("walk")
	_patrol_step()


func _patrol_step() -> void:
	if state != State.PATROLLING:
		return
	var offset := Vector2(randf_range(-80, 80), randf_range(-80, 80))
	var target := global_position + offset

	if target.x < global_position.x:
		_sprite.flip_h = true
	else:
		_sprite.flip_h = false

	_patrol_tween = create_tween().set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	_patrol_tween.tween_property(self, "global_position", target, randf_range(0.8, 1.5))
	_patrol_tween.tween_callback(_patrol_step)


func stop_action() -> void:
	_kill_tween()
	_kill_patrol()
	scale = _base_scale
	modulate.a = 1.0
	state = State.IDLE
	_update_label()
	_sprite.play("idle")
