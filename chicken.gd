extends Node2D

## NPC chicken. Idle only — wanders randomly within a small radius of its home.
## Uses ChickenPack sprites. Smaller than foxes.

@export var wander_radius: float = 30.0

var home_position: Vector2
var carried := false
var _wander_tween: Tween
var _sprite: AnimatedSprite2D

const FRAME_SIZE := Vector2(20, 21)
const SPRITE_SCALE := 2.0  # Foxes use 2.5; chickens stay smaller


func _ready() -> void:
	home_position = global_position
	_build_visuals()
	_wander_step()


func _build_visuals() -> void:
	var idle_sheet := load("res://ChickenPack/SpriteSheet/ChickenIdle-Sheet.png") as Texture2D
	var walk_sheet := load("res://ChickenPack/SpriteSheet/ChickenWalking.png") as Texture2D

	var frames := SpriteFrames.new()
	if frames.has_animation("default"):
		frames.remove_animation("default")

	# Idle: 5 frames (100/20)
	frames.add_animation("idle")
	frames.set_animation_speed("idle", 5.0)
	frames.set_animation_loop("idle", true)
	for i in 5:
		var atlas := AtlasTexture.new()
		atlas.atlas = idle_sheet
		atlas.region = Rect2(i * FRAME_SIZE.x, 0, FRAME_SIZE.x, FRAME_SIZE.y)
		frames.add_frame("idle", atlas)

	# Walk: 4 frames (80/20)
	frames.add_animation("walk")
	frames.set_animation_speed("walk", 6.0)
	frames.set_animation_loop("walk", true)
	for i in 4:
		var atlas := AtlasTexture.new()
		atlas.atlas = walk_sheet
		atlas.region = Rect2(i * FRAME_SIZE.x, 0, FRAME_SIZE.x, FRAME_SIZE.y)
		frames.add_frame("walk", atlas)

	_sprite = AnimatedSprite2D.new()
	_sprite.sprite_frames = frames
	_sprite.scale = Vector2(SPRITE_SCALE, SPRITE_SCALE)
	_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	add_child(_sprite)
	_sprite.play("idle")


func _kill_wander() -> void:
	if _wander_tween and _wander_tween.is_valid():
		_wander_tween.kill()
	_wander_tween = null


func set_carried(_by: Node2D) -> void:
	carried = true
	_kill_wander()
	_sprite.stop()
	_sprite.frame = 0


func _wander_step() -> void:
	if carried:
		return
	var offset := Vector2(randf_range(-wander_radius, wander_radius), randf_range(-wander_radius, wander_radius))
	var target := home_position + offset

	# Face movement direction
	if target.x < global_position.x:
		_sprite.flip_h = true
	else:
		_sprite.flip_h = false

	var duration := randf_range(1.0, 2.5)
	_sprite.play("walk")
	_wander_tween = create_tween().set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	_wander_tween.tween_property(self, "global_position", target, duration)
	_wander_tween.tween_callback(_schedule_next_wander)


func _schedule_next_wander() -> void:
	if carried:
		return
	_sprite.play("idle")
	_wander_tween = create_tween()
	_wander_tween.tween_interval(randf_range(0.5, 2.0))
	_wander_tween.tween_callback(_wander_step)
