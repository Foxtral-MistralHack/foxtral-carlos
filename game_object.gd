extends Node2D

## A landmark / point-of-interest in the game world.
## Foxes can be commanded to move to these locations.

@export var label_name: String = "Location"
@export var object_color: Color = Color.WHITE
@export var shape_type: String = "circle"  ## "circle", "rect", "triangle_cluster"

var _label: Label


func _ready() -> void:
	_build_visuals()


func _build_visuals() -> void:
	match shape_type:
		"triangle_cluster":
			_draw_forest()
		"rect":
			_draw_house()
		"circle":
			_draw_chickens()
		_:
			_draw_chickens()

	_label = Label.new()
	_label.text = label_name
	_label.add_theme_font_size_override("font_size", 16)
	_label.add_theme_color_override("font_color", Color.WHITE)
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_label.position = Vector2(-50, -80)
	_label.size = Vector2(100, 24)
	add_child(_label)


func _draw_forest() -> void:
	var offsets := [Vector2(-20, 0), Vector2(20, 0), Vector2(0, -10)]
	for offset in offsets:
		var tree := Polygon2D.new()
		tree.polygon = PackedVector2Array([
			Vector2(0, -35),
			Vector2(20, 10),
			Vector2(-20, 10),
		])
		tree.color = object_color
		tree.position = offset
		add_child(tree)

		var trunk := Polygon2D.new()
		trunk.polygon = PackedVector2Array([
			Vector2(-4, 10),
			Vector2(4, 10),
			Vector2(4, 22),
			Vector2(-4, 22),
		])
		trunk.color = Color(0.45, 0.3, 0.15)
		trunk.position = offset
		add_child(trunk)


func _draw_house() -> void:
	var walls := Polygon2D.new()
	walls.polygon = PackedVector2Array([
		Vector2(-30, -10),
		Vector2(30, -10),
		Vector2(30, 25),
		Vector2(-30, 25),
	])
	walls.color = object_color
	add_child(walls)

	var roof := Polygon2D.new()
	roof.polygon = PackedVector2Array([
		Vector2(-36, -10),
		Vector2(0, -40),
		Vector2(36, -10),
	])
	roof.color = object_color.darkened(0.3)
	add_child(roof)

	var door := Polygon2D.new()
	door.polygon = PackedVector2Array([
		Vector2(-7, 5),
		Vector2(7, 5),
		Vector2(7, 25),
		Vector2(-7, 25),
	])
	door.color = Color(0.35, 0.22, 0.1)
	add_child(door)


func _draw_chickens() -> void:
	var positions := [Vector2(0, 0), Vector2(-18, 12), Vector2(18, 8), Vector2(-8, -12), Vector2(12, -10)]
	for i in positions.size():
		var body := Polygon2D.new()
		var r := 8.0
		var pts := PackedVector2Array()
		for a in range(12):
			var angle := float(a) / 12.0 * TAU
			pts.append(Vector2(cos(angle) * r, sin(angle) * r))
		body.polygon = pts
		body.color = object_color
		body.position = positions[i]
		add_child(body)

		var beak := Polygon2D.new()
		beak.polygon = PackedVector2Array([
			Vector2(r, -2),
			Vector2(r + 5, 0),
			Vector2(r, 2),
		])
		beak.color = Color(1.0, 0.5, 0.0)
		beak.position = positions[i]
		add_child(beak)
