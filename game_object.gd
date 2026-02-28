extends Node2D

## A landmark / point-of-interest in the game world.
## Foxes can be commanded to move to these locations.
## Visuals come from the background art; this node is just a position marker.

@export var label_name: String = "Location"

var _label: Label


func _ready() -> void:
	_label = Label.new()
	_label.text = label_name
	_label.add_theme_font_size_override("font_size", 14)
	_label.add_theme_color_override("font_color", Color.WHITE)
	_label.add_theme_constant_override("outline_size", 3)
	_label.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.9))
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_label.position = Vector2(-40, -50)
	_label.size = Vector2(80, 24)
	add_child(_label)
