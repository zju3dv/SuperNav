# Memory-node Examples

## Area Node

```json
{
  "id": "pink-bedroom-01",
  "kind": "area",
  "name": "pink accent bedroom",
  "description": "Area node. Bedroom with a pink accent wall, white blinds over the window, a bed on one side, and a window ledge below the blinds.",
  "records": [
    "Entered and visually inspected this area.",
    "Contains white blinds and a window ledge; possible laptop target context.",
    "Open doorway beside the bed and below the white-blinds window leads toward the white bedroom hallway; explored.",
    "Narrow dark passage beside the wall niche and glass cabinet leads to an unseen dark space; unexplored."
  ],
  "pano_images": {
    "front": "pano:...:front",
    "right": "pano:...:right",
    "back": "pano:...:back",
    "left": "pano:...:left"
  }
}
```

## Door Node With One Known Side

```json
{
  "id": "door-white-hall-pink-bedroom-01",
  "kind": "door",
  "name": "doorway between white bedroom hall and pink bedroom",
  "description": "Door node. Open doorway from the white bedroom hallway toward a bedroom with a pink accent wall and white blinds visible inside.",
  "records": [
    "White wall panels and metallic trim frame a long corridor on the hall side.",
    "Pink accent wall and white blinds are visible beyond the threshold; the room beyond is unexplored."
  ],
  "pano_images": {}
}
```

## Door Node After Centered Threshold View

```json
{
  "id": "door-white-hall-pink-bedroom-01",
  "kind": "door",
  "name": "doorway between white bedroom hall and pink bedroom",
  "description": "Door node. Threshold connecting the white bedroom hallway to the pink-accent bedroom.",
  "records": [
    "White wall panels and metallic trim identify the hallway side.",
    "Pink accent wall, white blinds, bed, and window ledge identify the bedroom side.",
    "Both sides of the threshold have been visually inspected."
  ],
  "pano_images": {
    "hall_side": "pano:...:front",
    "room_side": "pano:...:back"
  }
}
```
