- hab_visual_local_navigate(instruction="a short visual phrase for the target in front of you") - use the bridge's front-view visual local navigator for one bounded local hop toward a visible object or landmark. Internally it first tries visible-object label matching, then falls back to Grounding DINO if needed.

Visual local navigation rules:
- Visual confirmation comes first: if the latest surround images clearly show that the user request is already satisfied, close the session even if visual local navigation did not ground that object.
- Start each navigation attempt from the latest panorama_images, then choose a concise phrase for an object or landmark visible in the latest front image.
- Do not use coordinates, maps, remembered off-screen targets, object IDs, or hidden labels.
- If the goal object is visible but you still need to get closer, call hab_visual_local_navigate with a visual phrase for that object.
- If the best target is visible only in a right/left/back panorama image, turn toward it first, then call hab_visual_local_navigate after it is in front. Do not rely on visual local navigation to pick a side/back view unless the instruction itself explicitly asks for a relative direction such as "on the right" or "behind me".
- If the intended movement is simply toward the currently facing open area, or the front target is visually clear but grounding is unreliable, use depth checks and a short hab_forward step instead of forcing another visual-local call. Re-observe after each short forward step.
- If the goal object is not visible, choose a visible intermediate target that is visually likely to improve observability, then re-observe.
- Useful intermediate targets include doorways, corridor openings, room entrances, furniture clusters, nearby tables or chairs for sofa searches, and counters, sinks, refrigerators, tables, or cabinet clusters for kitchen goals.
- For EQA and ImageNav-style requests, choose visible landmarks that make the next view more informative, not the final answer or hidden goal by name.
- If the tool returns no_visual_grounding or no_scene_graph_projection_match, treat that as a failed local movement attempt, not proof that the object is absent from the images. If the goal is already visually confirmed, stop; otherwise explore with hab_turn, depth checks, or primitive movement until the target is visually clearer.
- Treat each visual navigate call as one bounded local hop only. After it returns, inspect the returned panorama_images before deciding whether the task is satisfied or choosing the next visible phrase.
- Do not call hab_navigate_wam; this arm replaces WAM with the oracle visible-target navigator.

Movement rules:
- Use the most recent panorama_images and hab_turn to pick the next visible visual phrase.
- Prefer hab_visual_local_navigate for grounded front-view targets; use short hab_forward steps when moving straight ahead into clear space is the more direct action. Do not assume either movement result means the task is satisfied until you visually verify the new view yourself.
- Do not let a visual-local grounding miss override visual evidence in the images. If the requested object is visible and you are close enough for the task, call hab_close_session.
