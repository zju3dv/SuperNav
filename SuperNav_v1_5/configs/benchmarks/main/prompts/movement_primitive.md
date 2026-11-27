- hab_forward(distance_m=...) - move forward in small bounded steps.
- hab_backward(distance_m=...) - recover from tight spaces or collisions. Use it sparingly and then reorient.

Movement rules:
- You do NOT have WAM or any map/path planner.
- Move only with hab_forward / hab_backward / hab_turn.
- Before forward movement, check passability/depth for the direction you intend.
- After movement or any collision/recovery, inspect the returned surround images and reassess.
