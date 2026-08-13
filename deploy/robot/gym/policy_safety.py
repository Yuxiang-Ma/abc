"""Shared first-action safety prompts for policy execution."""

from __future__ import annotations

import numpy as np

_JOINT_LABELS = [
    "L_J0",
    "L_J1",
    "L_J2",
    "L_J3",
    "L_J4",
    "L_J5",
    "L_grip",
    "R_J0",
    "R_J1",
    "R_J2",
    "R_J3",
    "R_J4",
    "R_J5",
    "R_grip",
]


def prompt_first_action_safety_check(
    current_state: np.ndarray,
    action: np.ndarray,
    *,
    title: str = "FIRST ACTION SAFETY CHECK",
    enter_prompt: str | None = "Press ENTER to execute, or Ctrl-C to abort > ",
) -> None:
    """Print per-joint first-action deltas and require user confirmation."""
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)

    n = min(len(_JOINT_LABELS), len(current_state), action.shape[1])

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Joint':<8} {'Current':>9} {'Action[0]':>10} {'Δ[0]':>8}")
    print("-" * 50)
    for j in range(n):
        delta = action[0, j] - current_state[j]
        print(
            f"{_JOINT_LABELS[j]:<8} {current_state[j]:>+9.4f} "
            f"{action[0, j]:>+10.4f} {delta:>+8.4f}"
        )

    max_delta = np.abs(action[0, :n] - current_state[:n]).max()
    print("-" * 50)
    print(
        f"Max |Δ| from current state: {max_delta:.4f} rad ({np.degrees(max_delta):.1f}°)"
    )
    print(f"Action chunk shape: {action.shape}")
    print("\nPer-step max |Δ| from previous step in chunk:")
    for s in range(action.shape[0]):
        prev = current_state[:n] if s == 0 else action[s - 1, :n]
        step_delta = np.abs(action[s, :n] - prev).max()
        flag = " <<<" if step_delta > 0.3 else ""
        print(
            f"  step {s:>2d}: max|Δ|={step_delta:.4f} ({np.degrees(step_delta):.1f}°){flag}"
        )
    print("=" * 70)
    if max_delta > 0.5:
        print(f"\n*** WARNING: Large first-step delta ({max_delta:.3f} rad) ***")

    if enter_prompt is not None:
        input(enter_prompt)
