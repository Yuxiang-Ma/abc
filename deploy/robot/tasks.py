"""Shared task selection and data directory utilities for recording scripts."""

import os
import uuid
from datetime import datetime

TASKS = [
    "throw plastic bottles in bin",
    "set up chess pieces on the board",
]


def select_task() -> str:
    """Print a numbered menu of tasks, read user input, return selected task string.

    The user can pick a number or type a custom task name.
    """
    print("\n=== Task Selection ===")
    for i, task in enumerate(TASKS, 1):
        print(f"  {i}. {task}")
    print(f"  {len(TASKS) + 1}. (custom)")

    while True:
        choice = input("\nSelect task number (or type a custom task name): ").strip()
        if not choice:
            continue

        # Try interpreting as a number
        try:
            idx = int(choice)
            if 1 <= idx <= len(TASKS):
                selected = TASKS[idx - 1]
                print(f"Selected: {selected}")
                return selected
            elif idx == len(TASKS) + 1:
                custom = input("Enter custom task name: ").strip()
                if custom:
                    print(f"Selected: {custom}")
                    return custom
                print("Task name cannot be empty.")
                continue
            else:
                print(f"Please enter a number between 1 and {len(TASKS) + 1}.")
                continue
        except ValueError:
            # Not a number — treat the whole string as a custom task name
            print(f"Selected: {choice}")
            return choice


def task_to_collection_name(task: str) -> str:
    """Convert a task string to a filesystem-safe collection name.

    Example: ``"throw plastic bottles in bin"`` -> ``"throw_plastic_bottles_in_bin"``
    """
    return task.strip().replace(" ", "_")


def generate_recording_id() -> str:
    """Generate a 16-character lowercase hexadecimal recording ID."""
    return uuid.uuid4().hex[:16]


def prompt_session_tag() -> str:
    """Prompt for an optional session tag (e.g. 'cage_a').

    Returns the tag string, or empty string if skipped.
    """
    tag = input("\nSession tag (optional, e.g. 'cage_a'): ").strip()
    if tag:
        # Sanitise: replace spaces with underscores
        tag = tag.replace(" ", "_")
        print(f"Session tag: {tag}")
    return tag


def recording_filename(
    recording_id: str, collection_name: str, session_tag: str = ""
) -> str:
    """Build the H5 filename for a recording.

    Without tag: ``250410-1731-throw_plastic_bottles_in_bin-a1b2c3d4e5f67890.h5``
    With tag:    ``250410-1731-cage_a-throw_plastic_bottles_in_bin-a1b2c3d4e5f67890.h5``
    """
    date_prefix = datetime.now().strftime("%y%m%d-%H%M")
    if session_tag:
        return f"{date_prefix}-{session_tag}-{collection_name}-{recording_id}.h5"
    return f"{date_prefix}-{collection_name}-{recording_id}.h5"


def get_data_dir(subdir: str, cli_override: str | None = None) -> str:
    """Return the data directory for recordings.

    If the ``ABC_DATA_DIR`` environment variable is set, returns
    ``$ABC_DATA_DIR/<subdir>/`` (creating it if needed).  Otherwise returns
    *cli_override* (if provided) or a sensible default (``./data/<subdir>``).
    """
    env = os.environ.get("ABC_DATA_DIR")
    if env:
        path = os.path.join(env, subdir)
        os.makedirs(path, exist_ok=True)
        return path
    if cli_override:
        return cli_override
    return os.path.join("data", subdir)
