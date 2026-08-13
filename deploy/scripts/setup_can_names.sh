#!/bin/bash
set -e

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo)."
  exit 1
fi

USAGE="Usage: $0 [--followers-only]"
FOLLOWERS_ONLY=false

if [[ "$1" == "--followers-only" ]]; then
    FOLLOWERS_ONLY=true
elif [[ -n "$1" ]]; then
    echo "$USAGE"
    exit 1
fi

echo "Setting up persistent CAN interface names..."

# Read CAN USB serials from robot_profile.py via Python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -z "$ROBOT_PROFILE" ]; then
    echo "Error: ROBOT_PROFILE env var is not set."
    echo "Set it to one of: bbox_config, cbox_config"
    echo "Note: sudo does not preserve env vars. Use: sudo -E bash $0"
    exit 1
fi

read_can_serial() {
    local field="$1"
    python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from deploy.robot.config import PROFILES
cfg = PROFILES['$ROBOT_PROFILE'].can_devices
print(getattr(cfg, '$field'))
"
}

# If the config value is an interface name (e.g. "can0"), resolve it to the
# USB serial number via udevadm so users don't have to look up serials manually.
resolve_serial() {
    local value="$1"
    local label="$2"

    if [ -z "$value" ]; then
        return
    fi

    if [[ "$value" =~ ^can[0-9]+$ ]]; then
        local sysfs_path="/sys/class/net/$value"
        if [ ! -d "$sysfs_path" ]; then
            echo "Error: Interface $value (for $label) not found" >&2
            exit 1
        fi
        local serial
        serial=$(udevadm info -a "$sysfs_path" 2>/dev/null \
            | grep 'ATTRS{serial}==' | head -1 \
            | sed 's/.*=="//;s/".*//')
        if [ -z "$serial" ]; then
            echo "Error: Could not find USB serial for $value (for $label)" >&2
            exit 1
        fi
        echo "  Resolved $label: $value -> serial $serial" >&2
        echo "$serial"
    else
        echo "$value"
    fi
}

echo "Using profile: $ROBOT_PROFILE"

CAN_L_LEAD=$(resolve_serial "$(read_can_serial can_l_lead)" "can_l_lead")
CAN_L_FOLL=$(resolve_serial "$(read_can_serial can_l_foll)" "can_l_foll")
CAN_R_LEAD=$(resolve_serial "$(read_can_serial can_r_lead)" "can_r_lead")
CAN_R_FOLL=$(resolve_serial "$(read_can_serial can_r_foll)" "can_r_foll")

# Build udev rules, skipping entries with no serial configured
add_rule() {
    local serial="$1"
    local name="$2"
    if [ -n "$serial" ]; then
        echo "SUBSYSTEM==\"net\", ACTION==\"add\", ATTRS{serial}==\"${serial}\", NAME=\"${name}\""
    fi
}

RULES=""

if [ "$FOLLOWERS_ONLY" = true ]; then
    echo "Mode: FOLLOWERS ONLY"
    RULES="$(add_rule "$CAN_L_FOLL" "can_l_foll")
$(add_rule "$CAN_R_FOLL" "can_r_foll")"
else
    echo "Mode: LEADERS and FOLLOWERS (Default)"
    RULES="$(add_rule "$CAN_L_LEAD" "can_l_lead")
$(add_rule "$CAN_L_FOLL" "can_l_foll")
$(add_rule "$CAN_R_LEAD" "can_r_lead")
$(add_rule "$CAN_R_FOLL" "can_r_foll")"
fi

# Strip blank lines from unconfigured entries
RULES=$(echo "$RULES" | sed '/^$/d')

# Create udev rules
# Note: Network interface names must be <= 15 characters.
echo "$RULES" | tee /etc/udev/rules.d/99-can-names.rules

echo "Bringing down existing CAN interfaces to allow renaming..."
for iface in $(ip -o link show | awk -F': ' '{print $2}' | grep -E '^can([0-9]+|_(l|r)_(lead|foll))'); do
    echo "Stopping $iface..."
    ip link set "$iface" down
done

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger --subsystem-match=net --action=add

# Wait for udev to process
udevadm settle

echo "Configuring and bringing up CAN interfaces..."
INTERFACES=""
if [ "$FOLLOWERS_ONLY" = true ]; then
    [ -n "$CAN_L_FOLL" ] && INTERFACES="$INTERFACES can_l_foll"
    [ -n "$CAN_R_FOLL" ] && INTERFACES="$INTERFACES can_r_foll"
else
    [ -n "$CAN_L_LEAD" ] && INTERFACES="$INTERFACES can_l_lead"
    [ -n "$CAN_L_FOLL" ] && INTERFACES="$INTERFACES can_l_foll"
    [ -n "$CAN_R_LEAD" ] && INTERFACES="$INTERFACES can_r_lead"
    [ -n "$CAN_R_FOLL" ] && INTERFACES="$INTERFACES can_r_foll"
fi

for iface in $INTERFACES; do
    if ip link show "$iface" > /dev/null 2>&1; then
        echo "Setting up $iface..."
        ip link set "$iface" type can bitrate 1000000
        ip link set "$iface" up
    else
        echo "Warning: Interface $iface not found. It might not be connected."
    fi
done

echo "Done. Active CAN interfaces:$INTERFACES"
echo "You can verify this with: ip link show"
