#!/bin/bash
# One-shot bring-up of the React YAM station: CAN names + link up (once, with sudo), followers, probe.
#   bash deploy/scripts/bringup_react.sh install-udev   # once per machine (asks for sudo): persistent names + auto-up
#   bash deploy/scripts/bringup_react.sh                # every session: check, (re)start followers, print arm state
set -e
cd "$(dirname "$0")/../.."
export ROBOT_PROFILE=${ROBOT_PROFILE:-react_yam_config}
L_SERIAL=2097378145465006; R_SERIAL=2065376445465009   # verified 2026-09-23: left arm = ...006
RULE=/etc/udev/rules.d/90-react-yam-can.rules

if [ "$1" = "install-udev" ]; then
  # rename by USB serial AND bring the link up at plug time, so later sessions need no sudo
  sudo tee $RULE >/dev/null <<EOF
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="$L_SERIAL", NAME="can_l_foll", RUN+="/bin/sh -c 'ip link set can_l_foll up type can bitrate 1000000'"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="$R_SERIAL", NAME="can_r_foll", RUN+="/bin/sh -c 'ip link set can_r_foll up type can bitrate 1000000'"
EOF
  sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=net --action=add
  sleep 1; ip -br link | grep -E "can_[lr]_foll" || echo "interfaces not renamed yet: unplug and replug both CANable adapters, then rerun without arguments"
  exit 0
fi

echo "== CAN"; for iface in can_l_foll can_r_foll; do
  if ! ip link show "$iface" >/dev/null 2>&1; then echo "  $iface missing -> run: bash $0 install-udev  (then replug the adapters)"; exit 1; fi
  state=$(ip -br link show "$iface" | awk '{print $2}')
  if [ "$state" != "UP" ] && [ "$state" != "UNKNOWN" ]; then echo "  $iface is $state -> bringing up (sudo)"; sudo ip link set "$iface" up type can bitrate 1000000; fi
  echo "  $iface $(ip -br link show "$iface" | awk '{print $2}')"
done
echo "== followers (profile $ROBOT_PROFILE, no leaders, no cameras; motors hold, nothing moves)"
if pgrep -f "deploy.robot.followers.yam_follower" >/dev/null; then echo "  already running"; else
  nohup uv run deploy/robot/scripts/run_followers.py > /tmp/react_followers.log 2>&1 &
  echo "  started (pid $!, log /tmp/react_followers.log); waiting for state..."; sleep 8
fi
echo "== arm state (read-only)"; uv run deploy/robot/scripts/probe_yam_state.py --seconds 3
echo "== done. Stop the followers with: pkill -INT -f deploy.robot.followers.yam_follower   (they move to the zero pose over 2 s, then power off)"
