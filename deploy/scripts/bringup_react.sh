#!/bin/bash
# One-shot bring-up of the React YAM station: CAN names + link up (once, with sudo), followers, probe.
#   bash deploy/scripts/bringup_react.sh install-udev   # once per machine (asks for sudo): persistent names + auto-up
#   bash deploy/scripts/bringup_react.sh                # every session: check, (re)start followers, print arm state
#   bash deploy/scripts/bringup_react.sh stop           # stop the followers (zero pose, power off); needed after a CAN replug
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

if [ "$1" = "stop" ]; then
  PIDS=$(pgrep -f "deploy/robot/scripts/run_followers.p[y]" || true)
  if [ -z "$PIDS" ]; then echo "no followers running"; exit 0; fi
  echo "stopping followers $PIDS (arms move to the zero pose over 2 s, then power off -- keep clear)"
  kill -INT $PIDS; sleep 5
  LEFT=$(pgrep -f "deploy/robot/scripts/run_followers.p[y]" || true)
  if [ -n "$LEFT" ]; then echo "still alive after 5 s (stale after a CAN replug?): killing $LEFT"; kill -9 $LEFT; fi
  echo "stopped"; exit 0
fi

echo "== CAN"; for iface in can_l_foll can_r_foll; do
  if ! ip link show "$iface" >/dev/null 2>&1; then echo "  $iface missing -> run: bash $0 install-udev  (then replug the adapters)"; exit 1; fi
  state=$(ip -br link show "$iface" | awk '{print $2}')
  if [ "$state" != "UP" ] && [ "$state" != "UNKNOWN" ]; then echo "  $iface is $state -> bringing up (sudo)"; sudo ip link set "$iface" up type can bitrate 1000000; fi
  echo "  $iface $(ip -br link show "$iface" | awk '{print $2}')"
done
echo "== followers (profile $ROBOT_PROFILE, no leaders, no cameras; motors hold, nothing moves)"
# the follower processes' command line is run_followers.py (not the module name)
if pgrep -f "deploy/robot/scripts/run_followers.p[y]" >/dev/null; then
  echo "  already running (pids $(pgrep -f 'deploy/robot/scripts/run_followers.p[y]' | tr '\n' ' '))"
  echo "  if they predate a CAN replug they are stale: stop them first with  bash $0 stop"
else
  nohup uv run deploy/robot/scripts/run_followers.py > /tmp/react_followers.log 2>&1 &
  FPID=$!
  echo "  started (pid $FPID, log /tmp/react_followers.log); waiting for state..."
  for i in $(seq 1 20); do
    sleep 1
    if ! kill -0 $FPID 2>/dev/null; then echo "  follower process exited -- log:"; tail -20 /tmp/react_followers.log; exit 1; fi
  done
fi
echo "== arm state (read-only)"; uv run deploy/robot/scripts/probe_yam_state.py --seconds 3
echo "== done. Stop the followers with: bash $0 stop   (arms move to the zero pose over 2 s, then power off)"
