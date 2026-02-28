#!/bin/bash
# Starts a virtual display (Xvfb) and serves it via noVNC in the browser.
# Usage: ./start_vnc.sh
# Then open http://localhost:6080/vnc.html on your Mac.

set -e

# Install dependencies if missing
if ! command -v Xvfb &> /dev/null || ! command -v x11vnc &> /dev/null || ! command -v websockify &> /dev/null; then
  echo "[start_vnc] Installing Xvfb, x11vnc, novnc..."
  apt update -qq && apt install -y xvfb x11vnc novnc
fi

# Kill any existing instances
pkill -f "Xvfb :99" 2>/dev/null || true
pkill -f "x11vnc"   2>/dev/null || true
pkill -f "websockify" 2>/dev/null || true

# Start virtual framebuffer
echo "[start_vnc] Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1920x1080x24 &

# Give Xvfb a moment to start
sleep 1

# Start VNC server on port 5900 pointing at the virtual display
echo "[start_vnc] Starting x11vnc on port 5900..."
x11vnc -display :99 -nopw -forever -rfbport 5900 -quiet &

# Start noVNC websocket proxy on port 6080
echo "[start_vnc] Starting noVNC on port 6080..."
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo ""
echo "[start_vnc] Done. Open http://localhost:6080/vnc.html in your browser."
