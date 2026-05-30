#!/bin/bash
# Setup device permissions for OpenAvatarChat on Raspberry Pi
# Run with sudo: sudo bash scripts/setup-permissions.sh

set -e

USER_NAME="${1:-cyk}"

echo "=== Adding $USER_NAME to required groups ==="
usermod -aG dialout,video "$USER_NAME"

echo "=== Creating udev rules ==="
cat > /etc/udev/rules.d/99-openavatarchat.rules << 'EOF'
# Serial port for MCU communication
KERNEL=="ttyAMA0", MODE="0666"

# USB camera
SUBSYSTEM=="video4linux", MODE="0666"
EOF

echo "=== Reloading udev ==="
udevadm control --reload-rules
udevadm trigger

echo "=== Enabling linger for $USER_NAME ==="
loginctl enable-linger "$USER_NAME"

echo "=== Done ==="
echo "Please reboot for group changes to take full effect."
echo "  sudo reboot"
