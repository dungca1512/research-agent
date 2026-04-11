#!/bin/bash
# Run once on the Oracle Cloud VM to set up the server
# Usage: bash setup-server.sh
set -e

echo "=== Research Agent — Server Setup ==="

# 1. System update
sudo apt update && sudo apt upgrade -y

# 2. Install Docker
if ! command -v docker &> /dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker $USER
  echo "Docker installed. You may need to log out and back in."
fi

# 3. Install Docker Compose plugin
sudo apt install -y docker-compose-plugin
docker compose version

# 4. Mount persistent block volume (run only once after attaching volume in Oracle console)
# Check if /dev/sdb exists and is not yet mounted
if [ -b /dev/sdb ] && ! mountpoint -q /data; then
  echo "Formatting and mounting /dev/sdb to /data..."
  sudo mkfs.ext4 /dev/sdb
  sudo mkdir -p /data
  sudo mount /dev/sdb /data
  echo '/dev/sdb /data ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
fi

sudo mkdir -p /data
sudo chown -R $USER:$USER /data
echo "/data ready."

# 5. Clone repo
if [ ! -d /opt/research-agent ]; then
  sudo git clone https://github.com/$GITHUB_REPO /opt/research-agent
  sudo chown -R $USER:$USER /opt/research-agent
fi

# 6. Setup .env
if [ ! -f /opt/research-agent/.env ]; then
  cp /opt/research-agent/.env.example /opt/research-agent/.env
  echo ""
  echo "IMPORTANT: Edit /opt/research-agent/.env and add your API keys:"
  echo "  GOOGLE_API_KEY=..."
  echo "  TAVILY_API_KEY=..."
  echo ""
fi

# 7. Open firewall ports (Ubuntu ufw)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp
sudo ufw --force enable

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Edit /opt/research-agent/.env with your API keys"
echo "  2. cd /opt/research-agent && docker compose up -d"
echo "  3. Access UI at http://$(curl -s ifconfig.me):8080"
