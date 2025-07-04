# Create swap file
sudo fallocate -l 16G /swapfile

# Set permissions
sudo chmod 600 /swapfile

# Mark it as Swap space
sudo mkswap /swapfile

# Enable the Swap file
sudo swapon /swapfile

# Make It Persistent Across Reboots
# Add "/swapfile none swap sw 0 0" to the end of /etc/fstab
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# To check if its active run:
# free -h
