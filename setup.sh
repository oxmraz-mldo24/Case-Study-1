#!/bin/bash

VENV_DIR="venv"

#ensure package list
sudo add-apt-repository -y universe
#ensure python and requirements is installed
sudo apt install -qq -y python3-venv
sudo apt install -qq -y python3-pip
sudo apt install -y build-essential
sudo apt install -y gcc g++ 
sudo apt install -y screen


# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found. Creating a new one..."
  # Create a virtual environment
  python3 -m venv "$VENV_DIR"
  echo "Virtual environment created."

else
    echo "Virtual environment found."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment $VENV_DIR activated."

pip install --upgrade pip

if [ -d "Case-Study-1" ]; then
    echo "Repository folder 'Case-Study-1' already exists. Checking for updates."
    cd Case-Study-1
    if git pull | grep -q 'Already up to date.'; then
        echo "Repository is up to date. Proceeding with setup."

    else
        echo "Repository updated successfully. Proceeding to next step."
		git clone https://github.com/oxmraz-mldo24/Case-Study-1.git
	fi
else
    echo "Cloning repository..."
    if git clone https://github.com/oxmraz-mldo24/Case-Study-1.git; then
        echo "Repository cloned successfully. Proceeding to next step."
        cd Case-Study-1
    else
        echo "Failed to clone repository. Exiting."
        exit 1
    fi
fi
echo "Checking if http://127.0.0.1:7860 is running..."
if curl -s --head http://127.0.0.1:7860 | grep "200 OK" > /dev/null; then
    echo "URL is running.No further action required. Exiting."
    exit 0  # Exit script since the service is already running
else
    echo "URL is not running.Proceeding with setup."
    # Install dependencies and run the application
    pip install -r requirements.txt

    screen -S "app" -d -m bash -c 'python3 app.py'
fi
deactivate
exit 0