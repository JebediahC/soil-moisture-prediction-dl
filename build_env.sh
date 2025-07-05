#!/bin/bash

# to write requirements.txt
write_requirements() {
    pip freeze > requirements.txt
    conda list --explicit > environment.yml
    echo "requirements.txt and environment.yml have been created with the current environment's packages."
}

# write_requirements # uncomment to run

# to install the requirements
install_requirements() {
    if [ -f requirements.txt ]; then
        echo "Installing requirements from requirements.txt..."
        pip install -r requirements.txt
    else
        echo "requirements.txt not found. Skipping installation."
    fi
}

# install_requirements # uncomment to run