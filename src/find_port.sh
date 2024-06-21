#!/bin/bash

# Function to check if a port is available on the specified IP
is_port_available() {
    local ip=$1
    local port=$2
    # Attempt to listen on the specified port and IP. Use & to background the process.
    nc -l $ip $port &> /dev/null &

    # Capture the PID of the background process
    local pid=$!
    # Wait a short moment to ensure nc had time to bind to the port
    sleep 0.1

    # Check if nc is still running. If so, the port was available.
    if kill -0 $pid &> /dev/null; then
        # Kill the background nc process
        kill $pid &> /dev/null
        return 0  # True, port is available
    else
        return 1  # False, port is not available
    fi
}

# Function to find an available port on the specified IP
find_available_port() {
    local ip=$1
    local base_port=$2
    local max_port=$3

    for ((port=base_port; port<=max_port; port++)); do
        if is_port_available $ip $port; then
            echo $port
            return
        fi
    done
    echo "No available port between $base_port and $max_port for $ip." >&2
    return 1
}
