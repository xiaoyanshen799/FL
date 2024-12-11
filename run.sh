#!/bin/bash

echo "Starting server"
python -m fedprox.server &
sleep 60  # Sleep for 3s to give the server enough time to start

# set (seq 0 clients-1)
for i in $(seq 0 30); do
    echo "Starting client $i"
    python -m fedprox.client partition_id=$i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
