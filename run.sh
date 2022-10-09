#!/bin/bash

#export GRPC_POLL_STRATEGY=epoll1

echo "Starting simulation"
poetry run python scripts/main.py --is-federated &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait