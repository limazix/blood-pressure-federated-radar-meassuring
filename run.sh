#!/bin/bash

#export GRPC_POLL_STRATEGY=epoll1

echo "Starting server"
poetry run python scripts/main.py --is-global --is-federated &
sleep 3 # Sleep for 3s to give the server enough time to start

for i in `seq 1 30` # seq 1 30
do
    echo "Starting client $i"
    if [ $i -lt 10 ]
    then
        poetry run python scripts/main.py --is-federated --subject-id="GDN000$i" &
    else
        poetry run python scripts/main.py --is-federated --subject-id="GDN00$i" &
    fi
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait