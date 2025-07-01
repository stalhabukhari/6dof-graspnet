#!/bin/bash

DATADIR=$1

# # ----- for GUI
# VNC_PORT=5900
# VNC_PASSWORD=112358

# PARAMS=
# while (( "$#" )); do
#   case "$1" in
#     -p|--vnc-port)
#       VNC_PORT=$2
#       shift 2
#       ;;
#     -pw|--vnc-password)
#       VNC_PASSWORD=$2
#       shift 2
#       ;;
#     --) # end argument parsing
#       shift
#       break
#       ;;
#     -*|--*=) # unsupported flags
#       echo "Error: Unsupported flag $1" >&2
#       exit 1
#       ;;
#     *) # preserve positional arguments
#       PARAMS="$PARAMS $1"
#       shift
#       ;;
#   esac
# done

# xhost +local:root
# docker run -it --rm --gpus all --net=host -p $VNC_PORT:5900 -e VNC_PASSWORD=$VNC_PASSWORD \
#     --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="QT_X11_NO_MITSHM=1" \
#     --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
#     -v "$(pwd)":"/code-dir" --name 6dofgraspnet 6dofgraspnet-image:latest /bin/bash

# ------- without GUI -------
docker run -it --rm --gpus all -v "$(pwd)":"/code-dir" -v "$DATADIR":"/data-dir" \
  --name 6dofgraspnet 6dofgraspnet-image:latest /bin/bash
