#!/usr/bin/env bash

# change the value between quotes to change what folder in your computer tierpsy
# will have access to.
# If you don't need either the local folder or the network folder, you can leave
# them as empty strings.
#
# e.g :
# Give Tierpsy access to the Desktop
# 	local_folder_to_mount="$HOME/Desktop/"
# Give Tierpsy access to a hypothetical /DATA folder
# 	local_folder_to_mount="/DATA"
# Don't give Tierpsy access to any local folder
#   local_folder_to_mount=""
#
# Give Tierpsy access to a network share:
# assuming the share has been mounted already,
# you just need to use its path on your mac.
# Usually, that starts with "/Volumes"
#
# e.g :
# Give Tierpsy access to a shared named "my_network_share$" mounted in "/Volumes"
# 	network_folder_to_mount="/Volumes/my_network_share$"
# Don't give Tierpsy access to any network shares
#   network_folder_to_mount=""
#
# NOTE: don't add white spaces to the left or right of the equal sign:
# local_folder_to_mount="$HOME" works, while
# local_folder_to_mount = "$HOME" does not
#

local_folder_to_mount="$HOME"
network_folder_to_mount=""

############# please do not modify below this line ##########

# shortcut to make sure the X server is running
xlogo &
xlogo_pid=`pgrep -f xlogo`
while [[ -z "$xlogo_pid" ]];
do
    echo "Starting X server, will take a few seconds..."
    xlogo_pid=`pgrep -f xlogo`
    sleep 2
done

disown $xlogo_pid
kill $xlogo_pid

# add localhost to the allowed connections
xhost +localhost

# is docker running? it is, then docker stats does not return an error
if (! docker ps &> /dev/null );
then
    echo "Starting Docker, this will take a few seconds..."
    open -a Docker
    while (! docker ps &> /dev/null ); do
        # Docker takes a few seconds to initialize
		echo "Docker still starting up, will check again in a few seconds"
        sleep 2
    done
else
    echo "Docker already running"
fi

# create array of arguments for docker run.
# this allow to optionally mount local/network folders per user's wish
docker_arguments=("-it" "--rm")
docker_arguments+=("-e DISPLAY=host.docker.internal:0")
docker_arguments+=("--sysctl net.ipv4.tcp_keepalive_intvl=30")
docker_arguments+=("--sysctl net.ipv4.tcp_keepalive_probes=5")
docker_arguments+=("--sysctl net.ipv4.tcp_keepalive_time=100")
docker_arguments+=("--hostname tierpsydocker")

# if local folder not empty, add to array
if [[ ! -z "$local_folder_to_mount" ]];
then
docker_arguments+=("-v ${local_folder_to_mount}:/DATA/local_drive")
fi

# if network folder is not empty, add to array
if [[ ! -z "$network_folder_to_mount" ]];
then
docker_arguments+=("-v ${network_folder_to_mount}:/DATA/network_drive")
fi

# launch using the parameters in the array
docker run ${docker_arguments[@]} tierpsy/tierpsy-tracker
