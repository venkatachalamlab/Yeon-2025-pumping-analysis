# change the value between quotes to change what folder in your computer tierpsy
# will have access to.
#
# e.g :
# Give Tierpsy access to the D:\ drive
# 	$local_folder_to_mount = "d:/"
# Give Tierpsy access to the C:\ drive
# 	$local_folder_to_mount = "c:/"
#
# Give Tierpsy access to a network share,
# assuming the share is on a server at 192.168.1.20, called worms$,
# and your username is worm_scientist@awesome_institute.ac.uk
# 	$network_folder_to_mount = "//192.168.1.20/worms$"
# 	$network_user_at_domain = "worm_scientist@awesome_institute.ac.uk"

$local_folder_to_mount = "d:/"
$network_folder_to_mount = $null
$network_user_at_domain = $null


############# please do not modify below this line ##########

# start the X server
if((get-process "vcxsrv" -ErrorAction SilentlyContinue) -eq $Null){
	echo "VcXsrv not running, starting now"
	& $env:ProgramFiles\VcXsrv\vcxsrv.exe -multiwindow -wgl -ac -clipboard
}else{
	echo "VcXsrv already running"
}


#start docker
$isdockerup = docker ps 2>&1 | out-null
if($?){
	echo "Docker already running"
}else{
	echo "Starting Docker Desktop, this will take a few seconds"
	Start-Process $env:ProgramFiles\Docker\Docker\"Docker Desktop.exe" | Out-Null
}


# wait until docker is up, then continue
for ($num = 0 ; $num -le 20; $num++){
	$isdockerup = docker ps 2>&1 | out-null
	if ($?){
	    echo "Docker is running"
		# wait some more just in case
		Start-Sleep -Seconds 2
		break
	}else{
	    if ($num -gt 0){
			echo "Docker still starting up, will check again in a few seconds"
			}
		# wait before next
		Start-Sleep -Seconds 5
	}
}


# check if the volume we want to mount already exists

$volume_query_result = $null
$volume_query_result = docker volume ls -q --filter name=tierpsy_network_share
if ($volume_query_result -ne $null){

	echo "Docker volume tierpsy_network_share already exists."
	echo "if you want this to change, remove it using the docker interface and restart this launcher"

}elseif($network_folder_to_mount -eq $null){

	echo "No network share specified, none will be mounted"

}else{
	echo "Trying to mount $network_folder_to_mount as tierpsy_network_share with user $network_user_at_domain"
	$netpw = Read-Host "Please insert your network password: " -AsSecureString
	$plainpw =[Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($netpw))

	# create a docker volume to mount a network drive
	docker volume create `
		--driver local `
		--opt type=cifs `
		--opt device=$network_folder_to_mount `
		--opt o=user=$network_user_at_domain,password=$plainpw,file_mode=0777,dir_mode=0777 `
		tierpsy_network_share
}

# start the docker image
& docker run `
	-it --rm `
	-e DISPLAY=host.docker.internal:0 `
	-v tierpsy_network_share:/DATA/network_drive `
	-v ${local_folder_to_mount}:/DATA/local_drive `
	--hostname tierpsydocker `
	--sysctl net.ipv4.tcp_keepalive_intvl=30 `
	--sysctl net.ipv4.tcp_keepalive_probes=5 `
	--sysctl net.ipv4.tcp_keepalive_time=100 `
    tierpsy/tierpsy-tracker

Read-Host -Prompt "Press Enter to exit"