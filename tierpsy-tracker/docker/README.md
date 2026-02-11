# Build a docker image for tierpsy

### Build:
**From tierpsy-tracker folder**:
``` bash
docker build -t tierpsy-tracker . -f docker/Dockerfile
```

### Run:
``` bash
./run_tierpsy_docker.sh
```
(look inside the script for details)

### Tag:
``` bash
docker tag tierpsy-tracker tierpsy/tierpsy-tracker
```
Without specifying a tag, this is the same as adding `:latest`.
To manually pick a tag, e.g. 1.5.2, use
`docker tag tierpsy-tracker tierpsy/tierpsy-tracker:1.5.2`

### Publish
``` bash
docker push tierpsy/tierpsy-tracker
```