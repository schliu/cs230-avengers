# cs230-avengers

Prepare the docker container. The container file is way over the file limit, so it will have to stay on our server.
```
cat container.tar | docker import - openface:latest
```

Start docker
```
docker run -p 9000:9000 -p 8000:8000 -t -i openface:latest /bin/bash
```

Get docker container id
```
docker ps
```

Copy from local data directory to docker
```
docker cp data <container-id>:/root/
```

Run classifier
```
./classifier.py infer model/classifier.pkl data/test/<trailer-name>/* > output/<trailer-name>.txt
```

Save output back to server
```
docker cp <container-id>:/root/output output
```

Export docker container (from another terminal)
```
docker export <container-id> > container.tar
```

Stop docker
```
exit
```
