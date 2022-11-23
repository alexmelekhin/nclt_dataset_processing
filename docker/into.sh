#!/bin/bash

docker exec --user "docker_nclt" -it ${USER}_nclt_dataset_processing \
    /bin/bash -c "cd /home/docker_nclt; echo ${USER}_nclt_dataset_processing container; echo ; /bin/bash"