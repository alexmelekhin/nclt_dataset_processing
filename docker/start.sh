#!/bin/bash

if [ $# != 1 ]; then
  echo "Usage: 
        bash start.sh [DATASET_DIR]
       "
  exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    realpath -m "$PWD"/"$1"
  fi
}

DATASET_DIR=$(get_real_path "$1")

if [ ! -d $DATASET_DIR ]
then
    echo "error: DATASET_DIR=$DATASET_DIR is not a directory."
exit 1
fi

ARCH=`uname -m`

orange=`tput setaf 3`
reset_color=`tput sgr0`

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
cd $BASE_PATH

echo "Running on ${orange}${ARCH}${reset_color}"

docker run -it -d --rm \
    --privileged \
    --name ${USER}_nclt_dataset_processing \
    --net host \
    --ipc host \
    -v $BASE_PATH/..:/home/docker_nclt/nclt_dataset_processing: rw \
    -v $DATASET_DIR:/home/docker_nclt/Dataset:rw \
    nclt_dataset_processing