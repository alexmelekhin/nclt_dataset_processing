#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

ARCH=`uname -m`

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
cd $BASE_PATH/..

if [ "$ARCH" == "x86_64" ] 
then
    echo "Building for ${orange}${ARCH}${reset_color}"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

docker build . \
    -f $BASE_PATH/Dockerfile \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t nclt_dataset_processing