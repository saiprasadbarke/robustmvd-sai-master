#!/bin/bash

if [ -z "$1" ]
then
   echo "Please speficy a target path to this script, e.g.: /setup_gmdepth.sh /path/to/gmdepth";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading GMDepth repository https://github.com/autonomousvision/unimatch to $TARGET"
mkdir -p "$1"

git clone https://github.com/autonomousvision/unimatch $TARGET

OLD_PWD="$PWD"
cd $TARGET

mkdir pretrained
cd pretrained
wget --no-check-certificate https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-scannet-d3d1efb5.pth
wget --no-check-certificate https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth
wget --no-check-certificate https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth
wget --no-check-certificate https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-demon-bd64786e.pth
wget --no-check-certificate https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-demon-a2fe127b.pth
wget --no-check-certificate https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-demon-7c23f230.pth

cd "$OLD_PWD"
echo "Done"
exit 0
