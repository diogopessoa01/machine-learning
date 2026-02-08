#1/bin/bash

sudo docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined tensorflow-rocm:2.0.0 02_cnn.py 
