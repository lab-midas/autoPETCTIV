#!/bin/bash

./build.sh

docker save autopet_baseline | gzip -c > nnunet_baseline.tar.gz
