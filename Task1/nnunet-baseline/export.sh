#!/bin/bash

./build.sh

docker save nnunet_baseline | gzip -c > nnunet_baseline.tar.gz
