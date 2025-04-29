#!/bin/bash

./build.sh

docker save mast_baseline | gzip -c > mast_baseline.tar.gz
