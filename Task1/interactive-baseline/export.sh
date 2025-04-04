#!/bin/bash

./build.sh

docker save sw_infer | gzip -c > interactive_baseline.tar.gz
