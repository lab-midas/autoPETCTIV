#!/bin/bash

./build.sh

docker save interactive_baseline | gzip -c > interactive_baseline.tar.gz
