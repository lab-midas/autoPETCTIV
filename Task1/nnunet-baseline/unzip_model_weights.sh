#!/bin/bash

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
echo $SCRIPTPATH

unzip "weights.zip" 
# rm "weights.zip"