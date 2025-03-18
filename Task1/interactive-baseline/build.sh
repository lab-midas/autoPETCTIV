SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t sw_infer "$SCRIPTPATH" 
