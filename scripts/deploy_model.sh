#!/bin/bash
set -e

source ../my_env/bin/activate 2>/dev/null || true

MODEL_URI=$(cat best_model_uri.txt)
echo "Deploying model from: $MODEL_URI"

export BUILD_ID=dontKillMe
export JENKINS_NODE_COOKIE=dontKillMe

pkill -f "mlflow models serve" || true

nohup mlflow models serve -m "$MODEL_URI" -p 5003 --no-conda > mlflow_serve.log 2>&1 &

sleep 5 
echo "Service started on port 5003"
