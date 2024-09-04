#!/bin/zsh
if [ $# -eq 0 ];
  then
    tag=$(uuidgen)
  else
    tag=$1
fi

docker image build --platform linux/amd64 -t jestoncolelewis/f1-predictor:$tag .
docker run -p 8080:8080 f1-predictor:$tag