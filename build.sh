#!/bin/zsh
if [ $# -eq 0 ];
  then
    tag=$(uuidgen)
  else
    tag=$1
fi

docker image build -t jestoncolelewis/f1-predictor:$tag .
docker run jestoncolelewis/f1-predictor:$tag