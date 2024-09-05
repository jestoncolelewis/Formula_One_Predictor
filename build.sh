#!/bin/zsh
if [ $# -eq 0 ];
  then
    tag=$(uuidgen)
  else
    tag=$1
fi

docker image build -t jestoncolelewis/f1-predictor:$tag .
docker run -p 80:8501 jestoncolelewis/f1-predictor:$tag