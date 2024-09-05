#!/bin/zsh
if [ $# -eq 0 ];
  then
    tag=$(uuidgen)
  else
    tag=$1
fi

docker buildx build --platform=linux/amd64,linux/arm64 --tag=jestoncolelewis/f1-predictor:$tag .
docker push jestoncolelewis/f1-predictor:$tag
docker run -p 80:8501 jestoncolelewis/f1-predictor:$tag