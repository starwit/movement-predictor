#!/bin/bash

docker build -t starwitorg/sae-anomaly-detection:$(poetry version --short) .