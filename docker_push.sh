#!/bin/bash
docker login docker.internal.starwit-infra.de
docker push docker.internal.starwit-infra.de/sae-anomaly-detection:$(poetry version --short)