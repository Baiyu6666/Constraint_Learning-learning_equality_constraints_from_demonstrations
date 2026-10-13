# Docker

The remote runner uses Docker on the server. Build the image from the repository
root, either locally for testing or on the server after syncing:

```bash
docker/build.sh
```

Defaults can be overridden:

```bash
IMAGE=learneq:cuda12.8 CONTAINER_HOME=/home/learneq docker/build.sh
```

For an interactive shell:

```bash
PROJECT_DIR=$PWD RUNS_DIR=$HOME/runs docker/shell.sh
```

The container expects the repository at `/workspace` and writes experiment
artifacts to `/runs`.
