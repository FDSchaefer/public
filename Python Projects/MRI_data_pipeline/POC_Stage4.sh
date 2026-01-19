# POC_Stage4.sh - Demonstration of using the pipeline in a docker container
# to allow for seamless use in a production enviorment

# Build
docker build -t skull-stripper .

# Run in watch mode (default)
docker compose up

# View logs
docker logs -f skull-stripper

# Stop container
docker stop skull-stripper