# Text Semantic Embedding Server

This project contains a Dockerfile and Python code to run a FastAPI server that computes text semantic embeddings.

## Requirements

- Docker
- Python 3.7+

## Usage

To build the Docker image, run the following command:

```bash
docker build -t text-semantic-embedding:0.0 -f Dockerfile.cpu
```

To run the server in a Docker container, execute:
```bash
docker run --rm -it --name text-semantic-embedding -v path/to/transformers_cache:/home/solver/transformers_cache -p 8000:8000 text-semantic-embedding:0.0 start-services --port 8000 --hostname '0.0.0.0' --mounting_path '/'
```
This will start the server on `localhost:8000` with a mounted directory for the transformers cache.

## Dockerfile

The `Dockerfile` starts from the base image `python:3.7-slim-stretch` and installs the following dependencies:

- tzdata
- dialog
- apt-utils
- gcc
- pkg-config
- git
- curl
- build-essential
- libpoppler-cpp-dev
- wget
- unzip
- cmake

It then creates a new user and virtual environment, upgrades and installs Python dependencies, and copies the project code into the container.

## Server Code

The server code is written in Python and uses the following libraries:

- FastAPI
- uvicorn
- aiozmq
- loguru
- sentence-transformers
- spacy
- networkx

The `APIServer` class defines the routes for the API (`/` and `/compute_embedding`) and handles incoming requests. The `start()` method runs the server using `uvicorn`.

The `handle_compute_embedding()` method handles requests to compute text embeddings using a pre-trained vectorizer. It uses `asyncio` to manage multiple requests concurrently, and `aiozmq` to communicate with the vectorizer over ZeroMQ sockets.

The server can be started by calling the `start_server()` function, which creates an instance of `APIServer` and starts the server.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
