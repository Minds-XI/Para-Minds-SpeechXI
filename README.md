
# ASR-Service

## How-To-Run

### 1. Set the Project Path as an Environment Variable

First, set the project path as an environment variable to ensure that the project directories are available for Python imports:

```bash
export PYTHONPATH=/mnt/data/projects/Para-Minds-SpeechXI
```

### 2. Run the Script

Use the following command to run the `server.py` script with the necessary parameters:

```bash
uv run server.py   --server grpc.nvcf.nvidia.com:443   --use-ssl   --metadata function-id "1598d209-5e27-4d3c-8079-4751568b1081"   --metadata "authorization" "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"   --language-code en-US   --stop-threshold 0.6   --stop-history-eou 0   --stop-threshold-eou 0
```

### Parameters:

- **`--server`**: The server address to connect to (e.g., `grpc.nvcf.nvidia.com:443`).
- **`--use-ssl`**: Enables SSL encryption for the connection.
- **`--metadata function-id`**: Specifies the function ID for the request.
- **`--metadata authorization`**: Adds the required API authorization header.
  - Replace `$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC` with your actual API key if you're running outside of NGC.
- **`--language-code`**: Set the language code, e.g., `en-US`.
- **`--stop-threshold`**: Threshold value for stopping the audio.
- **`--stop-history-eou`**: Defines the end-of-utterance (EOU) history stop criteria.
- **`--stop-threshold-eou`**: Defines the threshold value for the EOU stop.


### Project Dependency
1. Create env using uv
  ```bash
    uv venv
  ```
2. install prerequisites packages
  ```bash
  sudo apt update

  sudo apt install python3-dev portaudio19-dev

  sudo apt install build-essential python3-dev
  ```
3. sync from pyproject.toml
  ```bash
  uv sync
  ```

### Project Structure Methodology
```bash
entrypoint → transport/grpc → application → domain
                         └→ infrastructure(adapter) → core
```

# Run docker services
```bash
docker compose -f docker/docker-compose-kafka.yaml -f docker/docker-compose-mongodb.yaml -f docker/docker-compose-connect.yaml --env-file .env up
```
