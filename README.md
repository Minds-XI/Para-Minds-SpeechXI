# ASR-Service
## How-To-Run
run the script using this command
```bash
uv run server.py \
  --server grpc.nvcf.nvidia.com:443 \
  --use-ssl \
  --metadata function-id "1598d209-5e27-4d3c-8079-4751568b1081" \
  --metadata "authorization" "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC" \
  --language-code en-US \
  --stop-threshold 0.6 \
  --stop-history-eou 0 \
  --stop-threshold-eou 0
```