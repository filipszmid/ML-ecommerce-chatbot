#!/usr/bin/env bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <base_model> <adapter_path>"
    echo "Example: $0 llama3.1:8b data/runs/llm_finetuning/run_123/adapter"
    exit 1
fi

BASE_MODEL="$1"
ADAPTER_PATH="$(realpath --relative-to=. "$2")"
MODEL_NAME="finetuned-${BASE_MODEL//:/_}"

echo "Creating Modelfile for $MODEL_NAME..."
sed -e "s|{{BASE_MODEL}}|$BASE_MODEL|g" \
    -e "s|{{ADAPTER_PATH}}|$ADAPTER_PATH|g" \
    infra/llm/Modelfile.finetuned.template > Modelfile.finetuned

# Create a temporary model.safetensors symlink if it doesn't exist
# Ollama client expects 'model.safetensors' in the directory when importing
CREATED_SYMLINK=false
if [ ! -f "$ADAPTER_PATH/model.safetensors" ]; then
    ln -s adapter_model.safetensors "$ADAPTER_PATH/model.safetensors"
    CREATED_SYMLINK=true
fi

echo "Building Ollama model from Modelfile..."
# Note: Ollama must be running locally or via docker port 11434
# If inside docker, we use the host network or standard url
ollama create "$MODEL_NAME" -f Modelfile.finetuned

# Cleanup
rm -f Modelfile.finetuned
if [ "$CREATED_SYMLINK" = true ]; then
    rm -f "$ADAPTER_PATH/model.safetensors"
fi

mkdir -p data/llm
cat > data/llm/latest_finetuned_ollama.json <<EOF
{
  "adapter_path": "$ADAPTER_PATH",
  "base_model": "$BASE_MODEL",
  "finetuned_ollama_model": "$MODEL_NAME"
}
EOF

echo "Model '$MODEL_NAME' created successfully."
echo "You can now run it with: ollama run $MODEL_NAME"
echo "Run ADK Web with it: make adk-web-finetuned"
