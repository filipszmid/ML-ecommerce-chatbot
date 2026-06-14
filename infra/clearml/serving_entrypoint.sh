#!/bin/sh
set -eu

mode="${1:-inference}"
task_file="${CLEARML_SERVING_TASK_FILE:-/app/data/clearml/serving_task_id}"
env_file="${CLEARML_SERVING_ENV_FILE:-/app/data/clearml/serving.env}"
service_name="${CLEARML_SERVING_SERVICE_NAME:-ml-ecommerce-chatbot-serving}"
project_name="${CLEARML_SERVING_PROJECT:-ML Ecommerce Chatbot/Serving}"
base_url="${CLEARML_SERVING_BASE_URL:-http://localhost:8082/serve}"
kafka_metric_server="${CLEARML_SERVING_KAFKA_METRIC_SERVER:-clearml-serving-kafka:9092}"
metric_log_freq="${CLEARML_SERVING_METRIC_LOG_FREQ:-1.0}"
api_host="${CLEARML_API_HOST:-http://clearml-apiserver:8008}"

wait_for_clearml_api() {
  echo "Waiting for ClearML API at ${api_host}..."
  for i in $(seq 1 60); do
    if curl -fsS "${api_host}/debug.ping" >/dev/null 2>&1; then
      echo "ClearML API is ready."
      return 0
    fi

    if [ "$i" -eq 60 ]; then
      echo "ClearML API did not become ready in time."
      return 1
    fi

    sleep 2
  done
}

load_serving_env() {
  if [ -s "$env_file" ]; then
    set -a
    . "$env_file"
    set +a
  fi
}

wait_for_serving_env() {
  for i in $(seq 1 60); do
    if [ -s "$env_file" ]; then
      load_serving_env
      return 0
    fi

    if [ "$i" -eq 60 ]; then
      echo "ClearML Serving env file was not created: ${env_file}"
      return 1
    fi

    sleep 2
  done
}

create_serving_task() {
  mkdir -p "$(dirname "$task_file")"
  mkdir -p "$(dirname "$env_file")"

  if [ ! -s "$env_file" ] && [ -s "$task_file" ]; then
    task_id="$(cat "$task_file")"
    printf "CLEARML_SERVING_TASK_ID=%s\n" "$task_id" > "$env_file"
  fi

  if [ -s "$env_file" ]; then
    load_serving_env
    echo "Using ClearML Serving service id: ${CLEARML_SERVING_TASK_ID}"
    return 0
  fi

  echo "Creating ClearML Serving service ${service_name}"
  output="$(clearml-serving --yes create --name "$service_name" --project "$project_name" --tags ecommerce classical-ml serving 2>&1)"
  printf "%s\n" "$output"
  task_id="$(printf "%s\n" "$output" | sed -nE 's/.*id=([a-fA-F0-9]{32}).*/\1/p' | tail -n 1)"

  if [ -z "$task_id" ]; then
    echo "Could not parse ClearML Serving task id from clearml-serving output."
    return 1
  fi

  printf "%s\n" "$task_id" > "$task_file"
  printf "CLEARML_SERVING_TASK_ID=%s\n" "$task_id" > "$env_file"
  load_serving_env
  echo "Saved ClearML Serving task id to ${task_file}"
}

configure_serving_task() {
  load_serving_env
  if [ -z "${CLEARML_SERVING_TASK_ID:-}" ]; then
    echo "CLEARML_SERVING_TASK_ID is missing."
    return 1
  fi

  clearml-serving --yes --id "$CLEARML_SERVING_TASK_ID" config \
    --base-serving-url "$base_url" \
    --kafka-metric-server "$kafka_metric_server" \
    --metric-log-freq "$metric_log_freq"
}

wait_for_clearml_api

case "$mode" in
  inference)
    create_serving_task
    configure_serving_task
    exec python -m uvicorn clearml_serving.serving.main:app --host 0.0.0.0 --port 8082
    ;;
  statistics)
    wait_for_serving_env
    configure_serving_task
    exec python -m clearml_serving.statistics.main
    ;;
  *)
    echo "Unknown ClearML Serving mode: ${mode}"
    exit 1
    ;;
esac
