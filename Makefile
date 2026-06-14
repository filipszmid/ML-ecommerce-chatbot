SHELL := /bin/bash

API_HOST ?= 0.0.0.0
API_PORT ?= 8000
ADK_PORT ?= 8001
POSTGRES_USER ?= ml_chatbot
POSTGRES_PASSWORD ?= ml_chatbot
POSTGRES_DB ?= ml_ecommerce_chatbot
DATABASE_URL ?= postgresql+psycopg://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:5432/$(POSTGRES_DB)
API_URL ?= http://localhost:$(API_PORT)
ADK_WEB_URL ?= http://localhost:$(ADK_PORT)
ADK_SESSION_SERVICE_URI ?= postgresql+asyncpg://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:5432/$(POSTGRES_DB)
OLLAMA_BASE_URL ?= http://localhost:11434
OLLAMA_MODEL ?= llama3.1:8b
OLLAMA_SKIP_PULL ?= false
OLLAMA_DOCKER ?= true
OLLAMA_HOST_PORT ?= 11434
OLLAMA_STOP_SYSTEM_SERVICE ?= true
LLM_BASE_MODEL ?=
LLM_EPOCHS ?= 3
LLM_BATCH_SIZE ?= 4
LLM_MAX_SEQ_LENGTH ?= 512
LLM_MAX_TRAIN_SAMPLES ?=
FINETUNE_OLLAMA_MODEL ?= tinyllama:1.1b
FINETUNE_BASE_MODEL ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
FINETUNE_EPOCHS ?= 1
FINETUNE_BATCH_SIZE ?= 1
FINETUNE_MAX_SEQ_LENGTH ?= 256
FINETUNE_MAX_TRAIN_SAMPLES ?= 200
FINETUNE_CLEARML ?= true
FINETUNED_OLLAMA_MODEL ?= $(shell poetry run python -c 'import json; from pathlib import Path; path=Path("data/runs/llm_finetuning/latest_finetune.json"); print(json.loads(path.read_text()).get("finetuned_ollama_model") or "finetuned-tinyllama_1.1b") if path.exists() else print("finetuned-tinyllama_1.1b")' 2>/dev/null)
LLM_ADAPTER_PATH ?= $(shell find data/runs/llm_finetuning -path '*/adapter' -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $$2}')
FINETUNE_AZURE_FILE_ID ?=
FINETUNE_BEDROCK_DATA_URI ?=
FINETUNE_VERTEX_DATA_URI ?=
OPENAI_MODEL ?= gpt-4o
LLM_PROVIDER ?= openai
DATA_PATH ?= data/customer_purchase_data.csv
MODEL ?= xgboost
MODELS ?= all
SPLIT ?= test
MAX_EVALS ?= 20
CV_FOLDS ?= 3
SMOTE ?= true
SMOTE_K_NEIGHBORS ?= 5
CLEARML ?= false
MODEL_DIR ?=
SELECTION_PATH ?= data/runs/classical_ml/latest_selection.json
CLEARML_SERVING_SERVICE_NAME ?= ml-ecommerce-chatbot-serving
CLEARML_SERVING_PROJECT ?= ML Ecommerce Chatbot/Serving
CLEARML_SERVING_ENDPOINT ?= product-category
CLEARML_LLM_SERVING_ENDPOINT ?= product-category-llm
CLEARML_SERVING_TASK_FILE ?= data/clearml/serving_task_id
CLEARML_SERVING_ENV_FILE ?= data/clearml/serving.env
CLEARML_SERVING_BASE_URL ?= http://localhost:8082/serve
CLEARML_SERVING_KAFKA_METRIC_SERVER ?= clearml-serving-kafka:9092
CLEARML_SERVING_METRIC_LOG_FREQ ?= 1.0
CLEARML_SERVING_SERVICES ?= clearml-serving-zookeeper clearml-serving-kafka clearml-serving-inference clearml-serving-statistics
ENV_FILE ?= .env
DEV_PID_DIR := data/dev-pids
API_PID := $(DEV_PID_DIR)/api.pid
ADK_PID := $(DEV_PID_DIR)/adk.pid

-include $(ENV_FILE)

CLEARML_PROJECT_NAME ?= ML Ecommerce Chatbot/Product Category
CLEARML_API_HOST ?= http://localhost:8008
CLEARML_WEB_HOST ?= http://localhost:8080
CLEARML_FILES_HOST ?= http://localhost:8081
CLEARML_API_ACCESS_KEY ?= ml-ecommerce-local
CLEARML_API_SECRET_KEY ?= ml-ecommerce-local-secret

export CLEARML_ENABLED
export CLEARML_PROJECT_NAME
export CLEARML_API_HOST
export CLEARML_WEB_HOST
export CLEARML_FILES_HOST
export CLEARML_API_ACCESS_KEY
export CLEARML_API_SECRET_KEY
export CLEARML_ACCESS_KEY
export CLEARML_SECRET_KEY
export CLEARML_SERVING_SERVICE_NAME
export CLEARML_SERVING_PROJECT
export CLEARML_SERVING_ENDPOINT
export CLEARML_SERVING_BASE_URL
export CLEARML_SERVING_KAFKA_METRIC_SERVER
export CLEARML_SERVING_METRIC_LOG_FREQ

ADK_AGENTS_DIR := interface/chat

.PHONY: install
install:
	poetry install --with app,serving --no-root

.PHONY: install-dev
install-dev:
	poetry install --with app,serving,dev --no-root

.PHONY: help
help: ## Show this help message
	@echo "$(_BOLD)Personal Memory Module - Available Commands:$(_DEFAULT)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(_CYAN)%-20s$(_DEFAULT) %s\n", $$1, $$2}'

.PHONY: free-ollama-port
free-ollama-port:
	@set -e; \
	port="$(OLLAMA_HOST_PORT)"; \
	echo "Freeing Ollama host port $$port..."; \
	docker compose stop ollama >/dev/null 2>&1 || true; \
	if command -v docker >/dev/null 2>&1; then \
		container_ids="$$(docker ps --filter "publish=$$port" --format '{{.ID}}' 2>/dev/null || true)"; \
		ollama_container_ids="$$(docker ps --filter "ancestor=ollama/ollama" --format '{{.ID}}' 2>/dev/null || true)"; \
		container_ids="$$(printf '%s\n%s\n' "$$container_ids" "$$ollama_container_ids" | sed '/^$$/d' | sort -u)"; \
		if [ -n "$$container_ids" ]; then \
			echo "Stopping Docker Ollama container(s) using port $$port..."; \
			echo "$$container_ids" | xargs -r docker stop >/dev/null; \
		fi; \
	fi; \
	if [ "$(OLLAMA_STOP_SYSTEM_SERVICE)" = "true" ] && command -v systemctl >/dev/null 2>&1; then \
		if systemctl --user is-active --quiet ollama 2>/dev/null; then \
			echo "Stopping user Ollama service..."; \
			systemctl --user stop ollama; \
		fi; \
		if systemctl is-active --quiet ollama 2>/dev/null; then \
			echo "Stopping system Ollama service..."; \
			if [ "$$(id -u)" -eq 0 ]; then \
				systemctl stop ollama; \
			elif command -v sudo >/dev/null 2>&1; then \
				sudo systemctl stop ollama; \
			else \
				echo "Cannot stop system Ollama service because sudo is unavailable."; \
				exit 1; \
			fi; \
		fi; \
	fi; \
	pids=""; \
	if command -v lsof >/dev/null 2>&1; then \
		pids="$$(lsof -nP -tiTCP:"$$port" -sTCP:LISTEN 2>/dev/null || true)"; \
	fi; \
	if [ -z "$$pids" ] && command -v fuser >/dev/null 2>&1; then \
		pids="$$(fuser -n tcp "$$port" 2>/dev/null || true)"; \
	fi; \
	if [ -z "$$pids" ] && command -v ss >/dev/null 2>&1; then \
		pids="$$(ss -ltnp "sport = :$$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)"; \
	fi; \
	if [ -z "$$pids" ] && command -v pgrep >/dev/null 2>&1; then \
		pids="$$(pgrep -x ollama 2>/dev/null || true)"; \
	fi; \
	if [ -n "$$pids" ]; then \
		echo "Terminating process(es) listening on port $$port..."; \
		kill $$pids 2>/dev/null || true; \
		sleep 1; \
		remaining_pids=""; \
		for pid in $$pids; do \
			if kill -0 "$$pid" 2>/dev/null; then \
				remaining_pids="$$remaining_pids $$pid"; \
			fi; \
		done; \
		if [ -n "$$remaining_pids" ]; then \
			kill -9 $$remaining_pids 2>/dev/null || true; \
		fi; \
	fi; \
	for i in {1..10}; do \
		if command -v ss >/dev/null 2>&1; then \
			if ! ss -ltn "sport = :$$port" 2>/dev/null | grep -q ":$$port"; then \
				exit 0; \
			fi; \
		elif command -v lsof >/dev/null 2>&1; then \
			if ! lsof -nP -iTCP:"$$port" -sTCP:LISTEN >/dev/null 2>&1; then \
				exit 0; \
			fi; \
		else \
			exit 0; \
		fi; \
		sleep 0.5; \
	done; \
	echo "Ollama host port $$port is still in use."; \
	echo "If this is the system Ollama service, run: sudo systemctl stop ollama"; \
	exit 1

.PHONY: up-dev-ollama
up-dev-ollama:
	$(MAKE) up-dev-run LLM_PROVIDER=ollama

.PHONY: up-dev
up-dev:
	$(MAKE) up-dev-run LLM_PROVIDER=openai

.PHONY: up-dev-run
up-dev-run:
	@set -e; \
	mkdir -p data/runs data/evals data/chat $(DEV_PID_DIR); \
	rm -f data/dev_api.log data/dev_adk.log $(API_PID) $(ADK_PID); \
	api_pid=""; \
	adk_pid=""; \
	compose_pid=""; \
	serving_pid=""; \
	if [ "$(LLM_PROVIDER)" = "ollama" ] && [ "$(OLLAMA_DOCKER)" = "true" ]; then \
		$(MAKE) --no-print-directory free-ollama-port; \
	fi; \
	if [ "$(LLM_PROVIDER)" = "openai" ] && [ -z "$(OPENAI_API_KEY)" ]; then \
		echo "OPENAI_API_KEY is required for LLM_PROVIDER=openai. Use make up-dev-ollama for the local no-OpenAI demo."; \
		exit 1; \
	fi; \
	echo "Starting Docker services with attached logs..."; \
	docker compose up postgres clearml-elastic clearml-mongo clearml-redis clearml-fileserver clearml-apiserver clearml-webserver $(if $(and $(filter ollama,$(LLM_PROVIDER)),$(filter true,$(OLLAMA_DOCKER))),ollama,) & \
	compose_pid=$$!; \
	cleanup() { \
		status=$$?; \
		trap - INT TERM EXIT; \
		echo ""; \
		echo "Stopping dev stack..."; \
		[ -n "$$api_pid" ] && kill "$$api_pid" 2>/dev/null || true; \
		[ -n "$$adk_pid" ] && kill "$$adk_pid" 2>/dev/null || true; \
		[ -n "$$compose_pid" ] && kill "$$compose_pid" 2>/dev/null || true; \
		[ -n "$$serving_pid" ] && kill "$$serving_pid" 2>/dev/null || true; \
		wait "$$api_pid" "$$adk_pid" "$$compose_pid" "$$serving_pid" 2>/dev/null || true; \
		rm -f "$(API_PID)" "$(ADK_PID)"; \
		docker compose stop $(CLEARML_SERVING_SERVICES) postgres clearml-webserver clearml-apiserver clearml-fileserver clearml-redis clearml-mongo clearml-elastic $(if $(and $(filter ollama,$(LLM_PROVIDER)),$(filter true,$(OLLAMA_DOCKER))),ollama,) >/dev/null 2>&1 || true; \
		exit $$status; \
	}; \
	trap cleanup INT TERM EXIT; \
	echo "Waiting for PostgreSQL..."; \
	for i in {1..30}; do \
		if docker compose exec -T postgres pg_isready -U ml_chatbot -d ml_ecommerce_chatbot >/dev/null 2>&1; then \
			echo "PostgreSQL is ready."; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "PostgreSQL did not become ready in time."; \
			exit 1; \
		fi; \
		sleep 2; \
	done; \
	if [ "$(LLM_PROVIDER)" = "ollama" ] && [ "$(OLLAMA_SKIP_PULL)" != "true" ]; then \
		echo "Ensuring Ollama model $(OLLAMA_MODEL) is available..."; \
		for i in {1..30}; do \
			if curl -fsS "$(OLLAMA_BASE_URL)/api/tags" >/dev/null 2>&1; then \
				break; \
			fi; \
			if [ $$i -eq 30 ]; then \
				echo "Ollama API is not ready."; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
		OLLAMA_MODEL="$(OLLAMA_MODEL)" docker compose run --rm ollama-pull; \
	elif [ "$(LLM_PROVIDER)" = "ollama" ]; then \
		echo "Skipping Ollama pull for local model $(OLLAMA_MODEL)."; \
	fi; \
	echo "Waiting for ClearML API..."; \
	for i in {1..60}; do \
		if curl -fsS "http://localhost:8008/debug.ping" >/dev/null 2>&1; then \
			echo "ClearML API is ready."; \
			break; \
		fi; \
		if [ $$i -eq 60 ]; then \
			echo "ClearML API did not become ready in time."; \
			exit 1; \
		fi; \
		sleep 2; \
	done; \
	echo "Starting ClearML Serving on $(CLEARML_SERVING_BASE_URL)"; \
	$(MAKE) --no-print-directory clearml-serving-up & \
	serving_pid=$$!; \
	echo "Starting FastAPI prediction service on http://localhost:$(API_PORT)"; \
	ADK_WEB_URL="$(ADK_WEB_URL)" \
	DATABASE_URL="$(DATABASE_URL)" \
	poetry run python -m uvicorn interface.api.app:app --reload --host "$(API_HOST)" --port "$(API_PORT)" & \
	api_pid=$$!; \
	echo "$$api_pid" > "$(API_PID)"; \
	echo "Starting ADK Web chat on http://localhost:$(ADK_PORT) using $(LLM_PROVIDER)"; \
	API_URL="$(API_URL)" \
	LLM_PROVIDER="$(LLM_PROVIDER)" \
	OPENAI_MODEL="$(OPENAI_MODEL)" \
	OLLAMA_BASE_URL="$(OLLAMA_BASE_URL)" \
	OLLAMA_API_BASE="$(OLLAMA_BASE_URL)" \
	OLLAMA_MODEL="$(OLLAMA_MODEL)" \
	DATABASE_URL="$(DATABASE_URL)" \
	poetry run adk web "$(ADK_AGENTS_DIR)" --host 0.0.0.0 --port "$(ADK_PORT)" --session_service_uri "$(ADK_SESSION_SERVICE_URI)" & \
	adk_pid=$$!; \
	echo "$$adk_pid" > "$(ADK_PID)"; \
	echo ""; \
	echo "Dev stack is running in the foreground."; \
	echo "ADK Web:  http://localhost:$(ADK_PORT)"; \
	echo "FastAPI:  http://localhost:$(API_PORT)"; \
	echo "ClearML:  http://localhost:8080"; \
	echo "ClearML Serving: $(CLEARML_SERVING_BASE_URL)"; \
	echo "Press Ctrl-C to stop."; \
	wait -n "$$api_pid" "$$adk_pid" "$$compose_pid" "$$serving_pid"

.PHONY: down-dev
down-dev:
	@if [ -f "$(API_PID)" ]; then \
		pid="$$(cat $(API_PID))"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Stopping API ($$pid)"; \
			kill -TERM "-$$pid" 2>/dev/null || kill -TERM "$$pid" 2>/dev/null || true; \
		fi; \
		rm -f "$(API_PID)"; \
	fi
	@pkill -TERM -f "[u]vicorn interface.api.app:app" 2>/dev/null || true
	@if [ -f "$(ADK_PID)" ]; then \
		pid="$$(cat $(ADK_PID))"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Stopping ADK Web ($$pid)"; \
			kill -TERM "-$$pid" 2>/dev/null || kill -TERM "$$pid" 2>/dev/null || true; \
		fi; \
		rm -f "$(ADK_PID)"; \
	fi
	@pkill -TERM -f "[a]dk web $(ADK_AGENTS_DIR)" 2>/dev/null || true
	@pkill -TERM -f "[a]dk web interface" 2>/dev/null || true
	docker compose stop $(CLEARML_SERVING_SERVICES) postgres ollama clearml-webserver clearml-apiserver clearml-fileserver clearml-redis clearml-mongo clearml-elastic || true

.PHONY: ollama-pull
ollama-pull:
	docker compose up -d ollama
	OLLAMA_MODEL="$(OLLAMA_MODEL)" docker compose run --rm ollama-pull

.PHONY: api
api:
	poetry run python -m uvicorn interface.api.app:app --reload --host $(API_HOST) --port $(API_PORT)

.PHONY: adk-web
adk-web:
	API_URL=$(API_URL) \
	LLM_PROVIDER=$(LLM_PROVIDER) \
	OPENAI_MODEL=$(OPENAI_MODEL) \
	OLLAMA_BASE_URL=$(OLLAMA_BASE_URL) \
	OLLAMA_API_BASE=$(OLLAMA_BASE_URL) \
	OLLAMA_MODEL=$(OLLAMA_MODEL) \
	poetry run adk web $(ADK_AGENTS_DIR) --host 0.0.0.0 --port $(ADK_PORT) --session_service_uri "$(ADK_SESSION_SERVICE_URI)"

.PHONY: adk-api
adk-api:
	API_URL=$(API_URL) \
	LLM_PROVIDER=$(LLM_PROVIDER) \
	OPENAI_MODEL=$(OPENAI_MODEL) \
	OLLAMA_BASE_URL=$(OLLAMA_BASE_URL) \
	OLLAMA_API_BASE=$(OLLAMA_BASE_URL) \
	OLLAMA_MODEL=$(OLLAMA_MODEL) \
	poetry run adk api_server $(ADK_AGENTS_DIR) --host 0.0.0.0 --port $(ADK_PORT) --session_service_uri "$(ADK_SESSION_SERVICE_URI)"

.PHONY: train
train:
	poetry run train-model \
		--model $(MODEL) \
		--data-path $(DATA_PATH) \
		--max-evals $(MAX_EVALS) \
		--cv-folds $(CV_FOLDS) \
		$(if $(filter true,$(SMOTE)),--smote,--no-smote) \
		--smote-k-neighbors $(SMOTE_K_NEIGHBORS) \
		$(if $(filter true,$(CLEARML)),--clearml,)

.PHONY: finetune-ollama
finetune-ollama:
	@set -e; \
	echo "Starting PostgreSQL and ClearML services..."; \
	docker compose up -d postgres clearml-elastic clearml-mongo clearml-redis clearml-fileserver clearml-apiserver clearml-webserver; \
	echo "Waiting for PostgreSQL..."; \
	for i in {1..30}; do \
		if docker compose exec -T postgres pg_isready -U ml_chatbot -d ml_ecommerce_chatbot >/dev/null 2>&1; then \
			echo "PostgreSQL is ready."; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "PostgreSQL did not become ready in time."; \
			exit 1; \
		fi; \
		sleep 2; \
	done; \
	echo "Waiting for ClearML API..."; \
	for i in {1..60}; do \
		if curl -fsS "http://localhost:8008/debug.ping" >/dev/null 2>&1; then \
			echo "ClearML API is ready."; \
			break; \
		fi; \
		if [ $$i -eq 60 ]; then \
			echo "ClearML API did not become ready in time."; \
			exit 1; \
		fi; \
		sleep 2; \
	done; \
	cmd=(poetry run finetune \
		--provider ollama \
		--model "$(FINETUNE_OLLAMA_MODEL)" \
		--epochs "$(FINETUNE_EPOCHS)" \
		--batch-size "$(FINETUNE_BATCH_SIZE)" \
		--max-seq-length "$(FINETUNE_MAX_SEQ_LENGTH)"); \
	if [ "$(FINETUNE_CLEARML)" = "true" ]; then \
		cmd+=(--clearml); \
	else \
		cmd+=(--no-clearml); \
	fi; \
	if [ -n "$(FINETUNE_BASE_MODEL)" ]; then \
		cmd+=(--base-model-id "$(FINETUNE_BASE_MODEL)"); \
	fi; \
	if [ -n "$(FINETUNE_MAX_TRAIN_SAMPLES)" ]; then \
		cmd+=(--max-train-samples "$(FINETUNE_MAX_TRAIN_SAMPLES)"); \
	fi; \
	CLEARML_ENABLED="$(FINETUNE_CLEARML)" "$${cmd[@]}"

.PHONY: serve-finetuned-ollama
serve-finetuned-ollama:
	@adapter_path="$(ADAPTER_PATH)"; \
	if [ -z "$$adapter_path" ]; then \
		adapter_path="$(LLM_ADAPTER_PATH)"; \
	fi; \
	if [ -z "$$adapter_path" ]; then \
		echo "No adapter path specified and no latest adapter found in data/runs/llm_finetuning/"; \
		exit 1; \
	fi; \
	./infra/llm/serve_adapter.sh $(FINETUNE_OLLAMA_MODEL) "$$adapter_path"

.PHONY: clearml-register-latest-llm
clearml-register-latest-llm:
	@if [ -z "$(LLM_ADAPTER_PATH)" ]; then \
		echo "LLM_ADAPTER_PATH is required, for example data/runs/llm_finetuning/run_id/adapter"; \
		exit 1; \
		fi
		poetry run register-llm-adapter \
			--adapter-path "$(LLM_ADAPTER_PATH)" \
			--model "$(FINETUNE_OLLAMA_MODEL)" \
			$(if $(FINETUNE_BASE_MODEL),--base-model-id "$(FINETUNE_BASE_MODEL)",) \
			--clearml

.PHONY: finetune-azure-openai
finetune-azure-openai:
		@if [ -z "$(FINETUNE_AZURE_FILE_ID)" ]; then \
			echo "FINETUNE_AZURE_FILE_ID is required, for example file-abc123"; \
			exit 1; \
		fi
		poetry run finetune \
			--provider azure_openai \
			--model "$(AZURE_OPENAI_DEPLOYMENT)" \
			$(if $(AZURE_OPENAI_FINE_TUNE_MODEL),--base-model-id "$(AZURE_OPENAI_FINE_TUNE_MODEL)",) \
			--data-path "$(FINETUNE_AZURE_FILE_ID)" \
		--epochs "$(FINETUNE_EPOCHS)" \
		--clearml

.PHONY: finetune-bedrock
finetune-bedrock:
		@if [ -z "$(FINETUNE_BEDROCK_DATA_URI)" ]; then \
			echo "FINETUNE_BEDROCK_DATA_URI is required, for example s3://bucket/train.jsonl"; \
			exit 1; \
		fi
		poetry run finetune \
			--provider bedrock \
			--model "$(BEDROCK_MODEL_ID)" \
			--data-path "$(FINETUNE_BEDROCK_DATA_URI)" \
			--epochs "$(FINETUNE_EPOCHS)" \
		--clearml

.PHONY: finetune-vertex
finetune-vertex:
		@if [ -z "$(FINETUNE_VERTEX_DATA_URI)" ]; then \
			echo "FINETUNE_VERTEX_DATA_URI is required, for example gs://bucket/train.jsonl"; \
			exit 1; \
		fi
		poetry run finetune \
			--provider vertex \
			--model "$(VERTEX_MODEL_ID)" \
			--data-path "$(FINETUNE_VERTEX_DATA_URI)" \
			--epochs "$(FINETUNE_EPOCHS)" \
		--clearml

.PHONY: up-dev-ollama-finetuned
up-dev-ollama-finetuned:
	$(MAKE) up-dev-run LLM_PROVIDER=ollama OLLAMA_MODEL="$(FINETUNED_OLLAMA_MODEL)" OLLAMA_SKIP_PULL=true OLLAMA_DOCKER=false

.PHONY: adk-web-finetuned
adk-web-finetuned:
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL..."; \
	for i in $$(seq 1 30); do \
		if docker compose exec -T postgres pg_isready -U ml_chatbot -d ml_ecommerce_chatbot >/dev/null 2>&1; then \
			echo "PostgreSQL is ready."; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "PostgreSQL did not become ready in time."; \
			exit 1; \
		fi; \
		sleep 1; \
	done
	$(MAKE) adk-web LLM_PROVIDER=ollama OLLAMA_MODEL="$(FINETUNED_OLLAMA_MODEL)"

.PHONY: select-model
select-model:
	poetry run select-model \
		--models $(MODELS) \
		--data-path $(DATA_PATH) \
		--max-evals $(MAX_EVALS) \
		--cv-folds $(CV_FOLDS) \
		$(if $(filter true,$(SMOTE)),--smote,--no-smote) \
		--smote-k-neighbors $(SMOTE_K_NEIGHBORS) \
		$(if $(filter true,$(CLEARML)),--clearml,)

.PHONY: demo-train
demo-train:
	$(MAKE) train MODEL=randomforest MAX_EVALS=0 CLEARML=false

.PHONY: clean-clearml
clean-clearml:
	poetry run clean-clearml --force

.PHONY: clean-clearml-dry-run
clean-clearml-dry-run:
	poetry run clean-clearml --dry-run

.PHONY: model-selection
model-selection:
	$(MAKE) select-model CLEARML=true MAX_EVALS=1
	$(MAKE) clearml-serving-deploy-latest

.PHONY: clearml-serving-create
clearml-serving-create:
	@set -e; \
	mkdir -p data/clearml; \
	if [ -s "$(CLEARML_SERVING_TASK_FILE)" ]; then \
		echo "ClearML Serving service id: $$(cat $(CLEARML_SERVING_TASK_FILE))"; \
		exit 0; \
	fi; \
	if [ -s "$(CLEARML_SERVING_ENV_FILE)" ]; then \
		task_id="$$(sed -nE 's/^CLEARML_SERVING_TASK_ID=(.+)$$/\1/p' "$(CLEARML_SERVING_ENV_FILE)" | tail -1)"; \
		if [ -n "$$task_id" ]; then \
			echo "$$task_id" > "$(CLEARML_SERVING_TASK_FILE)"; \
			echo "ClearML Serving service id: $$task_id"; \
			exit 0; \
		fi; \
	fi; \
	echo "Creating ClearML Serving service $(CLEARML_SERVING_SERVICE_NAME)"; \
	output="$$(poetry run clearml-serving --yes create --name "$(CLEARML_SERVING_SERVICE_NAME)" --project "$(CLEARML_SERVING_PROJECT)" --tags ecommerce classical-ml serving 2>&1)"; \
	printf "%s\n" "$$output"; \
	task_id="$$(printf "%s\n" "$$output" | sed -nE 's/.*id=([a-fA-F0-9]{32}).*/\1/p' | tail -1)"; \
	if [ -z "$$task_id" ]; then \
		echo "Could not parse ClearML Serving task id from clearml-serving output."; \
		exit 1; \
	fi; \
	echo "$$task_id" > "$(CLEARML_SERVING_TASK_FILE)"; \
	printf "CLEARML_SERVING_TASK_ID=%s\n" "$$task_id" > "$(CLEARML_SERVING_ENV_FILE)"; \
	echo "Saved ClearML Serving task id to $(CLEARML_SERVING_TASK_FILE)"

.PHONY: clearml-serving-config
clearml-serving-config: clearml-serving-create
	@task_id="$$(cat $(CLEARML_SERVING_TASK_FILE))"; \
	poetry run clearml-serving --yes --id "$$task_id" config \
		--base-serving-url "$(CLEARML_SERVING_BASE_URL)" \
		--kafka-metric-server "$(CLEARML_SERVING_KAFKA_METRIC_SERVER)" \
		--metric-log-freq "$(CLEARML_SERVING_METRIC_LOG_FREQ)"

.PHONY: clearml-serving-up
clearml-serving-up: clearml-serving-config
	docker compose up --build $(CLEARML_SERVING_SERVICES)

.PHONY: clearml-serving-start
clearml-serving-start: clearml-serving-config
	docker compose up -d --build $(CLEARML_SERVING_SERVICES)

.PHONY: clearml-serving-deploy-latest
clearml-serving-deploy-latest: clearml-serving-start
	@set -e; \
	model_id="$$(poetry run python -c 'import json; from pathlib import Path; payload=json.loads(Path("data/runs/classical_ml/latest_selection.json").read_text()); model_id=payload["runs"][0].get("clearml_model_id"); assert model_id, "latest_selection.json does not contain clearml_model_id. Rerun model selection with CLEARML=true."; print(model_id)')"; \
	task_id="$$(cat $(CLEARML_SERVING_TASK_FILE))"; \
	docker compose exec clearml-serving-inference clearml-serving --yes --id "$$task_id" model add \
		--engine custom \
		--endpoint "$(CLEARML_SERVING_ENDPOINT)" \
		--model-id "$$model_id" \
		--preprocess infra/clearml/serving_preprocess.py; \
	docker compose exec clearml-serving-inference clearml-serving --yes --id "$$task_id" metrics add \
		--endpoint "$(CLEARML_SERVING_ENDPOINT)" \
		--variable-value latency; \
	echo "ClearML Serving endpoint: $(CLEARML_SERVING_BASE_URL)/$(CLEARML_SERVING_ENDPOINT)"

.PHONY: clearml-serving-deploy-latest-llm
clearml-serving-deploy-latest-llm: clearml-serving-start
	@set -e; \
	model_id="$$(poetry run python -c 'import json; from pathlib import Path; payload=json.loads(Path("data/runs/llm_finetuning/latest_finetune.json").read_text()); model_id=payload.get("clearml_model_id"); assert model_id, "latest_finetune.json does not contain clearml_model_id. Rerun make finetune-ollama FINETUNE_CLEARML=true."; print(model_id)')"; \
	task_id="$$(cat $(CLEARML_SERVING_TASK_FILE))"; \
	docker compose exec clearml-serving-inference clearml-serving --yes --id "$$task_id" model add \
		--engine custom \
		--endpoint "$(CLEARML_LLM_SERVING_ENDPOINT)" \
		--model-id "$$model_id" \
		--preprocess infra/clearml/llm_serving_preprocess.py; \
	docker compose exec clearml-serving-inference clearml-serving --yes --id "$$task_id" metrics add \
		--endpoint "$(CLEARML_LLM_SERVING_ENDPOINT)" \
		--variable-value latency; \
	echo "ClearML LLM Serving endpoint: $(CLEARML_SERVING_BASE_URL)/$(CLEARML_LLM_SERVING_ENDPOINT)"

.PHONY: clearml-serving-list
clearml-serving-list: clearml-serving-create
	@task_id="$$(cat $(CLEARML_SERVING_TASK_FILE))"; \
	poetry run clearml-serving --id "$$task_id" model list

.PHONY: model-selection-long
model-selection-long:
	$(MAKE) select-model MAX_EVALS=500

.PHONY: evaluate
evaluate:
	@if [ -z "$(MODEL_DIR)" ]; then echo "MODEL_DIR is required"; exit 1; fi
	poetry run evaluate --model-dir $(MODEL_DIR) --data-path $(DATA_PATH) --split $(SPLIT)

.PHONY: evaluate-selection
evaluate-selection:
	poetry run evaluate-selection --selection-path $(SELECTION_PATH) --data-path $(DATA_PATH) --split $(SPLIT)

.PHONY: predict
predict:
		@if [ -z "$(JSON)" ]; then echo "JSON is required"; exit 1; fi
		poetry run predict $(if $(MODEL_DIR),--model-dir $(MODEL_DIR),) --json '$(JSON)'

.PHONY: docker-up
docker-up:
	docker compose up -d

.PHONY: docker-down
docker-down:
	docker compose down

.PHONY: format
format:
	poetry run python -m black .

.PHONY: format-check
format-check:
	poetry run python -m black --check src interface evaluation tests

.PHONY: lint
lint:
	poetry run python -m prospector --profile-path .prospector.yaml

.PHONY: test tests
test:
	poetry run pytest tests

tests: test

.PHONY: test-e2e
test-e2e:
	poetry run pytest tests/e2e

.PHONY: terraform-plan
terraform-plan:
	@if [ -z "$(CLOUD)" ]; then echo "CLOUD is required: aws, azure, or gcp"; exit 1; fi
	terraform -chdir=infra/$(CLOUD) init
	terraform -chdir=infra/$(CLOUD) plan

.PHONY: terraform-apply
terraform-apply:
	@if [ -z "$(CLOUD)" ]; then echo "CLOUD is required: aws, azure, or gcp"; exit 1; fi
	terraform -chdir=infra/$(CLOUD) init
	terraform -chdir=infra/$(CLOUD) apply

.PHONY: terraform-destroy
terraform-destroy:
	@if [ -z "$(CLOUD)" ]; then echo "CLOUD is required: aws, azure, or gcp"; exit 1; fi
	terraform -chdir=infra/$(CLOUD) init
	terraform -chdir=infra/$(CLOUD) destroy
