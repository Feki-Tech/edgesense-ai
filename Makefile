VENV := .venv
PY := $(VENV)/bin/python
MACHINES ?= 10

.PHONY: setup deps broker broker-down train export-onnx inference agent simulate dashboard mcp smoke test snap stack stack-down stack-logs stack-coap demo demo-offline demo-offline-coap eval benchmark fleet

stack:
	docker compose up -d --build

stack-coap:      ## stack with the CoAP uplink: agent -> coap-receiver -> cloud broker
	EDGESENSE_UPLINK_URL=coap://coap-receiver:5683 docker compose --profile coap up -d --build

stack-down:
	docker compose --profile coap down

stack-logs:
	docker compose logs -f --tail 50

demo:            ## live fault-injection demo against the running stack
	$(PY) scripts/demo.py

demo-offline:    ## uplink-outage / store-and-forward demo (stops+restarts cloud broker)
	$(PY) scripts/demo_offline.py

demo-offline-coap: ## same outage demo for the CoAP stack (stops+restarts the receiver)
	EDGESENSE_CLOUD_CONTAINER=edgesense-coap-receiver $(PY) scripts/demo_offline.py

eval:            ## offline model evaluation -> docs/EVALUATION.md
	$(PY) ml/evaluate.py --out docs/EVALUATION.md

benchmark:       ## public-dataset benchmark (downloads AI4I 2020 once) -> docs/BENCHMARK.md
	$(PY) ml/benchmark_public.py --out docs/BENCHMARK.md

fleet:           ## scale the simulated fleet, e.g. make fleet MACHINES=25
	EDGESENSE_MACHINES=$(MACHINES) docker compose up -d simulator

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	if command -v go >/dev/null 2>&1; then \
		cd edge-agent && go mod tidy; \
		cd ../coap-receiver && go mod tidy; \
	else \
		echo 'go not found — skipping Go deps (edge-agent, coap-receiver; only needed for `make agent` / `make test`); install Go 1.22+ e.g. `sudo apt install golang-go`'; \
	fi

broker:
	docker compose up -d mosquitto

broker-down:
	docker compose down

train:
	$(PY) ml/train.py

export-onnx:
	$(PY) ml/export_onnx.py

inference:
	$(PY) -m uvicorn inference.server:app --host 0.0.0.0 --port 8800

agent:
	cd edge-agent && go run .

simulate:
	$(PY) simulator/simulate.py --machines 3

dashboard:
	$(VENV)/bin/streamlit run dashboard/app.py

mcp:             ## MCP server on stdio (for local MCP clients)
	$(PY) mcp_server/server.py

smoke:
	$(PY) scripts/smoke.py

test:
	$(PY) -m pytest
	cd edge-agent && go test ./...
	cd coap-receiver && go test ./...

snap:
	snapcraft pack --verbosity=brief
