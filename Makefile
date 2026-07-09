VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup deps broker broker-down train export-onnx inference agent simulate dashboard smoke test snap

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	cd edge-agent && go mod tidy

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

smoke:
	$(PY) scripts/smoke.py

test:
	$(PY) -m pytest
	cd edge-agent && go test ./...

snap:
	snapcraft pack --verbosity=brief
