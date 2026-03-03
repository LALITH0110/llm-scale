.PHONY: setup-local setup-chameleon proto exp1 exp2 exp3 analyze clean

setup-local:
	bash setup/install_local.sh

setup-chameleon:
	bash setup/install_chameleon.sh

download-local:
	bash setup/download_models_local.sh

download-full:
	bash setup/download_models_full.sh

proto:
	python -m grpc_tools.protoc \
		-I src/disaggregated/proto \
		--python_out=src/disaggregated \
		--grpc_python_out=src/disaggregated \
		src/disaggregated/proto/kvcache.proto

exp1:
	python src/experiments/exp1_colocated.py

exp2:
	python src/experiments/exp2_disaggregated.py

exp3:
	python src/experiments/exp3_hetero_quant.py

analyze:
	python src/analysis/plot_scaling.py
	python src/analysis/plot_comparison.py
	python src/analysis/plot_kv_overhead.py
	python src/analysis/cost_analysis.py

prefill-server:
	python src/disaggregated/prefill_server.py

decode-server:
	python src/disaggregated/decode_server.py

router:
	python src/disaggregated/router.py

clean:
	rm -f results/*.csv results/*.json
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
