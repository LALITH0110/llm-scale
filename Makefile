.PHONY: setup-local setup-chameleon proto \
        exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 \
        analyze clean

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

exp4:
	python src/experiments/exp4_gpu_colocated.py

exp4-smoke:
	python src/experiments/exp4_gpu_colocated.py --smoke

exp5:
	python src/experiments/exp5_gpu_disagg.py

exp6:
	PREFILL_HOST=$(PREFILL_HOST) DECODE_HOSTS=$(DECODE_HOSTS) \
	python src/experiments/exp6_hybrid.py --phase a
	PREFILL_HOST=$(PREFILL_HOST) DECODE_HOSTS=$(DECODE_HOSTS) \
	python src/experiments/exp6_hybrid.py --phase b

exp6-smoke:
	python src/experiments/exp6_hybrid.py --smoke --phase a

exp7:
	PREFILL_HOST=$(PREFILL_HOST) DECODE_HOSTS=$(DECODE_HOSTS) \
	python src/experiments/exp7_reverse_hybrid.py

exp8:
	python src/experiments/exp8_gpu_batched.py

exp8-smoke:
	python src/experiments/exp8_gpu_batched.py --smoke

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
