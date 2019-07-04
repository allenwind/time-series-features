service=time-series-features
package = github.smartx.com/smartx/${service}
version = $(shell git describe --long --tags --dirty | awk '{print substr($$1,2)}')
build_dir = _build


.PHONY: pyclean
pyclean:
	find . -type d -name "__pycache__" | xargs rm -rf

.PHONY: benchmark
benchmark:
	python3 benchmark.py

.PHONY: utests
utests:
	python3 utests.py
