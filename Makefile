PY ?= python3
export PYTHONPATH := src

.PHONY: data test all clean phase4data bundle app docker docker-run snapshot snapshot-check rebuild review candidates benchmark sobol ahead limits phase5report audit

data:
	$(PY) -m atp.build

test:
	$(PY) -m pytest -q tests

all: data test

clean:
	rm -rf data/interim data/processed reports/data_card.md

phase1:
	$(PY) -m atp.phase1 validate && $(PY) -m atp.phase1 drift && $(PY) -m atp.phase1 step && $(PY) -m atp.phase1 stepcheck && $(PY) -m atp.phase1 final && $(PY) -m atp.phase1 export

export:
	$(PY) -m atp.phase1 export

phase3:
	$(PY) -m atp.phase3 world && $(PY) -m atp.phase3 refit && $(PY) -m atp.phase3 inputs && $(PY) -m atp.phase3 peak && $(PY) -m atp.phase3 engine && $(PY) -m atp.phase3 bands && $(PY) -m atp.phase3 sensitivity && $(PY) -m atp.phase3 validate && $(PY) -m atp.phase3 showcase && $(PY) -m atp.phase3 report

phase2:
	$(PY) -m atp.phase2 airline && $(PY) -m atp.phase2 checkins && $(PY) -m atp.phase2 lever && $(PY) -m atp.phase2 forecast && $(PY) -m atp.phase2 newroute && $(PY) -m atp.phase2 endtoend && $(PY) -m atp.phase2 guests && $(PY) -m atp.phase2 report

phase4data:
	$(PY) -m atp.phase4 data

bundle:
	$(PY) -m atp.phase4 bundle --out build/bundle

app:
	$(PY) -m atp.api

snapshot:
	$(PY) -m atp.phase5 snapshot

snapshot-check:
	$(PY) -m atp.phase5 check

rebuild:
	scripts/clean_rebuild.sh ../atp2026_rebuild python3.12

candidates:
	$(PY) -m atp.phase5 candidates && $(PY) -m atp.phase5 candidates-bc

review:
	$(PY) -m atp.phase5 review-inputs && $(PY) review/work/review.py && $(PY) -m atp.phase5 review

docker:
	docker build -t seats2stays .

docker-run: docker
	docker run --rm -p 8000:8000 seats2stays

benchmark:
	$(PY) -m atp.phase5 benchmark

sobol:
	$(PY) -m atp.phase5 sobol

ahead:
	$(PY) -m atp.phase5 ahead

limits:
	$(PY) -m atp.phase5 limits

audit:
	$(PY) scripts/audit_checks.py && $(PY) scripts/audit_diagnostics.py

phase5report:
	$(PY) -m atp.phase5 report
