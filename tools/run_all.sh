#!/usr/bin/env bash
# Run every benchmark binary sequentially on a single CPU, save results.
# Pinning to one core (default 0) and serial execution keeps measurements clean.

set -e

CPU=${CPU:-0}
ITERS=${ITERS:-100000}
SAMPLES=${SAMPLES:-31}
OUTDIR=${OUTDIR:-results/run_$(date +%Y%m%d_%H%M%S)}

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

run() {
    local name=$1
    local args=$2
    local out="$OUTDIR/${name}.txt"
    if [ ! -x "build/${name}" ]; then
        echo "skip ${name} (binary missing)"
        return
    fi
    echo "running ${name} ${args}"
    "./build/${name}" $args > "$out" 2>&1
    echo "  -> $out"
}

run uops_latency_linux "$CPU $ITERS $SAMPLES"
run uops_latency_pmc "$CPU $ITERS $SAMPLES"
run uops_throughput "$CPU $ITERS $SAMPLES"
run uops_op2op "$CPU $ITERS $SAMPLES"
run uops_vector "$CPU $ITERS $SAMPLES"
run uops_storefwd "$CPU $ITERS $SAMPLES"
run uops_memory "$CPU $SAMPLES"
run uops_generated "$CPU $ITERS $SAMPLES"

echo "all done. results in $OUTDIR"
