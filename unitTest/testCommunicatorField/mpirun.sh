SCRIPT_DIR=$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)
# mpirun -n 4 ./build/testCommunicatorField 2 2 1 > test.o 2> test.e
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=FALSE
mpirun -n 1 --output-filename "$SCRIPT_DIR/build/logs/" "$SCRIPT_DIR/build/testCommunicatorField" 1 1 1 > "$SCRIPT_DIR/build/test.o" 2> "$SCRIPT_DIR/build/test.e"
