SCRIPT_DIR=$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)
# mpirun -n 4 ./build/testCommunicatorField 2 2 1 > test.o 2> test.e
mpirun -n 2 --output-filename "$SCRIPT_DIR/build/logs/" "$SCRIPT_DIR/build/testCommunicatorField" 2 1 1 > "$SCRIPT_DIR/build/test.o" 2> "$SCRIPT_DIR/build/test.e"
