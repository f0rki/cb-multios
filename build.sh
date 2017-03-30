#!/usr/bin/env bash

set -e

# Root cb-multios directory
DIR=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
TOOLS="$DIR/tools"

# Install necessary python packages
if ! /usr/bin/env python -c "import yaml; import xlsxwriter" 2>&1 >/dev/null; then
    echo "Please install pyyaml and xlsxwriter" >&2
    echo "  $ sudo pip install pyyaml xlsxwriter" >&2
    exit 1
fi

echo "Running patcher"
${TOOLS}/cb_patcher.py $@

echo "Generating CMakelists"
${TOOLS}/makefiles.py

echo "Creating build directory"
mkdir -p ${DIR}/build
cd ${DIR}/build

echo "Creating Makefiles"
CMAKE_OPTS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# Honor CC and CXX environment variables
if [ -x "$CC" ]; then
  CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_C_COMPILER=$CC"
  CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_ASM_COMPILER=$CC"
fi
if [ -x "$CXX" ]; then
  CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_CXX_COMPILER=$CXX"
fi

LINK=${LINK:-SHARED}
case $LINK in
    SHARED) CMAKE_OPTS="$CMAKE_OPTS -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF";;
    STATIC) CMAKE_OPTS="$CMAKE_OPTS -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON";;
esac


if [[ $WITEVAL == 1 ]]; then
    echo "Patching CMakeLists.txt with no-vectorize CFLAGS"
    # add the following compiler flags to all the generated CMakeLists.txt files
    #    -fno-vectorize
    #    -fno-slp-vectorize
    # right after the -ON flag, so that it overrides the optimizer defaults
    for cmakelists in $(find ../processed-challenges -name CMakeLists.txt); do
        echo "$cmakelists"
        sed -i 's/\(-O[0-4sz]\)/\1 -fno-vectorize -fno-slp-vectorize/g' $cmakelists
    done
fi

# Prefer ninja over make, if it is available
if which ninja 2>&1 >/dev/null; then
  CMAKE_OPTS="-G Ninja $CMAKE_OPTS"
  BUILD_FLAGS="-- -v"
else
  BUILD_FLAGS="-- -j$(getconf _NPROCESSORS_ONLN)"
fi

cmake $CMAKE_OPTS ..

cmake --build . $BUILD_FLAGS
