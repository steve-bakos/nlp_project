
set -e

mkdir -p tools

cd tools

if [ ! -d fast_align ]; then
    git clone https://github.com/clab/fast_align.git
fi

cd fast_align

if [ ! -f build/fast_align ]; then
    mkdir -p build
    cd build
    cmake ..
    make
    cd ..
fi

cd ../..