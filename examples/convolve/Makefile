SOURCES = convolve.ml
RESULT = convolve
INCDIRS = ../../src
LIBS = bigarray opencl

test: all
	./convolve

all: nc

env:
	echo "source /etc/profile.d/ati-opencl.sh"

include OCamlMakefile