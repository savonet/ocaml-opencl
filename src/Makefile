SOURCES = openCL_stubs.c openCL.ml
RESULT = opencl
CLIBS = OpenCL
CFLAGS = -Wall -g

all: ncl

test: all test.ml
	$(OCAMLOPT) -I . opencl.cmxa test.ml -o test
	./test

env:
	source /etc/profile.d/ati-opencl.sh

include OCamlMakefile