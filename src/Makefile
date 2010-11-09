SOURCES = openCL_stubs.c openCL.ml
RESULT = opencl
CLIBS = OpenCL
CFLAGS = -Wall -g

all: ncl

t: test
	./test

test: all test.ml
	$(OCAMLOPT) -I . bigarray.cmxa opencl.cmxa test.ml -o test

gdb: test
	gdb ./test

env:
	source /etc/profile.d/ati-opencl.sh

include OCamlMakefile