SOURCES = openCL_stubs.c openCL.ml
RESULT = opencl
CLIBS = OpenCL

all: ncl

test: all test.ml
	$(OCAMLOPT) -I . opencl.cmxa test.ml -o test
	./test

include OCamlMakefile