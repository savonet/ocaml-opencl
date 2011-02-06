SYSTEM := $(shell ocamlc -config 2>/dev/null | grep system | sed 's/system: //')

SOURCES = openCL_stubs.c openCL.ml
RESULT = opencl
ifeq ($(SYSTEM),"macosx")
  CFRAMEWORKS = OpenCL
else
  CLIBS = OpenCL
endif
CFLAGS = -Wall -g -I /usr/include/nvidia-current
OCAMLFLAGS = -g

all: ncl

include OCamlMakefile
