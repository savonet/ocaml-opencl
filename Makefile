PROGNAME=ocaml-opencl
VERSION=0.1
DISTFILES= Makefile README \
	src/*.ml src/*.mli src/*.c src/OCamlMakefile src/Makefile \
	examples/convolve/*.ml examples/convolve/*Makefile* examples/convolve/*.cl

all clean doc:
	make -C src $@

dist:
	VERSION=$(VERSION); \
	mkdir $(PROGNAME)-$$VERSION; \
	cp -r --parents $(DISTFILES) $(PROGNAME)-$$VERSION; \
	tar zcvf $(PROGNAME)-$$VERSION.tar.gz $(PROGNAME)-$$VERSION; \
	rm -rf $(PROGNAME)-$$VERSION

.PHONY: dist
