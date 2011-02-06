PROGNAME=ocaml-opencl
VERSION=0.1
DISTFILES= Makefile README \
	src/*.ml src/*.mli src/*.c src/OCamlMakefile src/Makefile \
	examples/Makefile examples/OCamlMakefile \
	examples/convolve/*.ml examples/convolve/*Makefile* examples/convolve/*.cl \
	examples/mat_mult/*.ml examples/mat_mult/*Makefile* examples/mat_mult/*.cl \
	examples/mat_mult_fast/*.ml examples/mat_mult_fast/*Makefile* examples/mat_mult_fast/*.cl

all clean doc:
	make -C src $@

dist:
	VERSION=$(VERSION); \
	mkdir $(PROGNAME)-$$VERSION; \
	find $(DISTFILES) | cpio -pduv $(PROGNAME)-$$VERSION; \
	tar zcvf $(PROGNAME)-$$VERSION.tar.gz $(PROGNAME)-$$VERSION; \
	rm -rf $(PROGNAME)-$$VERSION

.PHONY: dist
