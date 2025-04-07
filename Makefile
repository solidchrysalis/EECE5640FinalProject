# High-level Makefile to build all CUDA implementations in impl/
# Usage: make all ARCH=sm_70

IMPL_DIRS := $(wildcard impl/*)

ARCH ?= sm_70

.PHONY: all clean $(IMPL_DIRS)

all: $(IMPL_DIRS)

$(IMPL_DIRS):
	$(MAKE) -C $@ ARCH=$(ARCH)

clean:
	for dir in $(IMPL_DIRS); do \
		$(MAKE) -C $$dir clean; \
	done
