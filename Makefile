# Grab all immediate subdirs under impl/
IMPL_SUBDIRS := $(wildcard impls/*)

# Optional architecture flag with a default
ARCH ?= sm_60

.PHONY: all $(IMPL_SUBDIRS)

# Main rule that builds everything
all: $(IMPL_SUBDIRS)

# For each impl/XYZ, run `make -C impl/XYZ ARCH=...`
$(IMPL_SUBDIRS):
	@echo "Building in $@ with ARCH=$(ARCH)"
	$(MAKE) -C $@ ARCH=$(ARCH)