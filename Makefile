# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
#
# Copyright © 2021, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
# 
# The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. 
# 
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Fix size of packs
LOG2_PACK_SIZE ?= 0

# Set default compiler
CXX ?= g++

# Set default architecture
ARCH ?= native

# Set default C++ flags
CXXFLAGS ?= -Wall \
            -Wpedantic \
            -Wno-vla \
            -Ofast \
            -ffast-math \
            -march=$(ARCH)

# Add flags for shared library
CXXFLAGS := $(CXXFLAGS) \
            -std=c++17 \
            -shared \
            -fPIC

# Add extra CXXFLAGS
CXXFLAGS += $(CXXFLAGS_EXTRA)

# Define OMP_FLAGS
OMP_FLAGS := -fopenmp

USE_MINIMAL_OMP ?= false
ifneq ($(USE_MINIMAL_OMP), false)
	ifeq ($(shell uname), Darwin)
		ifneq ($(shell $(CXX) --version | grep clang | wc -l), 0)
			# Get clang version
			CLANG_VERSION := $(shell $(CXX) --version | cut -d ' ' -f 4 | cut -d . -f 1)
	
			# Create temporary folder
			TMPDIR := $(shell mktemp -d)
	
			# Get right URL
			ifeq ($(CLANG_VERSION), 13)
				OMP_URL := https://mac.r-project.org/openmp/openmp-13.0.0-darwin21-Release.tar.gz
				OMP_SHA1 := 47af4cb0d1f3554969f2ec9dee450d728ea30024
			endif
			ifeq ($(CLANG_VERSION), 12)
				OMP_URL := https://mac.r-project.org/openmp/openmp-12.0.1-darwin20-Release.tar.gz
				OMP_SHA1 := 4fab53ccc420ab882119256470af15c210d19e5e
			endif
			ifeq ($(CLANG_VERSION), 11)
				OMP_URL := https://mac.r-project.org/openmp/openmp-11.0.1-darwin20-Release.tar.gz
				OMP_SHA1 := 0dcd19042f01c4f552914e2cf7a53186de397aa1
			endif
			ifeq ($(CLANG_VERSION), 10)
				OMP_URL := https://mac.r-project.org/openmp/openmp-10.0.0-darwin17-Release.tar.gz
				OMP_SHA1 := 9bf16a64ab747528c5de7005a1ea1a9e318b3cf0
			endif
			ifeq ($(CLANG_VERSION), 9)
				OMP_URL := https://mac.r-project.org/openmp/openmp-9.0.1-darwin17-Release.tar.gz
				OMP_SHA1 := e5bd8501a3f957b4babe27b0a266d4fa15dbc23f
			endif
			ifeq ($(CLANG_VERSION), 8)
				OMP_URL := https://mac.r-project.org/openmp/openmp-8.0.1-darwin17-Release.tar.gz
				OMP_SHA1 := e4612bfcb1bf520bf22844f7db764cadb7577c28
			endif
			ifeq ($(CLANG_VERSION), 7)
				OMP_URL := https://mac.r-project.org/openmp/openmp-7.1.0-darwin17-Release.tar.gz
				OMP_SHA1 := 6891ff6f83f2ed83eeed42160de819b50cf643cd
			endif
	
			# Download
			PHONY := $(shell curl -o $(TMPDIR)/openmp.tar.gz $(OMP_URL) --verbose)
	
			# Check
			ifneq ($(shell sha1sum $(TMPDIR)/openmp.tar.gz | cut -d ' ' -f 1), $(OMP_SHA1))
				$(error)
			endif
	
			# Extract
			PHONY := $(shell tar xvzf $(TMPDIR)/openmp.tar.gz -C $(TMPDIR) --strip-components=3 usr/local/lib/libomp.dylib)
	
			# Update flags
			OMP_FLAGS := -Xclang $(OMP_FLAGS)
			LDFLAGS += -L$(TMPDIR) -lomp
	
		endif
	endif
endif

# Check support for openMP
is_openmp_supported := $(shell echo | \
                       $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) \
                          $(OMP_FLAGS) -x c++ -c - -o /dev/null 2>/dev/null && \
                       echo yes || \
                       echo no)

# Update CXXFLAGS is OpenMP is supported
ifeq ($(is_openmp_supported), yes)
  CXXFLAGS += $(OMP_FLAGS)
endif

# Define command for compilation
COMPILE := $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS)

# Check support for AVX
avx := $(shell $(COMPILE) -x c++ -dM -E - < /dev/null | grep -i '__AVX__' | head -n 1 | wc -l)
avx2 := $(shell $(COMPILE) -x c++ -dM -E - < /dev/null | grep -i '__AVX2__' | head -n 1 | wc -l)
avx512 := $(shell $(COMPILE) -x c++ -dM -E - < /dev/null | grep -i '__AVX512' | head -n 1 | wc -l)

# If not specified, infer pack size
ifeq ($(LOG2_PACK_SIZE), 0)
  LOG2_PACK_SIZE := $(shell \
                      if [ $(avx512) -eq 1 ]; then \
                        echo 4; \
                      elif [ $(avx2) -eq 1 ]; then \
                        echo 3; \
                      else \
                        echo 3; \
                      fi)
endif

.PHONY=all
all: print_support \
     hybridq/utils/hybridq_swap.so \
     hybridq/utils/hybridq.so

.PHONY=print_support
print_support:
	@# Print support for OpenMP
	$(info # Support OpenMP? $(is_openmp_supported))

	@# Print support for AVX
	$(info # Support AVX? $(shell (exit $(avx)) && echo no || echo yes))
	$(info # Support AVX-2? $(shell (exit $(avx2)) && echo no || echo yes))
	$(info # Support AVX-512? $(shell (exit $(avx512)) && echo no || echo yes))

	@# Print pack size
	$(info # Size of Pack: 2^$(LOG2_PACK_SIZE))

hybridq/utils/hybridq_swap.so: include/python_swap.cpp \
                               include/swap.h \
                               include/utils.h \
                               include/pack.h
	@# Compile
	$(COMPILE) $< -o $@

hybridq/utils/hybridq.so: include/python_U.cpp \
                          include/U.h \
                          include/utils.h \
                          include/pack.h
	@# Compile
	$(COMPILE) $< -o $@ \
    -DLOG2_PACK_SIZE=$(LOG2_PACK_SIZE)

clean:
	-rm -f hybridq/utils/hybridq.so
	-rm -f hybridq/utils/hybridq_swap.so
