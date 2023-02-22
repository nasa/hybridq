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

# Set output path
OUTPUT_PATH ?= ./

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

# Check support for openMP
is_openmp_supported := $(shell echo 'int main(){}' | \
                       $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) \
                          -fopenmp -x c++ - -o /dev/null 2>/dev/null && \
                       echo yes || \
                       echo no)

# Update CXXFLAGS is OpenMP is supported
ifeq ($(is_openmp_supported), yes)
  CXXFLAGS += -fopenmp
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
     $(OUTPUT_PATH)/hybridq.so

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

$(OUTPUT_PATH)/hybridq.so: include/python_U.cpp \
                           include/U.h \
                           include/utils.h \
                           include/pack.h
	@# Compile
	$(COMPILE) $< -o $@ \
    -DLOG2_PACK_SIZE=$(LOG2_PACK_SIZE)

# Get name of swap library
SWAP_LIBNAME = $(OUTPUT_PATH)/hybridq_swap_$(swap_array_type)_$(swap_npos).so

# Get type for swap
ifeq ($(swap_array_type), float16)
	SWAP_ARRAY_TYPE = uint16_t
else ifeq ($(swap_array_type), float32)
	SWAP_ARRAY_TYPE = float
else ifeq ($(swap_array_type), float64)
	SWAP_ARRAY_TYPE = double
else ifeq ($(swap_array_type), float128)
	SWAP_ARRAY_TYPE = __float128
else ifeq ($(swap_array_type), int8)
	SWAP_ARRAY_TYPE = int8_t
else ifeq ($(swap_array_type), int16)
	SWAP_ARRAY_TYPE = int16_t
else ifeq ($(swap_array_type), int32)
	SWAP_ARRAY_TYPE = int32_t
else ifeq ($(swap_array_type), int64)
	SWAP_ARRAY_TYPE = int64_t
else ifeq ($(swap_array_type), uint8)
	SWAP_ARRAY_TYPE = uint8_t
else ifeq ($(swap_array_type), uint16)
	SWAP_ARRAY_TYPE = uint16_t
else ifeq ($(swap_array_type), uint32)
	SWAP_ARRAY_TYPE = uint32_t
else ifeq ($(swap_array_type), uint64)
	SWAP_ARRAY_TYPE = uint64_t
endif

# Define swap
swap: $(SWAP_LIBNAME)

# Compile swap
$(SWAP_LIBNAME): print_support \
					       include/swap.cpp
	$(COMPILE) $(filter-out $<, $^) -o $@ \
		-DHYBRIDQ_ARRAY_SWAP_N_POS=$(swap_npos) \
		-DHYBRIDQ_ARRAY_SWAP_ARRAY_TYPE=$(SWAP_ARRAY_TYPE)

clean:
	-rm -f $(OUTPUT_PATH)/hybridq.so
	-rm -f $(OUTPUT_PATH)/hybridq_swap_*.so
