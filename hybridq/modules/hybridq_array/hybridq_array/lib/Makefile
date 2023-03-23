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
LOG2_PACK_SIZE ?= -1

# Get type and number of bits. Types are:
# 0: float
# 1: unsigned
# 2: signed
DOT_NBITS := $(shell echo $(dot_type) | sed 's/[a-z]//g')
DOT_TYPE := $(shell echo $(dot_type) | sed 's/[0-9]//g')
ifeq ($(DOT_TYPE), float)
	DOT_TYPE=0
else ifeq ($(DOT_TYPE), uint)
	DOT_TYPE=1
else ifeq ($(DOT_TYPE), int)
	DOT_TYPE=2
else
	DOT_TYPE=-1
endif

ifneq ($(dot_log2_pack_size),)
	LOG2_PACK_SIZE := $(dot_log2_pack_size)
endif

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

# Check support for float16
is_float16_supported := $(shell echo 'int main(){ _Float16 a; }' | \
                        $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) \
                           -x c++ - -o /dev/null 2>/dev/null && \
                        echo yes || \
                        echo no)

# Check support for float128
is_float128_supported := $(shell echo 'int main(){ __float128 a; }' | \
                         $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) \
                            -x c++ - -o /dev/null 2>/dev/null && \
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
ifeq ($(LOG2_PACK_SIZE), -1)
  LOG2_PACK_SIZE := $(shell \
                      if [ $(avx512) -eq 1 ]; then \
                        echo 4; \
                      elif [ $(avx2) -eq 1 ]; then \
                        echo 3; \
                      else \
                        echo 3; \
                      fi)
endif

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

	@# Print support for float16 and float128
	$(info # Float16? $(is_float16_supported))
	$(info # Float128? $(is_float128_supported))

# Get name of dot library
DOT_LIBNAME = $(OUTPUT_PATH)/hybridq_dot_$(dot_type)_$(LOG2_PACK_SIZE)_$(dot_npos).so

# Get name of swap library
SWAP_LIBNAME = $(OUTPUT_PATH)/hybridq_swap_$(swap_nbytes)_$(swap_npos).so

# Define dot
dot: $(DOT_LIBNAME)

# Define swap
swap: $(SWAP_LIBNAME)

# Define dot options
DOT_OPTIONS += -DHYBRIDQ_ARRAY_DOT_LOG2_PACK_SIZE=$(LOG2_PACK_SIZE)
DOT_OPTIONS += -DHYBRIDQ_ARRAY_DOT_ARRAY_TYPE=$(DOT_TYPE)
DOT_OPTIONS += -DHYBRIDQ_ARRAY_DOT_ARRAY_NBITS=$(DOT_NBITS)
DOT_OPTIONS += -DHYBRIDQ_ARRAY_DOT_NPOS=$(dot_npos)
ifeq ($(is_float16_supported), yes)
	DOT_OPTIONS += -DHYBRIDQ_ARRAY_FLOAT16=1
endif
ifeq ($(is_float128_supported), yes)
	DOT_OPTIONS += -DHYBRIDQ_ARRAY_FLOAT128=1
endif

# Compile dot
$(DOT_LIBNAME): include/dot.cpp include/dot.hpp include/type.hpp
	@# Print support for OpenMP
	$(info # Support OpenMP? $(is_openmp_supported))

	@# Print support for AVX
	$(info # Support AVX? $(shell (exit $(avx)) && echo no || echo yes))
	$(info # Support AVX-2? $(shell (exit $(avx2)) && echo no || echo yes))
	$(info # Support AVX-512? $(shell (exit $(avx512)) && echo no || echo yes))

	@# Print pack size
	$(info # Size of Pack: 2^$(LOG2_PACK_SIZE))

	@# Print support for float16 and float128
	$(info # Float16? $(is_float16_supported))
	$(info # Float128? $(is_float128_supported))

	@# Print array type
	$(info # Array type: $(dot_type))

	@# Compile
	$(COMPILE) $< -o $@ $(DOT_OPTIONS)


# Compile swap
$(SWAP_LIBNAME): include/swap.cpp include/swap.hpp
	@# Print support for OpenMP
	$(info # Support OpenMP? $(is_openmp_supported))

	@# Compile
	$(COMPILE) $< -o $@ \
		-DHYBRIDQ_ARRAY_SWAP_N_POS=$(swap_npos) \
		-DHYBRIDQ_ARRAY_SWAP_N_BYTES=$(swap_nbytes)

clean:
	-rm -f $(OUTPUT_PATH)/hybridq_dot_*.so
	-rm -f $(OUTPUT_PATH)/hybridq_swap_*.so
