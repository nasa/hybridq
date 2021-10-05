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

CXX ?= g++
ARCH ?= native
CXXFLAGS ?= -Wall -Wpedantic -O3 -ffast-math -march=$(ARCH)
CXXFLAGS := $(CXXFLAGS) -std=c++17 -shared -fPIC
LOG2_PACK_SIZE ?= 3

.PHONY=all
all: hybridq/utils/hybridq_swap.so hybridq/utils/hybridq.so

hybridq/utils/hybridq_swap.so: include/python_swap.cpp include/swap.h include/utils.h include/pack.h
	$(CXX) $(CPPFLAGS) $(LDFLAGS) \
		$(CXXFLAGS) $(USE_OPENMP) \
		$(shell sh scripts/check_prerequisite.sh $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS)) \
		-DMAX_SWAP_SIZE=$(MAX_SWAP_SIZE) \
		$< -o $@

hybridq/utils/hybridq.so: include/python_U.cpp include/U.h include/utils.h include/pack.h
	$(CXX) $(CPPFLAGS) $(LDFLAGS) \
		$(CXXFLAGS) $(USE_OPENMP) \
		$(shell sh scripts/check_prerequisite.sh $(CXX) $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS)) \
		-DLOG2_PACK_SIZE=$(LOG2_PACK_SIZE) \
		-DLARGEST_GATE=$(LARGEST_GATE) \
		$< -o $@

clean:
	-rm -f hybridq/utils/hybridq.so
	-rm -f hybridq/utils/hybridq_swap.so
