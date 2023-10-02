# Define default compiler
CXX ?= g++

# Define default flags
CFLAGS ?= -std=c++17 -Ofast -Wall -Wpedantic -Wfatal-errors -shared -fPIC $(shell python -m pybind11 --includes)

# Get name of the library
OUTPUT := core$(shell python3-config --extension-suffix)

# Compile
$(OUTPUT): core.cpp *.hpp *.cpp
	$(CXX) $(CFLAGS) $< -o $@

# Clean
clean:
	-rm -f $(OUTPUT)
