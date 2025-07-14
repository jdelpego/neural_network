CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

# Find all .cpp files in the current directory
SOURCES = $(wildcard *.cpp)
# Infer executable names from source files, e.g., neural_network_1.cpp -> neural_network_1
EXECUTABLES = $(SOURCES:.cpp=)
# Object files
OBJS = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(EXECUTABLES)

# Rule to build an executable from its source file
%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(EXECUTABLES) $(OBJS)