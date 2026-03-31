# Makefile for C++ Kalman Filter benchmark
# Usage: make          (build optimized)
#        make debug    (build with debug symbols)
#        make clean    (remove binaries)
#        make run      (build + run)

CXX      = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++17
TARGET   = kalman
SRCS     = main.cpp kalman.cpp matrix.cpp

.PHONY: all clean debug run

all: $(TARGET)

$(TARGET): $(SRCS) kalman.h matrix.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

debug: CXXFLAGS = -g -Wall -Wextra -std=c++17 -fsanitize=address,undefined
debug: $(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe

run: $(TARGET)
	./$(TARGET)
