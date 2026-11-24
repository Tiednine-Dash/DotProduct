# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -O3 -Wall -Werror -std=c++26 -pthread -march=native -mavx -mfma
TESTFLAGS := $(CXXFLAGS) -DTESTING
# Linker
LINK := $(CXX)

# Libraries
LDLIBS := 

# Executables
EXEC := DotProduct
TEST := TestProduct

#############################################################
# Main program
#############################################################

$(EXEC) : $(EXEC).o
	$(LINK) $^ $(LDLIBS) -o $@

$(EXEC).o : $(EXEC).cc Timer.hpp
	$(CXX) $(CXXFLAGS) -c $(EXEC).cc

#############################################################
# Test program
#############################################################

$(TEST) : $(TEST).o DotProduct_test.o
	$(LINK) $^ $(LDLIBS) -o $@

$(TEST).o : $(TEST).cc
	$(CXX) $(TESTFLAGS) -c $(TEST).cc

DotProduct_test.o : DotProduct.cc Timer.hpp
	$(CXX) $(TESTFLAGS) -c DotProduct.cc -o DotProduct_test.o

#############################################################

.PHONY : clean

clean :
	@$(RM) $(EXEC) $(TEST)
	@$(RM) *.o *.d *~
	@echo "Cleanup complete."
