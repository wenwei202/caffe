CXX = icpc
CXXFLAGS = -fopenmp
LDFLAGS = -LSpMP -lspmp -mkl

ifeq (yes, $(DBG))
  CXXFLAGS += -O0 -g
else
  CXXFLAGS += -O3 -DNDEBUG -xHost
endif

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
