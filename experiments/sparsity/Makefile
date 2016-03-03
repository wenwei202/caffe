CXX = icpc
CXXFLAGS = -fopenmp
LDFLAGS = -LSpMP -lspmp -mkl

ifeq (yes, $(LIBXSMM))
  CXXFLAGS += -Ilibxsmm/include
  LDFLAGS += -Llibxsmm/lib -lxsmm
endif

ifeq (yes, $(DBG))
  CXXFLAGS += -O0 -g
else
  CXXFLAGS += -O3 -DNDEBUG -xHost -DMKL_DIRECT_CALL
endif

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

csrmm_test.s: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

clean:
	rm csrmm_test
