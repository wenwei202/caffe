CXX = icpc
CXXFLAGS = -fopenmp -I../../.. -ISpMP
LDFLAGS = -LSpMP -lspmp -mkl

#SEP = yes
ifeq (yes, $(SEP))
  CXXFLAGS += -I$(HOME)/sep/include -DSEP
  LDFLAGS += -L$(HOME)/sep/bin64 -lprog_api
endif

#VTUNE = yes
ifeq (yes, $(VTUNE))
  CXXFLAGS += -I/swtools/intel/vtune_amplifier_xe/include -DVTUNE
  LDFLAGS += -L/swtools/intel/vtune_amplifier_xe/lib64 -littnotify
endif

ifeq (yes, $(SNIPER))
  CXXFLAGS += -I$(SNIPER_ROOT)/include -DSNIPER -g
  LDFLAGS += -L$(SNIPER_ROOT)/lib -lsim_api
endif

ifeq (yes, $(SDE))
  CXXFLAGS += -DSDE -fasm-blocks
endif

ifeq (yes, $(LIBXSMM))
  CXXFLAGS += -Ilibxsmm/include
  LDFLAGS += -Llibxsmm/lib -lxsmm
endif

ifeq (yes, $(DBG))
  CXXFLAGS += -O0 -g
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  endif
else
  CXXFLAGS += -O3 -DNDEBUG -DMKL_DIRECT_CALL
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  else
    CXXFLAGS += -xHost
  endif
endif

conv1_test: conv1_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

conv1_test: conv1_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

csrmm_test.s: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

clean:
	rm csrmm_test
