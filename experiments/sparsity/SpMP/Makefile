LOADIMBA=0
LOADIMBA=${loadimba}

LIBRARY		= libspmp.a

CC		= icpc
CCFLAGS		= -openmp -std=c++11 -ipp -Wno-deprecated -Wall

ifeq (yes, $(DBG))
  CCFLAGS += -O0 -g
else
  CCFLAGS += -O3 -DNDEBUG
endif

ifeq (yes, $(XEON_PHI))
  CCFLAGS	+= -mmic
else
  CCFLAGS += -xHost
endif

ifeq (yes, $(MKL))
  CCFLAGS += -mkl -DMKL
endif

ifeq (${LOADIMBA}, 1)
  yesnolist += LOADIMBA
endif

DEFS += $(strip $(foreach var, $(yesnolist), $(if $(filter 1, $($(var))), -D$(var))))
DEFS += $(strip $(foreach var, $(deflist), $(if $($(var)), -D$(var)=$($(var)))))

CCFLAGS 	+= ${DEFS}
SRCS = $(wildcard *.cpp) $(wildcard reordering/*.cpp)
SYNK_SRCS = $(wildcard synk/*.cpp)
OBJS = $(SRCS:.cpp=.o) $(addsuffix .o, $(addprefix synk/, $(basename $(notdir $(SYNK_SRCS)))))

$(LIBRARY): $(OBJS)
	rm -f $(LIBRARY)
	ar qvs $(LIBRARY) $(OBJS) 

all: clean
	$(MAKE) $(LIBRARY)

test: test/gs_test test/reordering_test test/trsv_test

test/%: test/%.o $(LIBRARY)
	$(CC) $(CCFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CCFLAGS) -fPIC -c $< -o $@

clean:
	rm -f $(LIBRARY) $(OBJS) test/*.o
