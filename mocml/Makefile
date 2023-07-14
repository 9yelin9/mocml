CC = gcc
RM = rm -rf
USR_DIR = /home/9yelin9/.local
OMP_DIR = /opt/mpi/gcc-4.8.5/openmpi-4.1.0
CFLAGS = -g -O2 -Wall -mcmodel=medium -fPIC -I$(USR_DIR)/include -I$(OMP_DIR)/include -fopenmp
LDFLAGS = -shared -L$(USR_DIR)/lib -L$(OMP_DIR)/lib -fopenmp 
LINKS = -lz -lm -lopenblas
OBJS = b2d.o
TARGETS = b2d.so

.PHONY: all clean dep
.SUFFIXES : .c .o

.c .o :
	$(CC) $(CFLAGS) -c $<

all : $(TARGETS)
clean :
	$(RM) *.o
	$(RM) $(TARGET)
dep :
	$(CC) $(CFLAGS) -M $(OBJS:.o=.c) 

$(TARGETS) : $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGETS) $(OBJS)
