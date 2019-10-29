CPP = g++
CFLAGS = -std=c++11 -lopencv_core -lopencv_imgproc

all: superres

superres.ex: main.cpp GP.o
	$(CPP) $(CFLAGS) -o superres.ex main.cpp GP.o

GP.o: GP.cpp GP.h
	$(CPP) $(CFLAGS) -c GP.cpp

clean:
	rm -f superres.ex GP.o 
