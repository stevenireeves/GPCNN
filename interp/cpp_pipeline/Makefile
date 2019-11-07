CPP = clang++
CFLAGS = -std=c++11 -ffast-math -I/usr/local/opt/lapack/include -L/usr/local/opt/lapack/lib -lblas -llapacke
PKG = -I /usr/local/include/opencv4/opencv -I /usr/local/include/opencv4 \
-L /usr/local/lib -lopencv_dnn -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video \
-lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

ifdef USE_OMP
	OMPFLAGS= -fopenmp
else
	OMPFLAGS=
endif
ifdef DEBUG
	DB = -g
else
	DB = -O3
endif
all: superres.ex

superres.ex: main.cpp GP.o weights.o
	$(CPP) $(CFLAGS) $(PKG) $(OMPFLAGS) $(DB) -o superres.ex main.cpp GP.o weights.o

weights.o: weights.cpp weights.h
	$(CPP) $(CFLAGS) $(OMPFLAGS) $(DB) -c weights.cpp
GP.o: GP.cpp GP.h weights.h
	$(CPP) $(CFLAGS) $(OMPFLAGS) $(DB) -c GP.cpp
clean:
	rm -f superres.ex GP.o weights.o 
