CFLAGS = -std=c++11 -ffast-math -I/usr/local/opt/lapack/include -L/usr/local/opt/lapack/lib -lblas -llapacke
PKG = -I/usr/include/opencv4/opencv -I/usr/include/opencv4 -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core
ifdef DEBUG
	DB = -g
else
	DB = -O3
endif
ifdef PROFILE
	DB += -lprofiler
endif
ifdef USE_OMP
	OMPFLAGS= -fopenmp
else
	OMPFLAGS=
endif

platform =$(shell uname -s)

ifeq ($(platform),Darwin)
CPP = clang++
all: superres.ex

superres.ex: main.cpp GP.h weights.o
	$(CPP) $(CFLAGS) $(PKG) $(OMPFLAGS) $(DB) -o superres.ex main.cpp weights.o

weights.o: weights.cpp weights.h
	$(CPP) $(CFLAGS) $(OMPFLAGS) $(DB) -c weights.cpp
# GP.o: GP.cpp GP.h weights.h
# 	$(CPP) $(CFLAGS) $(OMPFLAGS) $(DB) -c GP.cpp

else ifeq ($(platform),Linux)
CPP = g++
all: superres.ex

superres.ex: main.cpp GP.h weights.o
	$(CPP) -o superres.ex main.cpp weights.o $(CFLAGS) $(PKG) $(OMPFLAGS) $(DB) 

weights.o: weights.cpp weights.h
	$(CPP) -c weights.cpp $(CFLAGS) $(OMPFLAGS) $(DB) 
#GP.o: GP.cpp GP.h weights.h
#	$(CPP)  -c GP.cpp $(CFLAGS) $(OMPFLAGS) $(DB)
endif 
clean:
	rm -f superres.ex GP.o weights.o 
