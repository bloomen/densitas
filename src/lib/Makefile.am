# the library names to build
lib_LTLIBRARIES = libdensitas.la

libdensitas_la_LDFLAGS = -version-info 0:0:0 -no-undefined

# where to install the headers on the system
libdensitas_ladir = $(includedir)/densitas

# the list of header files that belong to the library (to be installed later)
libdensitas_la_HEADERS = \
densitas/all.hpp \
densitas/density_estimator.hpp \
densitas/densitas_error.hpp \
densitas/math.hpp \
densitas/model_adaptor.hpp \
densitas/matrix_adaptor.hpp \
densitas/vector_adaptor.hpp \
densitas/type_check.hpp \
densitas/manipulation.hpp \
densitas/version.hpp

# the sources to add to the library and to add to the source distribution
libdensitas_la_SOURCES = \
densitas_error.cpp \
version.cpp
