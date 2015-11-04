# the library names to build
lib_LTLIBRARIES = libdensitas.la

libdensitas_la_LDFLAGS = -version-info 4:0:1 -no-undefined

# where to install the headers on the system
libdensitas_ladir = $(includedir)/densitas

# the list of header files that belong to the library (to be installed later)
libdensitas_la_HEADERS = \
densitas/all.hpp \
densitas/density_estimator.hpp \
densitas/densitas_error.hpp \
densitas/math.hpp \
densitas/model_adapter.hpp \
densitas/matrix_adapter.hpp \
densitas/vector_adapter.hpp \
densitas/type_check.hpp \
densitas/manipulation.hpp \
densitas/task_manager.hpp \
densitas/version.hpp

# the sources to add to the library and to add to the source distribution
libdensitas_la_SOURCES = \
densitas_error.cpp \
task_manager.cpp \
version.cpp
