AM_CXXFLAGS = -I$(top_srcdir)/src/lib -DDATADIR="\"$(abs_top_srcdir)/test\"" @AM_CXXFLAGS@
check_PROGRAMS = unittest

EXTRA_DIST = \
utils.hpp \
diabetes.txt

unittest_SOURCES = \
densitas_error.cpp \
density_estimator.cpp \
math_make_classification_target.cpp \
math_quantile.cpp \
math_quantiles.cpp \
math_quantiles_weighted.cpp \
math_linspace.cpp \
math_centers.cpp \
manipulation_assign_vector_to_row.cpp \
manipulation_extract_row.cpp \
math_minimum.cpp \
manipulation_predict_proba_for_row.cpp \
vector_adaptor.cpp \
matrix_adaptor.cpp \
model_adaptor.cpp \
version.cpp \
real_world_with_liblinear.cpp

unittest_LDADD = $(top_builddir)/src/lib/.libs/libdensitas.a -lunittest -larmadillo -llinear $(AM_LDFLAGS)
