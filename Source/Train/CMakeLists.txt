add_executable(Train Train.cpp)
target_link_libraries(Train ${OpenCV_LIBS} ${LIBSVM_LIBRARY})
