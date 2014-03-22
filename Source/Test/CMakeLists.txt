add_executable(Test Test.cpp)
target_link_libraries(Test ${OpenCV_LIBS} ${LIBSVM_LIBRARY})
