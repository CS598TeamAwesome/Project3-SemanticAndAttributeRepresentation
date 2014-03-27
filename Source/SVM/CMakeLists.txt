add_executable(SvmTrainingInput SvmTrainingInput.cpp)
target_link_libraries(SvmTrainingInput ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})

add_executable(SvmTestingInput SvmTestingInput.cpp)
target_link_libraries(SvmTrainingInput ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})
