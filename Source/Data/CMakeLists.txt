add_executable(PartitionDataSet PartitionDataSet.cpp)
target_link_libraries(PartitionDataSet ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})

add_executable(BuildUniversalVocabulary BuildUniversalVocabulary.cpp)
target_link_libraries(BuildUniversalVocabulary ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})

add_executable(GenerateGridSearch GenerateGridSearch.cpp)