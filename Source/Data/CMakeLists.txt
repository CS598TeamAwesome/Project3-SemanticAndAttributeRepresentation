add_executable(PartitionDataSet PartitionDataSet.cpp)
target_link_libraries(PartitionDataSet ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})

add_executable(BuildUniversalVocabulary BuildUniversalVocabulary.cpp)
target_link_libraries(BuildUniversalVocabulary ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})

add_executable(GenerateGridSearch GenerateGridSearch.cpp)

add_executable(InterpretResults InterpretResults.cpp)

add_executable(VisualizeCodebook VisualizeCodebook.cpp)
target_link_libraries(VisualizeCodebook ${P1_LIBRARY} ${P2_LIBRARY} ${OpenCV_LIBS})