add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/googletest" "third_party/googletest")

mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

set_target_properties(gtest PROPERTIES FOLDER third_party)
set_target_properties(gtest_main PROPERTIES FOLDER third_party)
set_target_properties(gmock PROPERTIES FOLDER third_party)
set_target_properties(gmock_main PROPERTIES FOLDER third_party)

macro(package_add_test TESTNAME TESTFILE)
    add_executable(${TESTNAME} ${TESTFILE}) 
    target_link_libraries(${TESTNAME} gtest gmock gtest_main ${ARGN})
    target_compile_features(${TESTNAME} PRIVATE cxx_std_17)
    target_compile_options(${TESTNAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    gtest_discover_tests(${TESTNAME}
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

macro(package_add_test_with_libraries TESTNAME FILES LIBRARIES TEST_WORKING_DIRECTORY)
    add_executable(${TESTNAME} ${FILES})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main ${LIBRARIES})
    target_compile_features(${TESTNAME} PRIVATE cxx_std_17)
    target_compile_options(${TESTNAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    gtest_discover_tests(${TESTNAME}
        WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

package_add_test(test_compressed_sparse_row_matrix test_compressed_sparse_row_matrix.cpp Eigen3::Eigen)
package_add_test(test_sparse_jonker_volgenant_solver test_sparse_jonker_volgenant_solver.cpp Eigen3::Eigen)
package_add_test(test_common test_common.cpp)
