#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clap::clap" for configuration "Debug"
set_property(TARGET clap::clap APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(clap::clap PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libclap.so"
  IMPORTED_SONAME_DEBUG "libclap.so"
  )

list(APPEND _cmake_import_check_targets clap::clap )
list(APPEND _cmake_import_check_files_for_clap::clap "${_IMPORT_PREFIX}/lib/libclap.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
